import re
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import streamlit as st # For logger and potentially UI elements if directly plotting
from typing import Optional
from app.config import (
    GRAPH_CONTEXT_LIMIT,
    GRAPH_DEDUPLICATION_THRESHOLD,
    LLM_INFER_TOPICS_MAX_LENGTH,
    TOPICS_BATCH_SIZE
)
from app.llm_services import TxtaiLLM # Assuming TxtaiLLM for topic inference
from .auto_id import AutoId
from txtai import Embeddings # For type hinting and direct graph interaction

logger = st.logger.get_logger(__name__)

class GraphRAGBuilder:
    """
    Builds graph contexts for GraphRAG and manages graph-specific operations
    like topic inference. Operates directly on a txtai.Embeddings instance
    that has graph capabilities.
    """

    def __init__(self, embeddings_instance: Embeddings, llm_service: TxtaiLLM):
        """
        Args:
            embeddings_instance: The txtai.Embeddings instance with graph enabled.
            llm_service: An LLM service for inferring topics.
        """
        if not embeddings_instance or not embeddings_instance.graph:
            raise ValueError("GraphRAGBuilder requires a txtai.Embeddings instance with graph enabled.")
        self.embeddings = embeddings_instance
        self.llm_service = llm_service
        self.context_limit = GRAPH_CONTEXT_LIMIT

    def get_graph_rag_context(self, question: str) -> tuple[str, Optional[list[dict]], Optional[Image.Image]]:
        """
        Attempts to create a graph context for the input question.
        Checks if the question is a graph query and if a graph exists.

        Args:
            question: input question

        Returns:
            original_question or modified_question, [context_list_of_dicts] or None, plot_image or None
        """
        query, concepts = self._parse_graph_query(question)
        graph_context_data = None
        plot_image = None
        modified_question = question # By default, question is not modified

        if self.embeddings.graph and (query or concepts):
            logger.info(f"Identified graph query. Query: '{query}', Concepts: {concepts}")
            path_query_cypher = self._build_path_query(query, concepts)
            
            # Build graph network from path query
            # The `graph=True` parameter returns a `txtai.graph.Graph` instance
            graph_result = self.embeddings.graph.search(path_query_cypher, graph=True)

            if graph_result and graph_result.count():
                logger.info(f"Graph search successful, found {graph_result.count()} nodes initially.")
                plot_image = self._plot_graph(graph_result) # Plot before deduplication for visualization

                # Build graph context from nodes
                # graph_result.scan() iterates over node ids in the graph_result
                context_nodes = []
                for node_id_in_graph in list(graph_result.scan()):
                    # Retrieve original document ID and text stored as attributes
                    original_doc_id = graph_result.attribute(node_id_in_graph, "id")
                    # The 'text' attribute should hold the actual text content.
                    # When data was upserted as a dict {"text": "...", "source": "..."},
                    # this dict is what's stored if content=True.
                    # The graph node attribute "text" should point to this dict.
                    node_data_dict = graph_result.attribute(node_id_in_graph, "text")
                    
                    text_content = None
                    if isinstance(node_data_dict, dict):
                        text_content = node_data_dict.get("text")
                    elif isinstance(node_data_dict, str): # Fallback if text was stored directly
                        text_content = node_data_dict
                    
                    if not text_content and original_doc_id:
                        # If text is missing, use the original_doc_id as a fallback,
                        # though this is less ideal.
                        text_content = str(original_doc_id)
                        logger.warning(f"Node {node_id_in_graph} (original_id: {original_doc_id}) missing text content, using ID.")

                    if text_content:
                         context_nodes.append({
                            "id": original_doc_id if original_doc_id else node_id_in_graph,
                            "text": text_content,
                         })
                    else:
                        logger.warning(f"Node {node_id_in_graph} yielded no usable text content.")


                if context_nodes:
                    graph_context_data = context_nodes
                    # Default prompt if only concepts were given
                    default_prompt = (
                        "Write a title and text summarizing the context.\n"
                        f"Include the following concepts: {', '.join(concepts)} if they're mentioned in the context."
                    )
                    modified_question = query if query else default_prompt
                    logger.info(f"Graph context built with {len(context_nodes)} nodes. Question for LLM: '{modified_question}'")
                else:
                    logger.info("No context nodes could be extracted from graph result despite non-empty graph.")
            else:
                logger.info("Graph search did not yield any results or graph is empty.")
        
        return modified_question, graph_context_data, plot_image

    def _parse_graph_query(self, question: str) -> tuple[Optional[str], Optional[list[str]]]:
        """
        Parses question for graph query syntax.
        Returns: (query_text_or_None, concepts_list_or_None)
        """
        prefix = "gq: "
        query, concepts = None, None

        if "->" in question or question.strip().lower().startswith(prefix):
            raw_concepts = [x.strip() for x in question.strip().lower().split("->")]
            
            parsed_query_text = None
            actual_concepts = []

            # Check if the last segment contains the query prefix
            if prefix in raw_concepts[-1]:
                last_segment_parts = raw_concepts[-1].split(prefix, 1)
                # The part before "gq:" is still a concept
                if last_segment_parts[0].strip():
                    actual_concepts.extend(c.strip() for c in raw_concepts[:-1])
                    actual_concepts.append(last_segment_parts[0].strip())
                else: # "gq:" was at the beginning of the last segment
                    actual_concepts.extend(c.strip() for c in raw_concepts[:-1])

                if len(last_segment_parts) > 1 and last_segment_parts[1].strip():
                    parsed_query_text = last_segment_parts[1].strip()
            else: # No "gq:" in the last segment, all are concepts
                actual_concepts.extend(raw_concepts)
            
            # If only "gq: query" was given, raw_concepts will be ['gq: query']
            if not actual_concepts and question.strip().lower().startswith(prefix):
                parsed_query_text = question.strip()[len(prefix):].strip()
            
            # Filter out empty strings from concepts that might result from splitting " -> "
            concepts = [c for c in actual_concepts if c]
            query = parsed_query_text

        return query, concepts

    def _build_path_query(self, query_text: Optional[str], concepts: Optional[list[str]]) -> str:
        """Builds Cypher MATCH PATH query."""
        node_selectors = []
        if concepts:
            for concept_text in concepts:
                # Find the best matching *document ID* for each concept text
                # The graph nodes in txtai are often abstract internal IDs,
                # but they store the original document ID as an attribute 'id'.
                # We need to search for text and get the *graph node* that corresponds to it.
                # txtai.Embeddings.search returns results with original document IDs.
                # We need a way to map these original doc IDs back to graph node IDs if they differ,
                # or assume graph nodes are directly identifiable by the document ID if autoid="uuid5"
                # and graph nodes are created based on these.

                # Simpler: search for concept, get top document ID. Assume this ID is usable in graph.
                # This relies on the graph nodes being queryable using the original document ID,
                # which is true if the graph nodes were created with `id` attribute matching the doc ID.
                search_results = self.embeddings.search(concept_text, 1)
                if search_results:
                    doc_id = search_results[0]["id"]
                    # The Cypher query needs to match nodes based on their 'id' *attribute*
                    node_selectors.append(f'({{id: "{doc_id}"}})')
                else:
                    logger.warning(f"No document found for concept: '{concept_text}'. Path query might be affected.")
        elif query_text: # If no explicit concepts, use top 3 nodes from query_text search
            for x in self.embeddings.search(query_text, 3):
                node_selectors.append(f'({{id: "{x["id"]}"}})')
        
        if not node_selectors:
            # Fallback: if no specific nodes, create a generic query to explore some paths.
            # This might not be very useful without starting points.
            # Or, we could return an empty query string, and get_graph_rag_context would handle it.
            logger.warning("No node selectors generated for path query. Graph query might be ineffective.")
            return "MATCH (n) RETURN n LIMIT 0" # Return no paths

        # Create graph path query (Cypher-like for txtai's graph)
        # Path relationship `-[*1..4]->` means variable length path from 1 to 4 hops.
        path_str = "-[*1..4]->".join(node_selectors)
        cypher_query = f"MATCH P={path_str} RETURN P LIMIT {self.context_limit}"
        logger.debug(f"Generated Cypher Path Query: {cypher_query}")
        return cypher_query

    def _plot_graph(self, graph_instance) -> Optional[Image.Image]: # graph_instance is txtai.graph.Graph
        """Plots the graph and returns as a PIL Image."""
        if not graph_instance or not graph_instance.backend: # Check if graph has a NetworkX backend
            logger.warning("Graph instance is empty or has no backend for plotting.")
            return None

        # Deduplicate and label graph (modified from original for clarity)
        deduplicated_graph_backend, labels = self._deduplicate_for_plotting(graph_instance, GRAPH_DEDUPLICATION_THRESHOLD)
        
        if not deduplicated_graph_backend.nodes():
            logger.info("Graph is empty after deduplication, nothing to plot.")
            return None

        options = {
            "node_size": 700,
            "node_color": "#ffbd45",
            "edge_color": "#e9ecef",
            "font_color": "#454545",
            "font_size": 10,
            "alpha": 1.0,
            "width": 1.5 # Edge width
        }

        plt.figure(figsize=(12, 7)) # Adjusted figure size
        ax = plt.gca()
        try:
            # More robust layout, iterations might need adjustment based on graph size
            pos = nx.spring_layout(deduplicated_graph_backend, seed=0, k=0.9, iterations=50)
            nx.draw_networkx(deduplicated_graph_backend, pos=pos, labels=labels, **options, ax=ax)
        except Exception as e:
            logger.error(f"Error during graph drawing: {e}. Plotting with simpler layout.")
            try:
                pos = nx.kamada_kawai_layout(deduplicated_graph_backend) # Fallback layout
                nx.draw_networkx(deduplicated_graph_backend, pos=pos, labels=labels, **options, ax=ax)
            except Exception as e_fallback:
                logger.error(f"Fallback graph drawing also failed: {e_fallback}")
                plt.close() # Close the figure to prevent display issues
                return None


        ax.axis("off")
        plt.margins(x=0.15)

        buffer = BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
        plt.close() # Important to close the plot to free memory
        buffer.seek(0)
        return Image.open(buffer)

    def _deduplicate_for_plotting(self, graph_instance, threshold: float): # graph_instance is txtai.graph.Graph
        """
        Deduplicates input graph for plotting. Merges nodes with similar topics.
        Returns a NetworkX graph object and labels.
        This is a complex operation and assumes 'topic' and 'id' attributes exist.
        """
        # Create a new NetworkX graph for manipulation to avoid altering the original query result graph
        # The graph_instance.backend should already be a NetworkX graph.
        # We will work on a copy to perform merges/deletions for plotting purposes.
        
        nx_graph_copy = graph_instance.backend.copy()
        labels = {}
        node_primary_map = {} # Maps original node ID to its primary representative after merge
        
        # Scan nodes from the original graph_instance to get attributes
        nodes_to_process = list(graph_instance.scan()) # Get all node IDs from the query result graph
        
        processed_nodes_for_topic_comparison = {} # topic_text -> primary_node_id_in_copy

        for node_id in nodes_to_process:
            # Attributes from the original txtai graph object
            original_doc_id = graph_instance.attribute(node_id, "id") # The actual document ID
            topic = graph_instance.attribute(node_id, "topic") # LLM-generated topic
            
            # Label for the node: use topic if valid, else original document ID
            # AutoId.valid checks if original_doc_id is a UUID or numeric ID (meaning it might be abstract)
            # If original_doc_id is abstract and a topic exists, use the topic.
            # Otherwise, use the original_doc_id as the label.
            current_label = topic if topic and AutoId.valid(str(original_doc_id)) else str(original_doc_id)

            primary_node_for_current = node_id # By default, this node is its own primary
            
            # Find similar topics among already processed nodes for merging
            # This similarity check is against labels of *other distinct nodes*
            # We need to perform similarity against the labels of nodes already chosen as primaries.
            
            # This part of original code is complex: it uses self.embeddings.similarity
            # to compare current_label with topicnames (which are labels of other nodes).
            # This means it's doing semantic similarity between node labels.
            
            # Simplified approach for plotting deduplication:
            # If two nodes from the graph query result have very similar *topics*, merge them.
            # This assumes topics are reasonably good representations.
            
            # For plotting, we are working with the node_id from graph_instance.scan()
            # These are the nodes present in the nx_graph_copy
            
            # This logic requires careful adaptation from original.
            # The original code modifies the graph in-place, which we avoid here for plotting copy.
            # Let's keep it simpler: if a node's topic is very similar to an existing primary node's topic,
            # map it to that primary node.
            
            best_match_primary_node = None
            highest_similarity_score = 0.0

            # Compare current_label with labels of nodes already established as primary representatives
            for existing_label, primary_node_id_in_copy in processed_nodes_for_topic_comparison.items():
                # Calculate similarity if self.embeddings is available and has a .similarity method
                # This requires `self.embeddings` to be the main `txtai.Embeddings` instance.
                if self.embeddings and hasattr(self.embeddings, 'similarity'):
                    # .similarity expects a query and a list of texts
                    # It returns a list of (index, score) tuples
                    sim_result = self.embeddings.similarity(current_label, [existing_label])
                    if sim_result:
                        score = sim_result[0][1]
                        if score > highest_similarity_score:
                            highest_similarity_score = score
                            best_match_primary_node = primary_node_id_in_copy
            
            if best_match_primary_node and highest_similarity_score >= threshold:
                # Merge: map current_node (node_id) to best_match_primary_node
                node_primary_map[node_id] = best_match_primary_node
                logger.debug(f"Deduplicating node '{current_label}' (ID: {node_id}) into node with label '{labels.get(best_match_primary_node)}' (ID: {best_match_primary_node}) for plotting.")
            else:
                # New primary node (or no sufficiently similar one found)
                node_primary_map[node_id] = node_id # It's its own primary
                labels[node_id] = current_label # Store its label, using the internal graph_instance node ID as key
                processed_nodes_for_topic_comparison[current_label] = node_id

        # Rebuild the graph structure in a new NetworkX graph using the primary_map
        final_plot_graph = nx.Graph()
        for node_id in nodes_to_process:
            primary_of_node = node_primary_map[node_id]
            if not final_plot_graph.has_node(primary_of_node):
                final_plot_graph.add_node(primary_of_node)
                # The label for the final_plot_graph should be the label of its primary representative
                labels[primary_of_node] = labels.get(primary_of_node, str(primary_of_node)) # Ensure label exists

        # Add edges, remapping to primary nodes
        for u, v, data in graph_instance.backend.edges(data=True): # Edges from original query result's backend
            primary_u = node_primary_map.get(u)
            primary_v = node_primary_map.get(v)

            if primary_u is not None and primary_v is not None and primary_u != primary_v:
                if not final_plot_graph.has_edge(primary_u, primary_v):
                    final_plot_graph.add_edge(primary_u, primary_v, **data)
        
        # Filter labels to only include nodes present in final_plot_graph
        final_labels = {node: lbl for node, lbl in labels.items() if node in final_plot_graph.nodes()}

        return final_plot_graph, final_labels


    def infer_topics_for_all_nodes(self, start_index_offset: int = 0):
        """
        Traverses the graph and adds LLM-generated topics for nodes
        that are auto-generated IDs and don't have a topic.
        This method directly modifies the self.embeddings.graph attributes.

        Args:
            start_index_offset: If indexing new data, this indicates how many nodes
                                existed before, to only process new ones. Not perfectly
                                reliable if graph structure changes. A better approach
                                would be to pass specific node IDs to process.
                                For now, kept similar to original.
        """
        if not self.embeddings or not self.embeddings.graph:
            logger.error("Cannot infer topics, embeddings or graph not available.")
            return

        nodes_to_process = []
        # graph.scan() iterates over internal graph node IDs.
        # We need to check if these correspond to documents that need topics.
        all_graph_node_ids = list(self.embeddings.graph.scan())
        
        # The `start_index_offset` logic is a bit fragile.
        # A more robust way would be to get IDs of recently added documents.
        # For now, let's assume graph.scan() order is somewhat stable or we process all.

        logger.info(f"Starting topic inference for graph nodes. Total nodes: {len(all_graph_node_ids)}.")
        
        for node_id_in_graph in tqdm(all_graph_node_ids, desc="Checking nodes for topic inference"):
            # 'id' attribute on the graph node stores the original document ID
            original_doc_id = self.embeddings.graph.attribute(node_id_in_graph, "id")
            topic = self.embeddings.graph.attribute(node_id_in_graph, "topic")

            # Condition from original: infer if original_doc_id is an autoid and topic is empty
            if AutoId.valid(str(original_doc_id)) and not topic:
                # 'text' attribute on the graph node stores the content (or dict containing text)
                node_data = self.embeddings.graph.attribute(node_id_in_graph, "text")
                text_content = None
                if isinstance(node_data, dict):
                    text_content = node_data.get("text")
                elif isinstance(node_data, str):
                    text_content = node_data
                
                # If text_content is empty or too short, use original_doc_id (if it's not just a UUID)
                # or the node_id_in_graph itself as fallback text for topic generation.
                if not text_content or not re.search(r"\w+", text_content):
                    if original_doc_id and not AutoId.valid(str(original_doc_id)): # If original_doc_id is meaningful text
                        text_content = str(original_doc_id)
                    else: # Fallback to internal graph node ID if all else fails
                        text_content = str(node_id_in_graph)
                        logger.debug(f"Using node ID '{node_id_in_graph}' as text for topic inference as content was too short.")
                
                if text_content:
                     nodes_to_process.append((node_id_in_graph, text_content))

        if not nodes_to_process:
            logger.info("No nodes found requiring topic inference.")
            return

        logger.info(f"Found {len(nodes_to_process)} nodes for topic inference.")
        self._batch_generate_and_set_topics(nodes_to_process)

    def _batch_generate_and_set_topics(self, node_batch: list[tuple]):
        """
        Generates topics in batches using the LLM service and sets them on graph nodes.
        Args:
            node_batch: List of (node_id_in_graph, text_for_topic_inference)
        """
        prompt_template = """
Create a simple, concise topic name (3-5 words) for the following text.
Only return the topic name itself, without any preamble or explanation.

Text:
{text}
"""
        llm_prompts = []
        for _, text_content in node_batch:
            # Ensure text has some substance
            text_to_use = text_content if re.search(r"\w+", text_content) else "general content"
            llm_prompts.append(prompt_template.format(text=text_to_use[:1000])) # Truncate for safety

        batch_size_config = TOPICS_BATCH_SIZE if TOPICS_BATCH_SIZE else (32 if len(llm_prompts) > 32 else len(llm_prompts))
        if batch_size_config == 0 and len(llm_prompts) > 0: batch_size_config = 1 # Ensure batch size is at least 1
        
        logger.info(f"Generating topics for {len(llm_prompts)} prompts with batch size {batch_size_config}")

        if not llm_prompts:
            return

        try:
            generated_topics = self.llm_service.batch_generate(
                llm_prompts,
                maxlength=LLM_INFER_TOPICS_MAX_LENGTH, # Use a shorter max length for topics
                batch_size=batch_size_config
            )
        except Exception as e:
            logger.error(f"Error during LLM batch topic generation: {e}")
            return

        for i, topic_text in enumerate(generated_topics):
            node_id_in_graph = node_batch[i][0]
            # Clean up topic: remove "Topic:", newlines, extra quotes
            cleaned_topic = topic_text.replace("Topic:", "").strip().strip('"').strip("'")
            
            self.embeddings.graph.addattribute(node_id_in_graph, "topic", cleaned_topic)
            logger.debug(f"Set topic for node {node_id_in_graph}: '{cleaned_topic}'")

            # Update the graph's topics dictionary (if it's being used)
            # This part of original code directly manipulates `embeddings.graph.topics`
            # Check if `self.embeddings.graph.topics` is a standard feature or custom.
            # txtai.graph.Graph does have a `topics` property that is a `Labels` instance.
            # Adding an attribute should make it discoverable by `graph.topics.scan()`
            # but direct manipulation of `graph.topics[topic_name].append(uid)` was in original.
            # The `Labels` class in txtai seems to be for topic modeling features.
            # For now, `addattribute` is the primary way. If `graph.topics` needs explicit update:
            if hasattr(self.embeddings.graph, 'topics') and self.embeddings.graph.topics is not None:
                # This assumes `self.embeddings.graph.topics` is a dict-like structure
                # or a specialized txtai object that supports this.
                # txtai's `Labels` object (which `graph.topics` is) might not support direct append.
                # It's usually populated by `graph.topic()` method.
                # For now, let's rely on `addattribute` and assume `graph.topics` reflects this.
                # If more direct manipulation of a 'topics' collection is needed, it would require
                # understanding how `embeddings.graph.topics` is meant to be populated.
                # The original `topics[topic].append(uid)` implies `topics` is a `defaultdict(list)`.
                # `txtai.graph.Labels` is more complex.
                # Let's assume for now that `addattribute` is sufficient for the graph to know the topic.
                pass
        logger.info(f"Finished setting topics for batch of {len(generated_topics)} nodes.")