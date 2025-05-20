import streamlit as st
from app.orchestrator import AppOrchestrator
from PIL.Image import Image as PILImage # Already there
import json # For displaying JSON configs
import pandas as pd # For nicer tables
from io import BytesIO # For file uploads
from app.config import PARSER_MAPPING, DEFAULT_CHUNK_SETTINGS, DEFAULT_CHUNKING_STRATEGY # For dropdowns

# Initialize logger for UI specific messages if needed
logger = st.logger.get_logger(__name__)

@st.cache_resource(show_spinner="Initializing application, models, and database...")
def get_orchestrator():
    logger.info("Creating/Retrieving AppOrchestrator instance.")
    return AppOrchestrator()

class StreamlitAppUIRefactored:
    def __init__(self, orchestrator: AppOrchestrator):
        self.orchestrator = orchestrator
        self.app_info = self.orchestrator.get_app_info() # Initial fetch

        if "selected_kb_id" not in st.session_state:
            st.session_state.selected_kb_id = self.app_info.get("default_kb_id")
        if "chat_messages" not in st.session_state: # Renamed from 'messages'
            st.session_state.chat_messages = [{"role": "assistant", "content": self._build_welcome_message()}]

    def _refresh_app_info(self):
        """Call this after operations that change system state, like adding docs/KBs."""
        st.session_state.app_info = self.orchestrator.get_app_info() # Update session state cache
        self.app_info = st.session_state.app_info


    def _build_welcome_message(self) -> str:
        # ... (similar to before, but remove # command, guide to Upload tab)
        welcome = f"Welcome to {self.app_info['title']}!\n\n"
        welcome += "Ask a question in the chat, or explore Knowledge Bases and Documents in the other tabs.\n"
        if self.orchestrator.vector_store.count() == 0: # Check actual vector store
            welcome += "**The vector knowledge base is currently empty.** Use the 'Knowledge Bases' tab to create a KB and then 'Documents' tab to upload files.\n\n"
        
        if self.app_info.get('supports_graph_rag', False):
            welcome += "This application also supports `📈 GraphRAG` queries. Examples:\n"
            for ex in self.app_info.get('examples', []):
                if "gq:" in ex or "->" in ex:
                    welcome += f"- `{ex}`\n"
        welcome += "\nUse the 'System Info' tab to see current configuration."
        return welcome

    def render_chat_interface(self):
        st.header("💬 Chat with your Knowledge")

        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "image" in msg and msg["image"]:
                    st.image(msg["image"], caption=msg.get("image_caption", "Query Result Graph"))
                if "sources" in msg and msg["sources"]:
                    with st.expander("View Sources"):
                        for i, source_chunk in enumerate(msg["sources"]):
                            # Assuming source_chunk is a dict with "text" and "metadata"
                            st.markdown(f"**Source [{i+1}]** (Score: {source_chunk.get('score', 'N/A'):.2f})")
                            if source_chunk.get('metadata'):
                                st.json(source_chunk['metadata'], expanded=False) # Show metadata
                            st.markdown(f"> {source_chunk.get('text', 'N/A')}")
                            st.divider()


        if user_query := st.chat_input("Ask your question..."):
            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Pass selected_kb_id for filtering if implemented later
                        query_result = self.orchestrator.process_query(
                            user_query, 
                            kb_id_filter=None, # Add st.session_state.selected_kb_id if filtering is ready
                            stream_response=True
                        )
                        
                        response_content = ""
                        response_image = query_result.get("graph_image")
                        
                        # TODO: Adapt this part when citation (M10) is fully done.
                        # For now, `query_result["answer"]` is the LLM text stream.
                        # We need a way for `process_query` to also return the retrieved chunks
                        # if we want to display them as sources here.
                        # Let's assume `process_query` is modified to return:
                        # {"answer": stream, "type": ..., "graph_image": ..., "retrieved_chunks": [...]}

                        if isinstance(query_result["answer"], type((lambda: (yield))())): # Check if generator
                            streamed_text = st.write_stream(query_result["answer"])
                            response_content = streamed_text
                        else:
                            response_content = query_result["answer"]
                            st.markdown(response_content)
                        
                        if response_image:
                            st.image(response_image, caption="Query Result Graph")
                        
                        # Placeholder for displaying sources - requires orchestrator to return them
                        retrieved_chunks_for_display = query_result.get("retrieved_chunks_for_display", [])


                    except Exception as e:
                        logger.error(f"Error processing query '{user_query}': {e}", exc_info=True)
                        response_content = f"😕 Apologies, I encountered an error: {e}"
                        st.error(response_content)
                
                assistant_msg = {"role": "assistant", "content": response_content}
                if response_image:
                    assistant_msg["image"] = response_image
                if retrieved_chunks_for_display: # Check if the list is not empty
                    # Store only a few for display to keep session state light, e.g., top 3-5
                    assistant_msg["sources"] = retrieved_chunks_for_display[:5] 
                st.session_state.chat_messages.append(assistant_msg)
                st.rerun()


# And the display loop for messages:
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "image" in msg and msg["image"]:
                    st.image(msg["image"], caption=msg.get("image_caption", "Query Result Graph"))
                if "sources" in msg and msg["sources"]: # This is where it's displayed
                    with st.expander("View Context Sources"): # Changed label for clarity
                        for i, source_chunk_info in enumerate(msg["sources"]):
                            # source_chunk_info is now a dict like:
                            # {"text": "chunk_text", "metadata": {all_metadata...}, "score": 0.85}
                            st.markdown(f"**Context Item [{i+1}]** (Score: {source_chunk_info.get('score', 'N/A'):.2f})")
                            
                            # Display key metadata using format_metadata_for_llm_context
                            # or pick specific fields from source_chunk_info['metadata']
                            display_meta_str = format_metadata_for_llm_context(source_chunk_info.get('metadata', {}))
                            if display_meta_str:
                                st.caption(display_meta_str.strip()) # Display as caption

                            st.markdown(f"> {source_chunk_info.get('text', 'N/A')}")

                            # Optionally, show full raw metadata for debugging/power users
                            if st.button(f"Toggle Raw Metadata for Item {i+1}", key=f"toggle_meta_{msg['content'][:10]}_{i}"): # Unique key needed
                                st.json(source_chunk_info.get('metadata', {}), expanded=True)
                            st.divider()

    def render_kb_management(self):
        st.header("📚 Knowledge Base Management")
        
        kbs_data = self.orchestrator.list_knowledge_bases_detailed() # Orchestrator needs this method
        if kbs_data:
            df_kbs = pd.DataFrame(kbs_data)
            st.dataframe(df_kbs[['id', 'name', 'description', 'doc_count', 'parser_id', 'chunking_strategy_display']], use_container_width=True) # Add chunking_strategy_display
        else:
            st.info("No Knowledge Bases found. Create one below.")

        with st.expander("➕ Create New Knowledge Base", expanded=not bool(kbs_data)):
            with st.form("create_kb_form"):
                name = st.text_input("KB Name*", help="Unique name for the Knowledge Base.")
                description = st.text_area("Description")
                
                # --- Parser and Chunking Config ---
                parser_id_options = list(PARSER_MAPPING.keys())
                parser_id = st.selectbox("Parser Engine*", options=parser_id_options, index=parser_id_options.index(DEFAULT_PARSER_ID) if DEFAULT_PARSER_ID in parser_id_options else 0)
                
                chunk_strategy_options = list(DEFAULT_CHUNK_SETTINGS.keys()) + ["none"]
                chunk_strategy = st.selectbox("Chunking Strategy (for naive/fallback parsing)*", 
                                              options=chunk_strategy_options, 
                                              index=chunk_strategy_options.index(DEFAULT_CHUNKING_STRATEGY) if DEFAULT_CHUNKING_STRATEGY in chunk_strategy_options else 0)
                
                st.markdown("Advanced Parser/Chunking JSON Configuration (Optional):")
                # Provide default structure based on selected parser/chunker
                default_cfg_for_json = {
                    "parser_id_override": parser_id, # Store selected for reference
                    "chunking_strategy_override": chunk_strategy, # Store selected for reference
                    # Add placeholders for common parser_options or chunking_settings
                    "pdf_parser_options": DEFAULT_PARSER_CONFIG.get("pdf_parser_options"),
                    "txt_parser_options": DEFAULT_PARSER_CONFIG.get("txt_parser_options"),
                    "chunking_settings_override": DEFAULT_CHUNK_SETTINGS.get(chunk_strategy, {})
                }
                parser_config_json_str = st.text_area("Full Parser & Chunking Config (JSON)", 
                                                      value=json.dumps(default_cfg_for_json, indent=2), 
                                                      height=200,
                                                      help="Overrides individual selections if valid JSON. Store parser-specific options and chunking_settings here.")

                submitted = st.form_submit_button("Create KB")
                if submitted:
                    if not name:
                        st.error("KB Name is required.")
                    else:
                        try:
                            # Parse the JSON config if provided
                            final_parser_config = json.loads(parser_config_json_str)
                            # Ensure it contains at least the selected chunking strategy/settings
                            # This logic can be refined to merge UI selections with JSON overrides
                            final_parser_config.setdefault("chunking_strategy", chunk_strategy)
                            final_parser_config.setdefault("chunking_settings", DEFAULT_CHUNK_SETTINGS.get(chunk_strategy, {}))
                            if "chunking_settings_override" in final_parser_config: # Prioritize override
                                final_parser_config["chunking_settings"] = final_parser_config["chunking_settings_override"]
                            
                        except json.JSONDecodeError:
                            st.error("Invalid JSON in Parser & Chunking Config.")
                            final_parser_config = DEFAULT_PARSER_CONFIG.copy() # Fallback
                            final_parser_config["chunking_strategy"] = chunk_strategy
                            final_parser_config["chunking_settings"] = DEFAULT_CHUNK_SETTINGS.get(chunk_strategy, {})
                        
                        result = self.orchestrator.create_knowledge_base(
                            name=name, 
                            description=description,
                            parser_id=parser_id, # This is the primary parser selection
                            parser_config=final_parser_config # This holds chunking + specific parser opts
                        )
                        if result["success"]:
                            st.success(result["message"])
                            self._refresh_app_info()
                            st.session_state.selected_kb_id = result["id"] # Select new KB
                            st.rerun()
                        else:
                            st.error(result["message"])
        
        # TODO: Add View/Edit/Delete KB functionality here
        # This would involve selecting a KB from the table and then showing a form/details.

    def render_document_management(self):
        st.header("📄 Document Management")

        kb_list = self.orchestrator.list_knowledge_bases_detailed() # Orchestrator needs this method
        kb_names_to_ids = {kb['name']: kb['id'] for kb in kb_list}

        if not kb_list:
            st.warning("No Knowledge Bases exist. Please create one in the 'Knowledge Bases' tab first.")
            return

        # Ensure selected_kb_id is valid
        if st.session_state.selected_kb_id not in kb_names_to_ids.values():
            st.session_state.selected_kb_id = kb_list[0]['id'] if kb_list else None
        
        selected_kb_name = st.selectbox(
            "Select Knowledge Base", 
            options=list(kb_names_to_ids.keys()),
            index=list(kb_names_to_ids.values()).index(st.session_state.selected_kb_id) if st.session_state.selected_kb_id in kb_names_to_ids.values() else 0,
            key="doc_mgmt_kb_select" # Add a key
        )
        st.session_state.selected_kb_id = kb_names_to_ids.get(selected_kb_name)

        if not st.session_state.selected_kb_id:
            st.info("Please select a Knowledge Base.")
            return

        st.subheader(f"Documents in '{selected_kb_name}' (ID: {st.session_state.selected_kb_id})")

        # --- File Uploader ---
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            accept_multiple_files=True, 
            type=['pdf', 'txt', 'md', 'docx', 'pptx', 'xlsx', 'html', 'htm', 'json', 'png', 'jpg', 'jpeg'] # Add image types if parsers support
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_bytes_io = BytesIO(uploaded_file.getvalue())
                with st.spinner(f"Adding '{uploaded_file.name}'..."):
                    result = self.orchestrator.add_document_and_trigger_processing(
                        kb_id=st.session_state.selected_kb_id,
                        original_filename=uploaded_file.name,
                        file_content_stream=file_bytes_io
                    )
                    if result["success"]:
                        st.success(result["message"])
                    else:
                        st.error(result["message"])
            self._refresh_app_info() # Refresh KB doc counts
            st.rerun() # Rerun to update document list

        # --- List Documents ---
        docs_data = self.orchestrator.get_documents_for_kb_detailed(st.session_state.selected_kb_id) # Orchestrator needs this method
        if docs_data:
            # Convert create_time from ms timestamp to datetime string for display
            for doc_item in docs_data:
                if 'create_time' in doc_item and doc_item['create_time']:
                    from app.database import timestamp_ms_to_datetime # Assuming this utility
                    dt_obj = timestamp_ms_to_datetime(doc_item['create_time'])
                    doc_item['Created'] = dt_obj.strftime('%Y-%m-%d %H:%M') if dt_obj else 'N/A'
                else:
                    doc_item['Created'] = 'N/A'
            
            df_docs = pd.DataFrame(docs_data)
            st.dataframe(df_docs[['id', 'name', 'status', 'progress', 'file_type', 'chunk_count', 'token_count', 'Created']], use_container_width=True)

            # --- View Document Details & Artifacts ---
            doc_ids_in_kb = [doc['id'] for doc in docs_data]
            selected_doc_id_for_detail = st.selectbox("View details for Document ID:", [""] + doc_ids_in_kb)

            if selected_doc_id_for_detail:
                doc_detail = self.orchestrator.get_document_details_with_artifacts(selected_doc_id_for_detail) # Orchestrator needs this
                if doc_detail:
                    st.markdown(f"### Details for Document: {doc_detail.get('name')}")
                    st.json(doc_detail, expanded=False) # Show all DB fields initially collapsed

                    # Display layout_analysis_results if present (C4)
                    layout_artifacts = doc_detail.get("layout_analysis_results")
                    if isinstance(layout_artifacts, list) and layout_artifacts:
                        st.markdown("#### Extracted Layout Artifacts:")
                        for artifact in layout_artifacts:
                            col1, col2 = st.columns([1,3])
                            with col1:
                                if artifact.get("stored_image_path") and os.path.exists(artifact["stored_image_path"]):
                                    try:
                                        st.image(artifact["stored_image_path"], width=150)
                                    except Exception as e_img:
                                        st.caption(f"Error loading image: {artifact.get('type', 'artifact')} (preview below)")
                                else:
                                     st.caption(f"No image for: {artifact.get('type', 'artifact')}")
                            with col2:
                                st.caption(f"Type: {artifact.get('type')}, Page: {artifact.get('page', 'N/A')}")
                                if artifact.get("text_content_preview"):
                                    st.markdown(f"> {artifact['text_content_preview']}")
                                if artifact.get("html_content") and artifact.get("type") == "table":
                                    with st.expander("View Table HTML"):
                                        st.code(artifact["html_content"], language="html")
                            st.divider()
                    elif layout_artifacts: # If it's not a list or malformed
                        st.markdown("#### Layout Artifacts (Raw JSON):")
                        st.json(layout_artifacts)


                    # TODO: Add Delete Document button here
                    if st.button(f"🗑️ Delete Document '{doc_detail.get('name')}'", key=f"del_{selected_doc_id_for_detail}"):
                        with st.spinner("Deleting document..."):
                            if self.orchestrator.delete_document_orchestrated(selected_doc_id_for_detail):
                                st.success("Document deleted.")
                                self._refresh_app_info()
                                st.rerun()
                            else:
                                st.error("Failed to delete document.")
                else:
                    st.error(f"Could not fetch details for document ID: {selected_doc_id_for_detail}")
        else:
            st.info("No documents found in this Knowledge Base yet. Upload some!")


    def render_system_info(self):
        st.header("⚙️ System Information")
        self._refresh_app_info() # Ensure latest info

        st.metric("Total Items in Vector Store", self.app_info.get('vector_store_total_count', 'N/A'))
        # active_tasks = self.orchestrator.get_active_task_count() # Orchestrator needs this
        # st.metric("Active Processing Tasks", active_tasks if active_tasks is not None else "N/A")
        
        st.subheader("Configuration Summary:")
        cfg_summary = self.app_info.get('config_summary', {})
        if cfg_summary:
            for key, value in cfg_summary.items():
                st.text(f"{key+':':<30} {value}")
        else:
            st.text("Configuration summary not available.")

        st.subheader("Example Queries:")
        for ex in self.app_info.get('examples', []):
            st.markdown(f"`{ex}`")
            
    def render(self):
        st.set_page_config(page_title=self.app_info.get("title", "RAG System"), layout="wide")
        st.title(self.app_info.get("title", "RAG System"))

        tab_titles = ["💬 Chat", "📚 Knowledge Bases", "📄 Documents", "⚙️ System Info"]
        tab_chat, tab_kbs, tab_docs, tab_sysinfo = st.tabs(tab_titles)

        with tab_chat:
            self.render_chat_interface()
        with tab_kbs:
            self.render_kb_management()
        with tab_docs:
            self.render_document_management()
        with tab_sysinfo:
            self.render_system_info()

# In main.py, you would call:
# orchestrator = get_orchestrator()
# ui = StreamlitAppUIRefactored(orchestrator)
# ui.render()