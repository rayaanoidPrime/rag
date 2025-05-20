import streamlit as st
from app.orchestrator import AppOrchestrator # The heart of the application logic
from PIL.Image import Image as PILImage
import platform # For torch workaround
import torch # For torch workaround

# Workaround for torch / streamlit issue if still needed
# Check if this is necessary with newer Streamlit/Torch versions
if platform.system() == "Windows" or platform.system() == "Linux": # Or more specific checks
    try:
        torch.classes.__path__ = []
    except Exception as e:
        st.logger.warning(f"Failed to apply torch.classes workaround: {e}")


# Initialize logger for UI specific messages if needed
logger = st.logger.get_logger(__name__)

# Cache the AppOrchestrator instance for the session
@st.cache_resource(show_spinner="Initializing application, models, and database...")
def get_orchestrator():
    logger.info("Creating/Retrieving AppOrchestrator instance.")
    return AppOrchestrator()

class StreamlitAppUI:
    def __init__(self, orchestrator: AppOrchestrator):
        self.orchestrator = orchestrator
        self.app_info = self.orchestrator.get_app_info()

        # Initialize session state for messages if not already present
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": self._build_welcome_message()}]

    def _build_welcome_message(self) -> str:
        welcome = f"Welcome! Ask a question about your data, or try an example like: `{self.app_info['examples'][0]}`\n\n"
        if self.orchestrator.vector_store.count() == 0:
            welcome += "**The knowledge base is currently empty.**\n\n"
        
        welcome += "You can add data by typing:\n"
        welcome += "- `# /path/to/your/file.pdf` (local file)\n"
        welcome += "- `# https://example.com/document.html` (URL)\n"
        # welcome += "- `# your custom text string here!` (direct text - not implemented in orchestrator yet)\n\n"
        
        if self.app_info['supports_graph_rag']:
            welcome += "This application also supports `📈 GraphRAG` queries. Examples:\n"
            for ex in self.app_info['examples']:
                if "gq:" in ex or "->" in ex:
                    welcome += f"- `{ex}`\n"
        
        welcome += "\nType `:settings` to see current configuration."
        return welcome

    def _display_settings(self) -> str:
        settings_md = "## Current Application Settings:\n\n"
        settings_md += f"| Setting                 | Value                                   |\n"
        settings_md += f"|-------------------------|-----------------------------------------|\n"
        settings_md += f"| Record Count in KB      | {self.app_info['vector_store_count']}   |\n"
        for key, value in self.app_info['config_summary'].items():
            settings_md += f"| {key:<23} | {value} |\n"
        return settings_md

    def render(self):
        st.title(self.app_info['title'])

        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"]) # Use markdown for better formatting
                if "image" in msg and msg["image"]:
                    st.image(msg["image"], caption=msg.get("image_caption", "Query Result Graph"))


        # Handle user input
        if user_input := st.chat_input("Ask your question or type a command..."):
            logger.info(f"User input received: {user_input}")
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Process input
            with st.chat_message("assistant"):
                response_content = ""
                response_image = None
                
                if user_input.startswith("#"):
                    url_or_path = user_input[1:].strip()
                    if not url_or_path:
                        response_content = "Please provide a valid file path or URL after `#`."
                    else:
                        with st.spinner(f"Processing and adding `{url_or_path}` to knowledge base..."):
                            try:
                                response_content = self.orchestrator.add_document_from_url_or_path(url_or_path)
                                # Update app_info after adding doc for correct count in welcome/settings
                                self.app_info = self.orchestrator.get_app_info()
                            except Exception as e:
                                logger.error(f"Error adding document {url_or_path}: {e}", exc_info=True)
                                response_content = f"❌ Error processing `{url_or_path}`: {e}"
                    st.markdown(response_content)

                elif user_input.lower() == ":settings":
                    response_content = self._display_settings()
                    st.markdown(response_content)
                
                elif user_input.lower() == ":help":
                    response_content = self._build_welcome_message() # Show instructions again
                    st.markdown(response_content)

                else: # It's a query for the RAG system
                    with st.spinner("Thinking..."):
                        try:
                            query_result = self.orchestrator.process_query(user_input, stream_response=True)
                            
                            # Handle streamed response
                            if isinstance(query_result["answer"], Generator):
                                full_response_md = st.write_stream(query_result["answer"])
                                response_content = full_response_md # Store the full response after streaming
                            else: # Non-streamed response (should not happen if stream_response=True)
                                response_content = query_result["answer"]
                                st.markdown(response_content)
                            
                            if query_result.get("graph_image"):
                                response_image = query_result["graph_image"]
                                # Display image directly if not already handled by message loop
                                # (The loop above will display it on next full re-render)
                                st.image(response_image, caption="Query Result Graph")

                        except Exception as e:
                            logger.error(f"Error processing query '{user_input}': {e}", exc_info=True)
                            response_content = f"😕 Apologies, I encountered an error: {e}"
                            st.error(response_content)
                
                # Append assistant's full response to session state
                assistant_message = {"role": "assistant", "content": response_content}
                if response_image:
                    assistant_message["image"] = response_image
                    assistant_message["image_caption"] = "Query Result Graph"
                st.session_state.messages.append(assistant_message)

# Main execution block for Streamlit page will be in main.py