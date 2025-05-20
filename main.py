import os
import streamlit as st
from app.config import APP_TITLE, TOKENIZERS_PARALLELISM
from app.streamlit_ui import StreamlitAppUI, get_orchestrator
from app.database.models import create_tables # Import the function
import traceback

st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧩",
        layout="centered",
        initial_sidebar_state="auto",
        menu_items={
            'Get Help': 'https://www.example.com/help',
            'Report a bug': "https://www.example.com/bug",
            'About': f"## {APP_TITLE}\nA RAG application with graph capabilities."
        }
    )

def main():
    # --- Database Table Creation ---
    # This should be run once, ideally.
    # In a production setup, migrations (e.g., with Peewee's migrator or Alembic) are better.
   try:
        create_tables()
        st.logger.info("Database tables creation/check successful.")
    except Exception as e:
        st.logger.error(f"Database table creation failed: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not connect to or set up the database. Error: {e}")
        return

    st.set_page_config(...) # As before

    os.environ["TOKENIZERS_PARALLELISM"] = TOKENIZERS_PARALLELISM
    
    orchestrator = get_orchestrator() # This is @st.cache_resource

    try:
        ui = StreamlitAppUI(orchestrator)
        ui.render()
    finally:
        # This might not be the ideal place for Streamlit due to its execution model.
        # A better approach for resource cleanup is often context managers or specific
        # app lifecycle hooks if the framework provides them.
        # For ThreadPoolExecutor, if it's session-scoped via @st.cache_resource,
        # direct shutdown might be tricky.
        # However, if orchestrator is a true singleton for the app process, this is okay.
        # Streamlit's @st.cache_resource objects are cleaned up when the app session ends
        # or the app server stops. For a ThreadPoolExecutor, a more explicit shutdown
        # as part of the application's teardown process is good practice if possible.
        # For now, let's assume this `finally` block in `main` will be reached
        # when the script exits (e.g., if Streamlit server is stopped).
        # A more robust solution for web apps might involve `atexit` module or framework signals.
        pass # orchestrator.shutdown_task_executor() # Let's defer this, see note below.

if __name__ == "__main__":
    main()
    # If orchestrator were a global singleton not managed by Streamlit's cache,
    # you could do orchestrator_instance.shutdown_task_executor() here after main() returns.
    # But with @st.cache_resource, the lifecycle is tied to Streamlit's caching.
    # Python's `atexit` module is a more reliable way to ensure cleanup on script exit.