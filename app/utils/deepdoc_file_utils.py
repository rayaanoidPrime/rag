import os
# We use your existing app.utils.file_utils.FileType
# from app.utils.file_utils import FileType 

# Simplified version of RAGFlow's get_project_base_directory
# This assumes your 'app' directory is under the main project root.
# Adjust if your structure is different, e.g., if run from within 'app'.
_PROJECT_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_project_base_directory(*args):
    """
    Returns the absolute path to the project's base directory.
    If *args are provided, they are joined to the base path.
    """
    if args:
        return os.path.join(_PROJECT_BASE_DIR, *args)
    return _PROJECT_BASE_DIR

# Add other utility functions from the provided RAGFlow api.utils.file_utils.py 
# ONLY IF they are directly imported and used by the deepdoc_components' PARSERS.
# Functions like thumbnail_img, load_yaml_conf etc., might not be needed by core parsers.
# For now, get_project_base_directory is the most critical one.