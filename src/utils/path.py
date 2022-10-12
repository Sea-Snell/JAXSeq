import os
from typing import Optional

project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

# convert relative paths to be absolute paths from project root
def convert_path(path: Optional[str]):
    if path is None:
        return None
    if path.startswith('/') or path.startswith('gcs://'):
        return path
    return path
