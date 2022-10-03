import contextlib
from typing import Any, Optional
import gcsfs

@contextlib.contextmanager
def open_pp(path: Any, mode: str="rb", gcloud_project: Optional[str]=None, 
            gcloud_token: Optional[Any]=None, **kwargs):
    
    # load from google cloud storage if starts with "gcs://"
    if path.startswith('gcs://'):
        file = gcsfs.GCSFileSystem(project=gcloud_project, token=gcloud_token).open(path[len('gcs://'):], mode=mode, **kwargs)
    else:
        file = open(path, mode=mode, **kwargs)
    
    yield file

    file.close()
