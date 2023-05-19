from typing import Any, Optional
import gcsfs
import os

GCLOUD_TOKEN_PATH = os.environ.get('GCLOUD_TOKEN_PATH', None)
GCLOUD_PROJECT = os.environ.get('GCLOUD_PROJECT', None)

def open_with_bucket(
    path: Any, 
    mode: str="rb", 
    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[Any]=None, 
    **kwargs, 
):
    # backup to env vars if None
    if gcloud_project is None:
        gcloud_project = GCLOUD_PROJECT
    if gcloud_token is None:
        gcloud_token = GCLOUD_TOKEN_PATH
    # load from google cloud storage if starts with "gcs://"
    if path.startswith('gcs://'):
        f = gcsfs.GCSFileSystem(project=gcloud_project, token=gcloud_token).open(path[len('gcs://'):], mode=mode, **kwargs)
    else:
        f = open(path, mode=mode, **kwargs)
    return f

def delete_with_bucket(
    path: str, 
    recursive: bool=True, 
    gcloud_project: Optional[str]=None, 
    gcloud_token: Optional[Any]=None, 
) -> None:
    # backup to env vars if None
    if gcloud_project is None:
        gcloud_project = GCLOUD_PROJECT
    if gcloud_token is None:
        gcloud_token = GCLOUD_TOKEN_PATH
    # delete from google cloud storage if starts with "gcs://"
    if path.startswith('gcs://'):
        path = path[len('gcs://'):]
        gcsfs.GCSFileSystem(project=gcloud_project, token=gcloud_token).rm(path, recursive=recursive)
    else:
        os.system(f"rm -{'r' if recursive else ''}f {path}")
