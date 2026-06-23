
from io import FileHandle
from tempfile import NamedTemporaryFile


def file_handle(tempfile: NamedTemporaryFile) -> ref [tempfile._file_handle] FileHandle:
    return tempfile._file_handle
    # NOTE: _file_handle is internal, and therefore probably unstable?
