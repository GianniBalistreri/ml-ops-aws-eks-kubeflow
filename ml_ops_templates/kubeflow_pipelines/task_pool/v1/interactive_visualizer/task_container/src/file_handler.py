"""

Handling files and directories

"""

import json
import os
import pickle


class FileHandlerExceptions(Exception):
    """
    Class for handling exceptions for function file_handler
    """
    pass


def file_handler(file_path: str, obj) -> None:
    """
    Handling files

    :param file_path: str
        Complete file path

    :param obj:
        File object to save
    """
    _file_path: str = ''
    _file_type: str = file_path.split('.')[-1]
    for directory in file_path.split('/'):
        _file_path = os.path.join(_file_path, directory) if _file_path != '' else directory
        if directory.find(f'.{_file_type}') >= 0:
            break
        if not os.path.isdir(_file_path):
            os.mkdir(path=_file_path)
    if _file_type == 'json':
        with open(file_path, 'w') as _file:
            json.dump(obj, _file)
    elif _file_type in ['p', 'pkl', 'pickle']:
        with open(_file_path, 'wb') as _file:
            pickle.dump(obj, _file, pickle.HIGHEST_PROTOCOL)
    elif _file_type == 'txt':
        with open(file_path, 'w') as _file:
            _file.write(obj)
    elif _file_type == 'html':
        with open(file_path, "w") as _file:
            _file.write(obj)
    else:
        raise FileHandlerExceptions(f'Saving file type ({_file_type}) not supported')
