from typing import List
import os


def get_data_files_from_folder(path: str, data_type: str = ".wav") -> List[str]:
    files = [f for f in os.listdir(path) if f.endswith(data_type)]
    return files
