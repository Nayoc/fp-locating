import os
from typing import List, Any, Union
from pathlib import Path


class TxtArrayTool:
    def write(self, file_path, data_list):
        """按行写入列表数据"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(str(item) + '\n')

    def read(self, file_path):
        """按行读取数据，返回列表"""
        if not Path(file_path).exists():
            return []
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
