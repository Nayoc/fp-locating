import os
from typing import List, Any, Union


class TxtArrayTool:
    """
    TXT 数组读写工具类，支持多维数组（列表）的写入与读取，保留原始维度结构
    支持数据类型：数值型（int/float）、字符串型，自动处理维度分隔
    """

    def __init__(self, sep: str = "\t", encoding: str = "utf-8"):
        """
        初始化工具类
        :param sep: 元素分隔符（默认制表符\t，可改为逗号,、空格等）
        :param encoding: 文件编码（默认utf-8，Windows系统可改为gbk）
        """
        self.sep = sep
        self.encoding = encoding
        # 维度标记（用于读取时识别多维结构，避免与数据冲突）
        self.dim_mark = "__DIM__"

    def _flatten_with_dim(self, data: Any) -> List[str]:
        """
        递归扁平化数组，并添加维度标记（内部使用）
        :param data: 任意维度的数组/列表
        :return: 扁平化后的字符串列表（含维度标记）
        """
        flat_data = []
        if isinstance(data, (list, tuple)):
            # 标记当前维度的开始和长度
            flat_data.append(f"{self.dim_mark}{len(data)}")
            for item in data:
                flat_data.extend(self._flatten_with_dim(item))
        else:
            # 非数组元素直接转为字符串
            flat_data.append(str(data))
        return flat_data

    def _reconstruct_from_flat(self, flat_data: List[str]) -> Any:
        """
        从扁平化数据递归重构多维数组（内部使用）
        :param flat_data: 含维度标记的扁平化字符串列表
        :return: 重构后的多维数组
        """
        if not flat_data:
            return []

        # 读取维度标记
        first_item = flat_data.pop(0)
        if first_item.startswith(self.dim_mark):
            # 获取当前维度的长度
            dim_len = int(first_item.replace(self.dim_mark, ""))
            # 递归构建当前维度的元素
            return [self._reconstruct_from_flat(flat_data) for _ in range(dim_len)]
        else:
            # 非维度标记，尝试转换为数值类型（失败则保留字符串）
            try:
                return int(first_item) if first_item.isdigit() else float(first_item)
            except (ValueError, TypeError):
                return first_item

    def write(self, file_path: str, data: Union[List, tuple]):
        """
        写入数组到 TXT 文件
        :param file_path: 输出文件路径（如 "./data/array.txt"）
        :param data: 任意维度的数组/列表（支持1D、2D、3D、4D等）
        :raises ValueError: 输入数据不是列表或元组
        :raises IOError: 文件写入失败
        """
        if not isinstance(data, (list, tuple)):
            raise ValueError("输入数据必须是列表或元组类型")

        # 确保目录存在
        dir_path = os.path.dirname(file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # 扁平化数据并添加维度标记
        flat_data = self._flatten_with_dim(data)
        # 用分隔符连接所有元素
        content = self.sep.join(flat_data)

        # 写入文件
        try:
            with open(file_path, "w", encoding=self.encoding) as f:
                f.write(content)
        except Exception as e:
            raise IOError(f"文件写入失败：{str(e)}")

    def read(self, file_path: str) -> Any:
        """
        从 TXT 文件读取数据并重构为数组
        :param file_path: 输入文件路径
        :return: 重构后的多维数组（与写入时维度一致）
        :raises FileNotFoundError: 文件不存在
        :raises IOError: 文件读取失败
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")

        # 读取文件内容
        try:
            with open(file_path, "r", encoding=self.encoding) as f:
                content = f.read().strip()
        except Exception as e:
            raise IOError(f"文件读取失败：{str(e)}")

        if not content:
            return []

        # 按分隔符分割为扁平化列表
        flat_data = content.split(self.sep)
        # 重构多维数组
        data = self._reconstruct_from_flat(flat_data)
        return data

    def append(self, file_path: str, data: Union[List, tuple]):
        """
        追加数组到 TXT 文件末尾（仅支持1D数组，或与原文件维度一致的多维数组）
        :param file_path: 文件路径
        :param data: 要追加的数组（1D或与原文件维度一致）
        """
        if not isinstance(data, (list, tuple)):
            raise ValueError("输入数据必须是列表或元组类型")

        # 如果文件不存在，直接写入
        if not os.path.exists(file_path):
            self.write(file_path, data)
            return

        # 读取原有数据的扁平化结构
        with open(file_path, "r", encoding=self.encoding) as f:
            content = f.read().strip()
        original_flat = content.split(self.sep) if content else []

        # 扁平化要追加的数据
        new_flat = self._flatten_with_dim(data)
        # 追加到原有数据后
        all_flat = original_flat + new_flat

        # 重新写入文件
        with open(file_path, "w", encoding=self.encoding) as f:
            f.write(self.sep.join(all_flat))