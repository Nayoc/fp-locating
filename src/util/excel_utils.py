import csv
import os

from openpyxl import Workbook
from openpyxl import load_workbook


def read_excel(path):
    wb = load_workbook(filename=path)
    ws = wb.active
    data = []
    for row in ws.iter_rows(values_only=True):
        r = list(row)
        data.append(r)
    return data


# 写入 Excel 文件，写入整个List
def write_excel(data, output_file):
    wb = Workbook()
    ws = wb.active
    for row in data:
        ws.append(row)
    wb.save_model(output_file)


# 写入 Excel 文件,写入指定行
def write_excel(data, row, output_file):
    wb = load_workbook(filename=output_file)
    sheet = wb['Sheet1']

    for col_index, value in enumerate(data, start=1):
        sheet.cell(row=row + 1, column=col_index, value=value)

    wb.save_model(output_file)


def read_csv(file_name):
    """读取CSV文件并返回数据列表"""
    data = []
    with open(file_name, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)  # 将每行数据添加到列表中
    return data


def write_to_csv(file_path, headers, data):
    """
    创建CSV文件并写入数据，自动创建所需的上级文件夹，支持自定义表头

    参数:
        file_path: CSV文件的完整路径
        data: 要写入的数据，列表的列表格式，如[[row1_col1, row1_col2], [row2_col1, ...]]
        headers: CSV文件的表头列表，如["列1", "列2"]，可选参数
    """
    try:
        # 获取文件所在的目录路径
        directory = os.path.dirname(file_path)

        # 如果目录不存在，则创建目录（包括所有上级目录）
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"已创建目录: {directory}")

        # 写入CSV文件
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            # 创建CSV写入器
            writer = csv.writer(csvfile)

            # 如果提供了表头，则先写入表头
            if headers:
                # 验证表头和数据的列数是否一致
                if data and len(headers) != len(data[0]):
                    print(f"警告: 表头列数({len(headers)})与数据列数({len(data[0])})不匹配")
                writer.writerow(headers)
                print(f"已写入表头: {headers}")

            # 写入数据
            writer.writerows(data)
            print(f"已写入 {len(data)} 行数据")

        print(f"CSV文件已成功创建: {file_path}")
        return True

    except Exception as e:
        print(f"创建CSV文件时出错: {str(e)}")
        return False
