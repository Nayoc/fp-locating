import os
import numpy as np
import optimizer
import pandas as pd
from pathlib import Path


def build_location_point(path='/Users/demons/Python/WorkSpace/fp-locating/data/ecnu'):
    # 获取当前路径
    current_path = Path(path)

    # 获取当前路径下所有文件的名称
    files = [f.name for f in current_path.iterdir() if f.is_file()]

    for file in files:
        if not file.endswith('.csv'):
            continue

        print(file)
        # data = pd.read_csv(file)
        #
        # longitude = data['Longitude'].tolist()
        # latitude = data['Latitude'].tolist()
        #
        # location = [f"{x},{y}" for x, y in zip(longitude, latitude)]
        # tu.writeList(file.split('.')[0] + '.txt', location)


if __name__ == '__main__':
    build_location_point()
