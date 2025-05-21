import numpy as np
from pyproj import Proj, Transformer

"""
data = [[servTa, servHaoa, servVaoa, servPci_x, servPci_y, axis],
        % [nbrTa_1, nbrHaoa_1, nbrVaoa_1, nbrPci_1_x, nbrPci_1_y, axis],
        % [nbrTa_2, nbrHaoa_2, nbrVaoa_2, nbrPci_2_x, nbrPci_2_y, axis],
        % [nbrTa_3, nbrHaoa_3, nbrVaoa_3, nbrPci_3_x, nbrPci_3_y, axis],
        % [nbrTa_4, nbrHaoa_4, nbrVaoa_4, nbrPci_4_x, nbrPci_4_y, axis],
        % [nbrTa_5, nbrHaoa_5, nbrVaoa_5, nbrPci_5_x, nbrPci_5_y, axis]]
"""


def ta_aoa(data):
    # 从输入数据提取 TA、HAOA、VAOA 和基站坐标
    ta = data[:, 0]
    haoa = data[:, 1]
    vaoa = data[:, 2]
    data_num = len(ta)

    bsHeight = np.full((data_num, 1), 3)
    bsPosition = np.column_stack((data[:, 4], data[:, 3], bsHeight))
    bsSite = np.zeros_like(bsPosition)

    # 计算基站位置在 ENU 坐标系下的坐标
    for i in range(data_num):
        bsSite[i, :] = lla2enu(bsPosition[i, 0], bsPosition[i, 1], bsPosition[i, 2], bsPosition[0, 0], bsPosition[0, 1],
                               bsPosition[0, 2])

    # 初始化
    ueDistance = np.zeros(data_num)
    ueSite = np.zeros((data_num, 3))
    uePosition = np.zeros((data_num, 3))

    c = 3 * 10 ** 8
    Ts = 1 / (15000 * 2048)

    # 计算每个基站的用户设备位置
    for i in range(data_num):
        # 用户距离
        ueDistance[i] = ((ta[i] + 0.5) * 8) * Ts * c / 2
        ueSite[i, :] = [
            ueDistance[i] * np.sin(np.radians(vaoa[i])) * np.sin(np.radians(haoa[i])),
            ueDistance[i] * np.sin(np.radians(vaoa[i])) * np.cos(np.radians(haoa[i])),
            ueDistance[i] * np.cos(np.radians(vaoa[i]))
        ]
        uePosition[i, :] = enu2lla(*ueSite[i, :], bsPosition[i, 0], bsPosition[i, 1], bsPosition[i, 2])  # 经纬度，高度

    # 计算用户位置的平均位置
    ueAvePosition = np.mean(uePosition, axis=0)

    return ueAvePosition[:2]


def lla2enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    # 定义投影：ECEF 投影（地心地固坐标系）和 LLA 投影（经纬度坐标系）
    ecef_proj = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla_proj = Proj(proj="latlong", ellps="WGS84", datum="WGS84")

    # 创建转换器对象
    transformer_to_ecef = Transformer.from_proj(lla_proj, ecef_proj, always_xy=True)

    # 将参考点的 LLA 坐标转换为 ECEF 坐标
    ref_x, ref_y, ref_z = transformer_to_ecef.transform(ref_lon, ref_lat, ref_alt)

    # 将目标点的 LLA 坐标转换为 ECEF 坐标
    x, y, z = transformer_to_ecef.transform(lon, lat, alt)

    # 计算目标点相对于参考点的 ECEF 坐标偏移
    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    # 将 ECEF 偏移转换为 ENU 坐标
    t = np.array([[-np.sin(np.radians(ref_lon)), np.cos(np.radians(ref_lon)), 0],
                  [-np.sin(np.radians(ref_lat)) * np.cos(np.radians(ref_lon)),
                   -np.sin(np.radians(ref_lat)) * np.sin(np.radians(ref_lon)), np.cos(np.radians(ref_lat))],
                  [np.cos(np.radians(ref_lat)) * np.cos(np.radians(ref_lon)),
                   np.cos(np.radians(ref_lat)) * np.sin(np.radians(ref_lon)), np.sin(np.radians(ref_lat))]])

    enu = np.dot(t, np.array([dx, dy, dz]))

    return enu[0], enu[1], enu[2]  # 返回 ENU 坐标


def enu2lla(east, north, up, ref_lat, ref_lon, ref_alt):
    # 定义投影：ECEF 投影（地心地固坐标系）和 LLA 投影（经纬度坐标系）
    ecef_proj = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla_proj = Proj(proj="latlong", ellps="WGS84", datum="WGS84")

    # 创建转换器对象
    transformer_to_ecef = Transformer.from_proj(lla_proj, ecef_proj, always_xy=True)
    transformer_to_lla = Transformer.from_proj(ecef_proj, lla_proj, always_xy=True)

    # 将参考点的 LLA 坐标转换为 ECEF 坐标
    ref_x, ref_y, ref_z = transformer_to_ecef.transform(ref_lon, ref_lat, ref_alt)

    # 将 ENU 坐标转换为 ECEF 坐标
    t = np.array([[-np.sin(np.radians(ref_lon)), np.cos(np.radians(ref_lon)), 0],
                  [-np.sin(np.radians(ref_lat)) * np.cos(np.radians(ref_lon)),
                   -np.sin(np.radians(ref_lat)) * np.sin(np.radians(ref_lon)), np.cos(np.radians(ref_lat))],
                  [np.cos(np.radians(ref_lat)) * np.cos(np.radians(ref_lon)),
                   np.cos(np.radians(ref_lat)) * np.sin(np.radians(ref_lon)), np.sin(np.radians(ref_lat))]])

    ecef_offset = np.dot(t.T, np.array([east, north, up]))
    x = ref_x + ecef_offset[0]
    y = ref_y + ecef_offset[1]
    z = ref_z + ecef_offset[2]

    # 将 ECEF 坐标转换回 LLA 坐标
    lon, lat, alt = transformer_to_lla.transform(x, y, z)

    return lat, lon, alt  # 返回经纬度和高度


if __name__ == '__main__':
    data = np.array([[1, 0.1, 0.2, 113.71362, 23.61383, 0],
                     [1, 0.1, 0.2, 113.484203, 23.376537, 0],
                     [1, 0.1, 0.2, 113.143421, 23.462614, 0],
                     [1, 0.1, 0.2, 113.29888, 23.39739, 0],
                     [1, 0.1, 0.2, 113.484203, 23.376537, 0],
                     [1, 0.1, 0.2, 113.71362, 23.61383, 0]])

    position = ta_aoa(data)
    print(position)
