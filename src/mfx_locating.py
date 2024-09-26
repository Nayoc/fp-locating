import scipy.io
import pandas as pd
import numpy as np
from pyproj import Proj, Transformer
from datetime import datetime, timedelta


c = 3 * 10 ** 8
Ts = 1 / (15000 * 2048)

# 定义投影：ECEF 投影（地心地固坐标系）和 LLA 投影（经纬度坐标系）
ecef_proj = Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla_proj = Proj(proj="latlong", ellps="WGS84", datum="WGS84")

# 问题：
# 1.录入时需要针对频ue接受的频次进行排序
# 2.录入时需要针对频timestamp进行排序


# 基站，工参数据在外部使用其他函数批量读取后写入本类中
class MBaseData:
    def __init__(self, enci, pci, tadv, haoa, vaoa, timestamp,
                 wgs84_x, wgs84_y, height, azimuth, downtilt,
                 longitude='', latitude=''):
        self.tadv = tadv
        self.haoa = haoa
        self.vaoa = vaoa
        self.wgs84_x = wgs84_x
        self.wgs84_y = wgs84_y
        self.height = height
        self.azimuth = azimuth
        self.downtilt = downtilt
        self.timestamp = timestamp
        self.longitude = longitude
        self.latitude = latitude


# UE
class MUE:
    def __init__(self, msisdn, mbases: []):
        self.msisdn = msisdn
        self.mbases = mbases


# 工参处理和定位计算方法
def mlocate(mue: MUE):
    """
    参数:
    mue (MUE): 单个 UE 在一个时间点与基站的信号和工参数据
    c (float): 光速 (默认为 3 * 10^8 米/秒)
    Ts (float): 采样时间 (默认为 1 / (15000 * 2048))

    返回:
    (float, float): 计算得到的经纬度
    """

    # 计算相对坐标
    ref_mbase = mue.mbases[0]  # 选择第一个基站作为参考基站
    ref_coords = [ref_mbase.wgs84_y, ref_mbase.wgs84_x, ref_mbase.height]  # 第一个基站的 WGS84 坐标
    mue.mbases[0].xyz_enu_onePCI = [0, 0, 0]  # 第一个基站的 ENU 坐标为 [0, 0, 0]

    # 对每个基站进行 ENU 坐标计算，基于参考基站（第一个基站）的 WGS84 坐标
    for mbase in mue.mbases[1:]:
        # 计算当前基站相对于第一个基站的 ENU 坐标
        relative_enu = lla2enu(mbase.wgs84_y, mbase.wgs84_x, mbase.height,
                               *ref_coords)

        # 计算当前基站的最终 ENU 坐标
        # 将当前基站的 xyz_enu 坐标转换到以第一个基站为参考点的坐标系
        mbase.xyz_enu_onePCI = [
            relative_enu[0],
            relative_enu[1],
            relative_enu[2]
        ]

    # 针对每个基站计算距离、AOA、ZOA 和 ENU 坐标
    for mbase in mue.mbases:
        # 计算距离（基于 TADV）
        distance = ((mbase.tadv + 0.5) * 8) * Ts * c / 2
        mbase.distance = distance

        # 处理 HAOA 值（如果大于180度，进行角度调整）
        if mbase.haoa > 180:
            mbase.haoa -= 360

        # 计算 AOA 和 ZOA
        mbase.aoa = mbase.azimuth - (mbase.haoa + 0.5)
        mbase.zoa = mbase.downtilt + (mbase.vaoa + 0.5)

        # 计算 ENU 坐标
        mbase.xyz_enu = [
            distance * np.sin(np.radians(mbase.zoa)) * np.sin(np.radians(mbase.aoa)),
            distance * np.sin(np.radians(mbase.zoa)) * np.cos(np.radians(mbase.aoa)),
            distance * np.cos(np.radians(mbase.zoa))
        ]

        # 坐标归一化到参考基站,暂时没有用上
        mbase.xyz_enu_onePCI = [
            mbase.xyz_enu[0] + mbase.xyz_enu_onePCI[0],
            mbase.xyz_enu[1] + mbase.xyz_enu_onePCI[1],
            mbase.xyz_enu[2] + mbase.xyz_enu_onePCI[2]
        ]

        # ENU转为经纬度
        mbase.lla = enu2lla(*mbase.xyz_enu, mbase.wgs84_y, mbase.wgs84_x, mbase.height)  # TODO

    # ue的lla使用当前第一个base计算的lla
    mue.lla_0 = mue.mbases[0].lla
    mue.lla_avg = np.mean([mbase.lla for mbase in mue.mbases],axis=0)

    return mue


def lla2enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):

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

    # 创建转换器对象
    transformer_to_ecef = Transformer.from_proj(lla_proj, ecef_proj, always_xy=True)
    transformer_to_lla = Transformer.from_proj(ecef_proj, lla_proj, always_xy=True)

    # 将参考点的 LLA 坐标转换为 ECEF 坐标
    ref_x, ref_y, ref_z = transformer_to_ecef.transform(ref_lon, ref_lat, ref_alt)

    # 将 ENU 坐标转换为 ECEF 坐标
    t = np.array([[-np.sin(np.radians(ref_lon)),  np.cos(np.radians(ref_lon)), 0],
                  [-np.sin(np.radians(ref_lat)) * np.cos(np.radians(ref_lon)), -np.sin(np.radians(ref_lat)) * np.sin(np.radians(ref_lon)), np.cos(np.radians(ref_lat))],
                  [ np.cos(np.radians(ref_lat)) * np.cos(np.radians(ref_lon)),  np.cos(np.radians(ref_lat)) * np.sin(np.radians(ref_lon)), np.sin(np.radians(ref_lat))]])

    ecef_offset = np.dot(t.T, np.array([east, north, up]))
    x = ref_x + ecef_offset[0]
    y = ref_y + ecef_offset[1]
    z = ref_z + ecef_offset[2]

    # 将 ECEF 坐标转换回 LLA 坐标
    lon, lat, alt = transformer_to_lla.transform(x, y, z)

    return lat, lon, alt  # 返回经纬度和高度



if __name__ == '__main__':
    # 写入基站数据测试，数据来源MR6月22文件

    # 16824
    base1 = MBaseData(enci=31409496070, pci=737, tadv=6, haoa=57, vaoa=20, timestamp='2024-06-14T23:32:51.840',
                      wgs84_x=113.307999, wgs84_y=23.134001, height=55, azimuth=280, downtilt=12)

    # 16829
    base2 = MBaseData(enci=31409496070, pci=737, tadv=7, haoa=56, vaoa=20, timestamp='2024-06-14T23:30:48.960',
                      wgs84_x=113.307999, wgs84_y=23.134001, height=55, azimuth=280, downtilt=12)

    # 18434
    base3 = MBaseData(enci=31415017484, pci=683, tadv=12, haoa=301, vaoa=354, timestamp='2024-06-14T22:59:35.040',
                      wgs84_x=113.304001, wgs84_y=23.1313, height=50, azimuth=340, downtilt=7)

    # 创建 UE 实例
    ue = MUE(msisdn='cb105a76be531df3541c91', mbases=[base1, base2, base3])

    mlocate(ue)

    for mbase in ue.mbases:
        print(mbase.lla)

