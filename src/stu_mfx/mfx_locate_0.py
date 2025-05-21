import csv
from datetime import datetime

import numpy as np
# 文件读取
from openpyxl import load_workbook
from pyproj import Proj, Transformer
from sklearn.cluster import DBSCAN

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
                 wgs84_x, wgs84_y, height, dir, tilt,
                 longitude='', latitude=''):
        self.enci = enci
        self.pci = pci
        self.tadv = float(tadv)
        self.haoa = float(haoa)
        self.vaoa = float(vaoa)
        self.wgs84_x = float(wgs84_x)
        self.wgs84_y = float(wgs84_y)
        self.height = float(height)
        self.dir = float(dir)
        self.tilt = float(tilt)
        self.timestamp = timestamp
        self.longitude = longitude
        self.latitude = latitude


# UE
class MUE:
    def __init__(self, msisdn, mbases: [], zero_nci):
        self.msisdn = msisdn
        self.mbases = mbases
        # 基准基站索引
        self.zero_nci = zero_nci


# servTa, servHaoa, servVaoa, servPci_x, servPci_y, axis
# tadv, haoa, vaoa,latitude, longitude
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
        mbase.aoa = mbase.dir - (mbase.haoa + 0.5)
        mbase.zoa = mbase.tilt + (mbase.vaoa + 0.5)

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
    mue.lla_avg = np.mean([mbase.lla for mbase in mue.mbases], axis=0)

    # 筛选夜间数据
    night_data = []
    for mbase in mue.mbases:
        # TODO 日期转格式需按照最后输入格式调整
        timestamp = datetime.strptime(mbase.timestamp, "%Y-%m-%dT%H:%M:%S.%f")
        if timestamp.hour >= 19 or timestamp.hour < 7:  # 夜间时间段
            night_data.append(mbase)

    mue.night_data = night_data

    # 使用 DBSCAN 聚类算法来找出基站之间的关系，目前只使用夜间数据,获取2D坐标
    coordinates_2d_night = [(mbase.xyz_enu_onePCI[0], mbase.xyz_enu_onePCI[1]) for mbase in mue.night_data]
    coordinates_2d_night = np.array(coordinates_2d_night)

    # 第一次 DBSCAN 聚类
    epsilon1 = 200
    minPts1 = 50
    db1 = DBSCAN(eps=epsilon1, min_samples=minPts1).fit(coordinates_2d_night)
    labels1 = db1.labels_

    # 计算每个簇的聚类中心
    numClusters = max(labels1) + 1

    # 全是噪声点时
    centroids = np.zeros((numClusters + 1, 2))

    for i in range(numClusters):
        clusterPoints = coordinates_2d_night[labels1 == i, :]  # 获取每个簇的点
        centroids[i, :] = np.mean(clusterPoints, axis=0)  # 计算聚类中心

    # 第二次 DBSCAN 聚类，决定集中位置
    if numClusters > 0:
        epsilon2 = 50
        minPts2 = 20

        # 第二次聚类只使用第一次聚类中标记为有效（> 0）点的坐标
        coordinates_2d_2nd = coordinates_2d_night[labels1 > -1, :]
        db2 = DBSCAN(eps=epsilon2, min_samples=minPts2).fit(coordinates_2d_2nd)
        labels2 = db2.labels_

        # 聚类结果的唯一标签
        unique_clusters_2nd = np.unique(labels2[labels2 > -1])
        numClusters_2nd = max(unique_clusters_2nd) + 1  # 聚类数量
        final_centroids = np.zeros((numClusters_2nd, 2))  # 预分配最终聚类中心坐标矩阵
        cluster_count = np.zeros(numClusters_2nd)  # 记录每个簇的点数

        for i in range(numClusters_2nd):
            cluster_centroid_points = coordinates_2d_2nd[labels2 == i, :]
            final_centroids[i, :] = np.mean(cluster_centroid_points, axis=0)  # 计算新的质心
            cluster_count[i] = len(cluster_centroid_points)

    # 聚类结果处理
    if numClusters == -1 or numClusters_2nd == -1:
        # 无聚类中心
        print(f"用户 {mue.msisdn} 无聚类结果")
        mue.centroid = [0, 0]
    elif numClusters_2nd == 1:
        # 只有一个聚类中心，直接输出结果
        finalCentroid = final_centroids[0, :]
        mue.centroid = finalCentroid
    else:
        # 有多个聚类中心，计算中心之间的距离
        firstCentroid = final_centroids[0, :]
        centroids_distances = np.sqrt(np.sum((final_centroids[1:, :] - firstCentroid) ** 2, axis=1))

        # 保留距离小于等于400米的聚类中心
        validCentroids = final_centroids[
            np.concatenate(([True], centroids_distances <= 400))]  # 包含聚类中心1，2倍的初步聚类半径范围
        mue.centroid = np.mean(validCentroids, axis=0)  # 计算所有保留的聚类中心的平均位置

    mue.centroidPoint = enu2lla(mue.centroid[0], mue.centroid[1], mue.night_data[0].height,
                                mue.night_data[0].wgs84_y, mue.night_data[0].wgs84_x, mue.night_data[0].height)

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


def test_temp_read_file():
    data = read_csv('../valid.csv')[1:]

    ues = []

    for d in data:
        # 0:msisdn,1：nci:,2：pci,3：tadv,4:haoa,5:vaoa,6：orig_time,7:wgs84_x,8:wgs84_y,9:height,10:dir,11:tilt
        if len(d) < 11:
            continue

        quit = False
        for i in d:
            if i is None or i == 'NULL':
                quit = True

        if quit:
            continue

        base = MBaseData(d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11])
        exist_ue = False
        for ue in ues:
            if ue.msisdn == d[0]:
                ue.mbases.append(base)
                exist_ue = True
                break
        if not exist_ue:
            ue = MUE(d[0], [base], base.enci)
            ues.append(ue)

    # 找出每个ue的原点基站
    for ue in ues:
        count = {}
        for b in ue.mbases:
            if b.enci not in count.keys():
                count[b.enci] = 1
            else:
                count[b.enci] += 1

        ue.zero_nci = max(count, key=count.get)

    return ues


def build_valid_data():
    data = read_excel('广州样本用户-5GXDR表.xlsx')
    gz = read_excel('ngz.xlsx')
    data = data[1:]

    # nci:6,pci:9,tadv:13,longitude:15,latitude:16,msisdn:138,orig_time:146,haoa:150,vaoa:151,rsrp:10
    indexs1 = [138, 6, 9, 13, 150, 151, 146]
    indexs2 = [3, 4, 5, 6, 7]

    result = []

    for sublist in data:
        temp = []
        # 0:msisdn,1：nci:,2：pci,3：tadv,4:haoa,5:vaoa,6：orig_time,
        for index in indexs1:
            if index < len(sublist):
                temp.append(sublist[index - 1])

        # 添加工参,7:wgs84_x,8:wgs84_y,9:height,10:dir,11:tilt
        nci = sublist[5]

        for igz in gz:
            if (igz[0] == nci):
                for index in indexs2:
                    if index < len(sublist):
                        temp.append(igz[index - 1])
                break

        if temp:
            result.append(temp)

    header = ['msisdn', 'nci', 'pci', 'tadv', 'haoa', 'vaoa', 'orig_time', 'wgs84_x', 'wgs84_y', 'height', 'dir',
              'tilt']
    # 写入csv，加速读取
    write_csv("../valid.csv", header, result)


def read_excel(path):
    wb = load_workbook(filename=path)
    ws = wb.active
    data = []
    for row in ws.iter_rows(values_only=True):
        r = list(row)
        data.append(r)
    return data


def read_csv(file_name):
    """读取CSV文件并返回数据列表"""
    data = []
    with open(file_name, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)  # 将每行数据添加到列表中
    return data


def write_csv(file_name, header, rows):
    """将数据写入CSV文件"""
    with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # 写入表头
        writer.writerows(rows)  # 写入多行数据


if __name__ == '__main__':
    # build_valid_data()
    # build_nci()

    ues = test_temp_read_file()
    ue = mlocate(ues[0])
    print(ue)
