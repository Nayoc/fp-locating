
import numpy as np

def ARC_Pos2D_TOA_RSS(gNB_pos,toa,RSRP):
    # 数据说明
    # 输入 gNB_pos 基站坐标
    # 输入 toa 基站与终端之间距离
    # 输入 RSRP 信号功率值
    # 输出 pos UE 坐标

    n_num=100 #非线性迭代次数
    N = len(gNB_pos)  # gNB数量
    b = np.zeros(N)   # RSRP NLOS误差
    beta = np.zeros(N) # TOA NLOS误差
    d0 = 50  # 参考的距离
    P0 = -70  # 参考点功率
    r = 2  # 路损系数
    y = np.array([0, 0, 0])
    # 一些中间变量，详情见论文推导，这里不详述
    p=np.zeros(N)
    d_0=np.zeros(N)
    d_1=np.zeros(N)
    E=np.zeros(N)

    # 线性解算
    A = np.zeros((2 * N, 3))
    PP = np.zeros(2 * N)
    for i in range(N):
        p[i] = 10 * 10 ** ((P0 - b[i]) / 10 / r)
        d_0[i] = d0 * 10 ** ((P0 - RSRP[i] - b[i]) / 10 / r)
        d_1[i] = toa[i] - beta[i]
        E[i] = 10 ** (RSRP[i] / 10 / r)

    for i in range(2 * N):
        if i < N:
            A[i, 0] = 2 * E[i] ** 2 * gNB_pos[i, 0]
            A[i, 1] = 2 * E[i] ** 2 * gNB_pos[i, 1]
            A[i, 2] = -E[i] ** 2
            PP[i] = E[i] ** 2 * (gNB_pos[i, 0] ** 2 + gNB_pos[i, 1] ** 2) - p[i] ** 2
        else:
            A[i, 0] = 2 * gNB_pos[i - N, 0]
            A[i, 1] = 2 * gNB_pos[i - N, 1]
            A[i, 2] = -1
            PP[i] = (gNB_pos[i - N, 0] ** 2 + gNB_pos[i - N, 1] ** 2) - d_1[i - N] ** 2
    pos = np.linalg.pinv(A.T @ A) @ A.T @ PP
    pos=pos[:2]

    # 非线性解算
    a=np.zeros((N,2))
    pp=np.zeros(N)
    for ii in range(n_num):    
        for i in range(N):
            p[i] = 10 * 10 ** ((P0 - b[i]) / 10 / r)
            d_0[i] = d0 * 10 ** ((P0 - RSRP[i] - b[i]) / 10 / r)
            d_1[i] = toa[i] - beta[i]
            E[i] = 10 ** (RSRP[i] / 10 / r)

        for i in range(N):
            a[i, 0] = E[i] ** 2 * (pos[0] - gNB_pos[i, 0]) / E[i] / d_0[i] + (pos[0] - gNB_pos[i, 0]) / d_1[i]
            a[i, 1] = E[i] ** 2 * (pos[1] - gNB_pos[i, 1]) / E[i] / d_0[i] + (pos[1] - gNB_pos[i, 1]) / d_1[i]
            pp[i] = (E[i] ** 2 * ((pos[0] - gNB_pos[i, 0]) ** 2 + (pos[1] - gNB_pos[i, 1]) ** 2) - p[i] ** 2) / 2 / E[i] / d_0[i] + (((pos[0] - gNB_pos[i, 0]) ** 2 + (pos[1] - gNB_pos[i, 1]) ** 2) - d_1[i] ** 2) / 2 / d_1[i]

        pos_new = pos - np.linalg.pinv(a.T @ a) @ a.T @ pp

        for i in range(N):
            b[i] = (P0 - RSRP[i] - 10 * r * np.log(np.linalg.norm(pos_new - gNB_pos[i, :2]) / d0))
            beta[i] = toa[i] - np.linalg.norm(pos_new - gNB_pos[i, :2])
            if b[i] < 0:
                b[i] = 0
            if beta[i] < 0:
                beta[i] = 0

        if np.linalg.norm(pos - pos_new) < 0.01: #两次迭代差值较小时，停止迭代，跳出循环
            break
        pos = pos_new
    return pos


# 输入示例，以仿真数据集一条数据为例，包含18个基站坐标和对应的toa RSRP信息

gNB_pos =np.matrix( [
    [10, 10, 1.5],
    [10, 30, 1.5],
    [10, 50, 1.5],
    [30, 10, 1.5],
    [30, 30, 1.5],
    [30, 50, 1.5],
    [50, 10, 1.5],
    [50, 30, 1.5],
    [50, 50, 1.5],
    [70, 10, 1.5],
    [70, 30, 1.5],
    [70, 50, 1.5],
    [90, 10, 1.5],
    [90, 30, 1.5],
    [90, 50, 1.5],
    [110, 10, 1.5],
    [110, 30, 1.5],
    [110, 50, 1.5]
]
) 
toa=[26.8554687500000,	9.76562500000000,	19.5312500000000,	29.2968750000000,	12.2070312500000,	21.9726562500000,	39.0625000000000,	31.7382812500000,	43.9453125000000,	61.0351562500000,	51.2695312500000,	53.7109375000000,	104.980468750000,	83.0078125000000,	73.2421875000000,	119.628906250000,	100.097656250000,	97.6562500000000]
RSRP=[-71.1112870206190,	-68.4186504386199,	-70.5082379781484,	-78.6977184354546,	-67.0596635220034,	-78.6538712093793,	-76.9738806335884,	-69.2333447637259,	-78.0568132144033,	-86.9679618173754,	-77.0132682128579,	-80.7449071988934,	-80.3302799302913,	-84.5182608091890,	-79.5406098732805,	-88.8382875571822,	-85.7810620506808,	-91.3338746126288]

pos=ARC_Pos2D_TOA_RSS(gNB_pos,toa,RSRP)
print(pos)