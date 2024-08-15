from geopy.distance import geodesic
import torch

# 定义经纬度的范围
MIN_LATITUDE, MAX_LATITUDE = -90, 90
MIN_LONGITUDE, MAX_LONGITUDE = -180, 180


def calc_geodesic_distance(y_hat, y):
    # 使用列表推导式计算所有样本的地理距离
    distance = torch.tensor([
        geodesic(
            (
                max(min(y_hat[i, 0].item(), MAX_LATITUDE), MIN_LATITUDE),
                max(min(y_hat[i, 1].item(), MAX_LONGITUDE), MIN_LONGITUDE)
            ),
            (y[i, 0].item(), y[i, 1].item())
        ).meters
        for i in range(y_hat.size(0))
    ])

    return distance
