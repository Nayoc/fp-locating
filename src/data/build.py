from pathlib import Path

import torch
import torch.utils.data as data

from data.optimizer import ZScoreNorm

root_dir = str(Path(__file__).parent.parent.parent)
cnn_dataset_dir = root_dir + '/data/dataset/cnn/'


class CnnDataBuilder:
    def __init__(self, name: str):
        self.name = name
        dir = cnn_dataset_dir + name

        train = torch.load(dir + '/train.pth')
        validation = torch.load(dir + '/valid.pth')
        test = torch.load(dir + '/test.pth')

        self.train_data = train['data']
        self.validation_data = validation['data']
        self.test_data = test['data']

        self.train_label = train['labels']
        self.validation_label = validation['labels']
        self.test_label = test['labels']

    def norm(self):
        znorm_x = ZScoreNorm()
        znorm_y = ZScoreNorm()

        self.train_data = znorm_x.fit_norm(self.train_data)
        self.validation_data = znorm_x.norm(self.validation_data)
        self.test_data = znorm_x.norm(self.test_data)

        self.train_label = znorm_y.fit_norm(self.train_label)
        self.validation_label = znorm_y.norm(self.validation_label)
        self.test_data = znorm_y.norm(self.test_label)

        return znorm_x, znorm_y

    def extend_channel(self):
        if len(self.train_data.shape) == 3:
            self.train_data = torch.unsqueeze(self.train_data, dim=1)
            self.validation_data = torch.unsqueeze(self.validation_data, dim=1)
            self.test_data = torch.unsqueeze(self.test_data, dim=1)

    def load(self, batch_size=64):
        self.extend_channel()

        train_set = data.TensorDataset(self.train_data, self.train_label)
        validation_set = data.TensorDataset(self.validation_data, self.validation_label)
        test_set = data.TensorDataset(self.test_data, self.test_label)

        train_iter = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                     pin_memory=True)
        validation_iter = data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                          pin_memory=True)
        test_iter = data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                    pin_memory=True)

        return train_iter, validation_iter, test_iter

# class SignalData:
#     def __init__(self):
#         self.rsrp_min = -140
#         self.rsrp_max = -44
#         self.rsrq_min = -20
#         self.rsrq_max = 0
#         self.sinr_min = -10
#         self.sinr_max = 30
#         self.longitude_min = 180
#         self.longitude_max = 0
#         self.latitude_min = 0
#         self.latitude_max = 90
#
#
# class EcnuDataBuilder(SignalData):
#     def __init__(self):
#         super(EcnuDataBuilder, self).__init__()
#
#         self.file_name_suffix = root_dir + '/data/dataset/ecnu/'
#         self.train_file = 'train.pth'
#         self.validation_file = 'valid.pth'
#         self.test_file = 'test.pth'
#         self.data_norm = optimizer.Norm(torch.tensor(
#             [[self.rsrp_min, self.rsrp_max], [self.rsrq_min, self.rsrq_max], [self.sinr_min, self.sinr_max]]))
#         self.label_norm = optimizer.Norm(
#             torch.tensor([[self.longitude_min, self.longitude_max], [self.latitude_min, self.latitude_max]]))
#
#         try:
#             train = torch.load(self.file_name_suffix + self.train_file)
#             validation = torch.load(self.file_name_suffix + self.validation_file)
#             test = torch.load(self.file_name_suffix + self.test_file)
#
#             self.train_data = train['data']
#             self.validation_data = validation['data']
#             self.test_data = test['data']
#
#             self.train_label = train['labels']
#             self.validation_label = validation['labels']
#             self.test_label = test['labels']
#         except:
#             d1, l1 = self.data_query(root_dir + '/data/original/ecnu/0602-104700_UE1_source_0606150621-12964.csv')
#             d2, l2 = self.data_query(root_dir + '/data/original/ecnu/0531-164319_UE1_source_0606150615-3350.csv')
#             d3, l3 = self.data_query(root_dir + '/data/original/ecnu/0601-172142_UE1_source_0606150619-12933.csv')
#             d4, l4 = self.data_query(root_dir + '/data/original/ecnu/0602-113342_UE1_source_0606150623-9649.csv')
#             d5, l5 = self.data_query(root_dir + '/data/original/ecnu/0531-165707_UE1_source_0606150615-407.csv')
#             d6, l6 = self.data_query(root_dir + '/data/original/ecnu/0601-170116_UE1_source_0606150618-6128.csv')
#             d7, l7 = self.data_query(root_dir + '/data/original/ecnu/验证集0601-161411_UE1_source_0606150616-13752.csv')
#             d8, l8 = self.data_query(root_dir + '/data/original/ecnu/测试集0602-120648_UE1_source_0606150624-7014.csv')
#
#             train_data = d1 + d2 + d3 + d4 + d5 + d6
#             train_label = l1 + l2 + l3 + l4 + l5 + l6
#
#             validation_data = d7
#             validation_label = l7
#
#             test_data = d8
#             test_label = l8
#
#             # 因为时间维度都是单时间点，因此所有数据需要扩展纬度
#             train_data = torch.tensor(train_data)
#             train_data = torch.unsqueeze(train_data, dim=2)
#             train_label = torch.tensor(train_label)
#             print(train_data.shape, train_label.shape)
#
#             validation_data = torch.tensor(validation_data)
#             validation_data = torch.unsqueeze(validation_data, dim=2)
#             validation_label = torch.tensor(validation_label)
#             print(validation_data.shape, validation_label.shape)
#
#             test_data = torch.tensor(test_data)
#             test_data = torch.unsqueeze(test_data, dim=2)
#             test_label = torch.tensor(test_label)
#             print(test_data.shape, test_label.shape)
#
#             # 归一化数据
#             self.train_data = self.data_norm.norm(train_data)
#             self.validation_data = self.data_norm.norm(validation_data)
#             self.test_data = self.data_norm.norm(test_data)
#
#             self.train_label = self.label_norm.norm(train_label)
#             self.validation_label = self.label_norm.norm(validation_label)
#             self.test_label = self.label_norm.norm(test_label)
#
#             torch.save({'data': self.train_data, 'labels': self.train_label},
#                        self.file_name_suffix + self.train_file)
#             torch.save({'data': self.validation_data, 'labels': self.validation_label},
#                        self.file_name_suffix + self.validation_file)
#             torch.save({'data': self.test_data, 'labels': self.test_label},
#                        self.file_name_suffix + self.test_file)
#
#         train_set = data.TensorDataset(self.train_data, self.train_label)
#         validation_set = data.TensorDataset(self.validation_data, self.validation_label)
#         test_set = data.TensorDataset(self.test_data, self.test_label)
#
#     def data_name(self):
#         return '闵行校测绕行数据'
#
#     def load_build(self, batch_size=64):
#         train_iter = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
#                                      pin_memory=True)
#         validation_iter = data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=8,
#                                           pin_memory=True)
#         test_iter = data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8,
#                                     pin_memory=True)
#
#         return train_iter, validation_iter, test_iter
#
#     def data_query(self, file_path: str, pcell_num=7, ncell_num=4):
#         file_path = file_path
#         data = pd.read_csv(file_path)
#
#         longitude = data['Longitude'].tolist()
#         latitude = data['Latitude'].tolist()
#
#         rsrp = {}
#         rsrq = {}
#         sinr = {}
#         for i in range(1, pcell_num + 1):
#             rsrp[i] = data['PCell' + str(i) + ' -Beam SS-RSRP'].tolist()
#             rsrq[i] = data['PCell' + str(i) + ' -Beam SS-RSRQ'].tolist()
#             sinr[i] = data['PCell' + str(i) + ' -Beam SS-SINR'].tolist()
#
#         for i in range(1, ncell_num + 1):
#             rsrp[i + pcell_num] = data['NCell' + str(i) + ' -Beam SS-RSRP'].tolist()
#             rsrq[i + pcell_num] = data['NCell' + str(i) + ' -Beam SS-RSRQ'].tolist()
#             sinr[i + pcell_num] = data['NCell' + str(i) + ' -Beam SS-SINR'].tolist()
#
#         data_length = max(len(rsrq[1]), len(longitude))
#
#         _data = []
#         _label = []
#
#         for i in range(data_length):
#             # 删除无坐标数据
#             if np.isnan(longitude[i]) or np.isnan(latitude[i]):
#                 continue
#
#             # 无信号设置为最低值
#             data_rsrp = []
#             for k, v in rsrp.items():
#                 if np.isnan(v[i]):
#                     v[i] = self.rsrp_min
#                 data_rsrp.append(v[i])
#
#             data_rsrq = []
#             for k, v in rsrq.items():
#                 if np.isnan(v[i]):
#                     v[i] = self.rsrq_min
#                 data_rsrq.append(v[i])
#
#             data_sinr = []
#             for k, v in sinr.items():
#                 if np.isnan(v[i]):
#                     v[i] = self.sinr_min
#                 data_sinr.append(v[i])
#
#             _data_one = []
#             _data_one.append(data_rsrp)
#             _data_one.append(data_rsrq)
#             _data_one.append(data_sinr)
#             _data.append(_data_one)
#             _label.append((latitude[i], longitude[i]))
#
#         return _data, _label
