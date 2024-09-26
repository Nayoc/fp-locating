import torch
from torch import nn
import torch.utils.data as data
from view.mplt import Animator
import os
import util.coor_utils as cu
import logging

project_name = 'fp-locating'
lr_file = 'last_lr.txt'
cur_path = os.path.dirname(__file__)
root_path = cur_path[:cur_path.find(project_name) + len(project_name)]
model_path = root_path + '/models/'
batch_size_changes = {100: 64, 200: 32}  # 在第 100 和 200 epoch 改变 batch_size

# 坐标误差范围表示准确率
error_scale_3 = 3

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def extand(data, dim=1):
    # 将数据的通道维度扩展为 1
    return torch.unsqueeze(data, dim=dim)


# 返回四个维度的数据，规定误差内的准确数量，平均误差距离，最小误差距离，最大误差距离
def count_geodesic_distance(y_hat, y, norm, p=False):
    """计算预测正确的数量"""
    y_hat = norm.denorm(y_hat)
    y = norm.denorm(y)

    distance = cu.calc_geodesic_distance(y_hat, y)
    accuracy = (distance < error_scale_3).sum().item()

    return accuracy, distance.mean(), distance.min(), distance.max()


class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_result(net, device, data_iter, norm):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            distance = count_geodesic_distance(net(X), y, norm)
            metric.add(distance[0], y.numel() / 2)
    return metric[0] / metric[1], distance[1], distance[2], distance[3]


def train_epoch(net, device, train_iter, loss, updater, scheduler, label_norm):
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数、平均误差距离、最小误差距离、最大误差距离
    metric = Accumulator(3)
    for batch_idx, (X, y) in enumerate(train_iter):
        X, y = X.to(device), y.to(device)
        print(f"Batch {batch_idx}: Data is on device: {X.device}")

        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward(retain_graph=True)
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward(retain_graph=True)
            updater(X.shape[0])
        scheduler.step()
        distance = count_geodesic_distance(y_hat, y, label_norm)
        # y.numel()/2是因为最后距离是是坐标聚合出来的，所以总数只有一半
        metric.add(float(l.sum()), distance[0], y.numel() / 2)
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2], distance[1], distance[2], distance[3]


def judge_loss_weight(num):
    i = 0
    while num > 1 or num < 0.1:
        if num >= 1:
            num /= 10
            i += 1
            continue
        if num < 0.1:
            num *= 10
            i -= 1
            continue
        break

    return num, i


def train(net, train_iter, test_iter, loss, num_epochs, label_norm,
          model_file='fpcnn.params', record_term=100):
    # 加载上次保存的学习率，若没有则使用默认学习率
    lr = load_lr()
    print('current learning rate:' + str(lr))

    # 加载历史训练模型
    net = load_model(net, model_file)
    device, net = gpu_parallel(net)
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, step_size=50, gamma=0.1)

    # 权重系数，放大loss观察值
    global_loss_weight = 1
    term_loss_weight = 1

    # 预训练一次，确定损失数量级
    train_loss, train_acc, mean_error, min_error, max_error = train_epoch(net, device, train_iter, loss, trainer,
                                                                          scheduler,
                                                                          label_norm)

    animator_global = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                               legend=['train loss ', 'train acc', 'test acc'])
    animator_term = Animator(xlabel='epoch', xlim=[1, record_term], ylim=[0, 1],
                             legend=['train loss', 'train acc', 'test acc'])

    # 创建学习率调度器
    # scheduler = ExponentialLR(updater, gamma=0.9)

    # 打印当前学习率
    # current_lr = updater.state_dict()['param_groups'][0]['lr']

    """训练模型"""
    for epoch in range(num_epochs):

        train_loss, train_acc, train_mean_error, train_min_error, train_max_error = train_epoch(net, device, train_iter,
                                                                                                loss,
                                                                                                trainer, scheduler,
                                                                                                label_norm)
        test_acc, test_mean_error, test_min_error, test_max_error = evaluate_result(net, device, test_iter, label_norm)

        if epoch == 0:
            # 调整损失值到0.1-1区间方便观察
            global_weight_train_loss, _e = judge_loss_weight(train_loss)
            term_weight_train_loss = global_weight_train_loss
            global_loss_weight = 10 ** _e
            term_loss_weight = 10 ** _e
            _e = str(_e) if _e > 0 else '(' + str(_e) + ')'
            animator_global.rename_legend('train loss' + ' 10^' + _e, _index=0)
            animator_term.rename_legend('train loss' + ' 10^' + _e, _index=0)
        else:
            global_weight_train_loss = train_loss / global_loss_weight
            # 周期迭代记录一次文件并绘图——100次后记录并绘图
            if epoch % record_term == 0:
                term_weight_train_loss, _e = judge_loss_weight(train_loss)
                term_loss_weight = 10 ** _e
                animator_term.draw()
                animator_term.clear()
                _e = str(_e) if _e > 0 else '(' + str(_e) + ')'
                animator_term.rename_legend('train loss' + ' 10^' + _e, _index=0)
                save_model(net, model_file)
                print('---------current learning rate:' + str(trainer.param_groups[0]['lr']))

            else:
                term_weight_train_loss = train_loss / term_loss_weight

        animator_global.update(epoch + 1, (global_weight_train_loss, train_acc) + (test_acc,))
        animator_term.update(epoch % record_term + 1, (term_weight_train_loss, train_acc) + (test_acc,))
        torch.set_printoptions(sci_mode=False, precision=8)
        print(
            f'epoch:{epoch},loss:{round(train_loss, 8)},train_acc:{round(train_acc, 5)},test_acc:{round(test_acc, 5)} | '
            f'mean_error:{round(train_mean_error.item(), 5)}/{round(test_mean_error.item(), 5)},'
            f'min_error:{round(train_min_error.item(), 5)}/{round(test_min_error.item(), 5)},'
            f'max_error:{round(train_max_error.item(), 5)}/{round(test_max_error.item(), 5)}')

        # adjust_learning_rate(updater, epoch)
        # 每50 epoch调整batchsize

    animator_global.draw()

    # assert train_loss < 1, train_loss
    assert train_acc <= 1 and train_acc >= 0, train_acc
    assert test_acc <= 1 and test_acc >= 0, test_acc

    # 训练结束后，保存最终学习率到文件
    save_lr(scheduler.get_last_lr()[0])

    save_model(net, model_file)

    return train_loss, train_acc


def gpu_parallel(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用 DataParallel 来分配模型到多个 GPU
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    # 将模型移动到 GPU 上
    net = net.to(device)
    # 展示数据所在设备
    print('设备:' + "cuda————gpu_num:" + str(torch.cuda.device_count()) if torch.cuda.is_available() else "cpu")
    return device, net


def save_model(net, params_file='fpcnn.params'):
    torch.save(net.state_dict(), model_path + params_file)


def load_model(net, filename):
    try:
        net.load_state_dict(torch.load(model_path + filename))
        net.eval()
        print(filename + ' is loaded')
        return net
    except Exception as e:
        print('no model is loaded')
        print(e)
        return net


def save_lr(lr):
    with open(lr_file, 'w') as f:
        f.write(str(lr))


def load_lr():
    lr = 0.001
    if os.path.exists(lr_file):
        with open(lr_file, 'r') as f:
            lr = float(f.read().strip())
    return lr


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']

    if lr < 0.0001:
        return

    if epoch % 100 == 0 and epoch != 0:
        lr = lr * 0.9  # 学习率没100个epoch乘以0.9

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f'---learning rate change---,current lr is:{lr}')
