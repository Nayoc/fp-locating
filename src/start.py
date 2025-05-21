import torch

import data.build as build
import exec.mtrain as train
import net.mnn as mnn


def run_train(batch_size=128, epochs=100, record_term=10):
    # 数组显示8位
    torch.set_printoptions(precision=8)

    # 构建数据
    builder = build.CnnDataBuilder('syl')
    # builder = build.CnnDataBuilder('ecnu')
    x_norm, y_norm = builder.norm()
    builder.extend_channel()

    train_iter, validation_iter, test_iter = builder.load(batch_size=batch_size)

    net = mnn.MCnn1(builder.train_data.shape)
    loss = mnn.CombinedLoss(y_norm)

    data_name = builder.name

    model_file = data_name + '_' + type(net).__qualname__ + '.params'
    print('——————————————————————start training——————————————————————')
    train.train(net, train_iter, validation_iter, loss, epochs, y_norm,
                model_file=model_file, record_term=record_term)
    print('——————————————————————ending training——————————————————————')


if __name__ == '__main__':
    run_train(batch_size=128, epochs=300, record_term=50)
