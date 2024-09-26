import torch
import exec.mtrain as train
import net.mnn as mnn
import data.build as build


def run_train(batch_size=128, epochs=100, record_term=10):
    # 数组显示8位
    torch.set_printoptions(precision=8)

    # 构建数据
    ecnu_builder = build.EcnuDataBuilder()
    train_iter, validation_iter, test_iter = ecnu_builder.load_build(batch_size=batch_size)

    norm = ecnu_builder.label_norm
    net = mnn.MCNN1(ecnu_builder.train_data_tensor.shape)
    loss = mnn.CombinedLoss(norm)

    data_name = ecnu_builder.data_name()

    model_file = data_name + '_' + type(net).__qualname__ + '.params'
    print('——————————————————————start training——————————————————————')
    train.train(net, train_iter, validation_iter, loss, epochs, norm,
                model_file=model_file, record_term=record_term)
    print('——————————————————————ending training——————————————————————')


if __name__ == '__main__':
    run_train(batch_size=128, epochs=300, record_term=50)
