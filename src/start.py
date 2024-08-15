import torch
import exec.mtrain as train
import net.mnn as mnn
import data.build as build


def run_train(lr=0.001, batch_size=128, epochs=100, record_term=10):
    torch.set_printoptions(precision=8)
    ecnu_builder = build.EcnuDataBuilder()
    train_iter, validation_iter, test_iter = ecnu_builder.load_build(batch_size=batch_size)

    norm = ecnu_builder.label_norm
    net = mnn.MCNN1(ecnu_builder.train_data_tensor.shape)
    net = net.to(device=train.try_gpu())
    # 展示数据所在设备
    print('设备:' + str(next(net.parameters()).device))
    loss = mnn.CombinedLoss(norm)
    # trainer = torch.optim.SGD(net.parameters(), lr=lr)
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    data_name = ecnu_builder.data_name()

    model_file = data_name + '_' + type(net).__qualname__ + '.params'

    train.train(net, train_iter, validation_iter, loss, epochs, trainer, norm,
                model_file=model_file, record_term=record_term)

    print('ending training......')


if __name__ == '__main__':
    run_train(lr=0.001, batch_size=128, epochs=100, record_term=10)
