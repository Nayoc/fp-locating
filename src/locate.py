import torch

import exec.mtrain as train
import net.mnn as mnn
from data.optimizer import MaxminNorm


def run(data: [],model_file):
    data = torch.Tensor(data).unsqueeze(dim=0)
    norm_x = MaxminNorm()
    norm_y = MaxminNorm(max=8,min=-1)

    wrapped_tensor = data.view(1, 1, 1, data.size(0))
    wrapped_tensor = norm_x.norm(wrapped_tensor)

    net = mnn.MCnn2(wrapped_tensor.shape)
    net = train.load_model(net, model_file)

    predications = train.calculate(net, train.try_gpu(), wrapped_tensor, norm_y)
    print(predications)
    return predications



if __name__ == '__main__':
    run([])
