import torch
import torch.nn as nn
from torch import optim


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tensora = nn.Parameter(torch.FloatTensor([[2, 3], [3, 4]]))
        self.tensorb = torch.ones((3, 2)).scatter_(0, torch.LongTensor([[0, 0], [1, 1]]), self.tensora)
        # self.tensorb = nn.Parameter(torch.randn((3,2)))
        # self.tensorc = torch.cat([self.tensora, self.tensorb], 0)
        self.tensorb.scatter_(0, torch.LongTensor([[0, 0], [1, 1]]), self.tensora)
    def forward(self, ref):
        # self.renderer.eye = nr.get_points_from_angles(2.732, 0, np.random.uniform(0, 360))
        # src = self.tensorb.scatter_(0, torch.LongTensor([[0, 0], [1, 1]]), self.tensora)
        # print(src)
        v = self.tensora[0]
        print(v)
        loss = torch.sum((v - ref) ** 2)
        return loss


def main():
    ref = torch.rand((2))
    print(ref)
    model = Model()
    # pdb.set_trace()
    for name, param in model.named_parameters():
        # if param.requires_grad:
        print(name)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True)
    for i in range(100):
        loss = model(ref)
        print(loss)
        optimizer.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward
        optimizer.step()
        # print(model.tensora)


if __name__ == '__main__':
    main()
