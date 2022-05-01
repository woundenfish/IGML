import torch
from torch import nn


class more_focal_loss(nn.Module):

    def __init__(self, alpha=220, beta=20, gamma=100, theta=2):
        """
        alpha:weight of positive sample
        beta: weight of hard negative sample
        @type beta: object
        gamma = focal loss weight
        """
        super(more_focal_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta


    def forward(self, preds, labels):

        fn2 = self.gamma * (1 + labels) * labels * (labels - 1) * torch.pow(preds, self.theta) * torch.log(1.000001 - preds)
        fn1 = - self.beta / 10 * (2 + labels) * labels * (labels - 1) * torch.pow(preds, self.theta) * torch.log(1.000001 - preds)
        f0 =  self.beta * (2 + labels) * (labels + 1) * (labels - 1) * torch.pow(preds, self.theta) * torch.log(1.000001 - preds)
        fp1 = - self.alpha * (2 + labels) * labels * (labels + 1) * torch.pow(1 - preds, self.theta) * torch.log(preds + 0.000001)

        # print('fn2:', fn2)
        # print('fn1:', fn1)
        # print('f0:', f0)
        # print('fp1:', fp1)


        loss = fn2 + fn1 + f0 + fp1
        loss = loss.mean()
        # print('loss:', loss)
        # loss = loss.sum()
        return loss



if __name__ == '__main__':
    labels = torch.zeros(1, 12, 28, 28)
    labels[0][0][0][1] = -2
    labels[0][0][0][2] = 1
    labels[0][0][0][0] = 0

    labels[0][0][1][2] = -1

    preds = torch.zeros(1, 12, 28, 28) + 0.01
    print('preds:', preds)

    preds[0][0][0][1] = 0.01
    preds[0][0][0][2] = 0.01
    preds[0][0][0][0] = 0.01

    preds[0][0][1][2] = 0.01



    loss_fn = more_focal_loss()
    loss = loss_fn(preds, labels)
    print(loss)
