import torch
from torch import nn
from torch.nn import functional as F
import numpy as np



class Learner(nn.Module):
    """

    """

    def __init__(self, config):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()

        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.adapted_head = nn.ParameterList()
        self.vars_bn = nn.ParameterList()
        self.idx = 0
        self.bn_idx = 0
        self.head_idx = 0

        # for i, (name, param) in enumerate(self.depth_config):
        #     self.make_layer(name, param)
        for i, (name, param) in enumerate(self.config):
            self.make_layer(name, param)



    def forward(self, x, adapt_vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x:
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if adapt_vars is None:
            adapt_vars = self.adapted_head

        self.idx = 0
        self.bn_idx = 0
        self.head_idx = 0
        # self.head_bn_idx = 0
        # for name, param in self.depth_config:
        #     x_depth = self.layer_forward(x_depth, vars, name, param)

        for name, param in self.config:
            if name is not 'adapted_head':
                x = self.layer_forward(x, self.vars, name, param)
            else:
                x = self.layer_forward(x, adapt_vars, name, param)
        return x


    def make_layer(self, name, param):
        if name is 'conv2d':
            # [ch_out, ch_in, kernelsz, kernelsz]
            w = nn.Parameter(torch.ones(*param[:4]))
            # gain=1 according to cbfin's implementation
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            # [ch_out]
            self.vars.append(nn.Parameter(torch.zeros(param[0])))
            # print('w  conv2d.shape:', w.shape)

        elif name is 'adapted_head':
            w = nn.Parameter(torch.ones(*param[:4]))
            # gain=1 according to cbfin's implementation
            torch.nn.init.kaiming_normal_(w)
            self.adapted_head.append(w)
            # [ch_out]
            self.adapted_head.append(nn.Parameter(torch.zeros(param[0])))

        elif name is 'linear':
            # [ch_out, ch_in]
            w = nn.Parameter(torch.ones(*param))
            # gain=1 according to cbfinn's implementation
            torch.nn.init.kaiming_normal_(w)
            self.vars.append(w)
            # [ch_out]
            self.vars.append(nn.Parameter(torch.zeros(param[0])))

        elif name is 'bn':
            # [ch_out]
            w = nn.Parameter(torch.ones(param[0]))
            self.vars.append(w)
            # [ch_out]
            self.vars.append(nn.Parameter(torch.zeros(param[0])))

            # must set requires_grad=False
            running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
            running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
            self.vars_bn.extend([running_mean, running_var])
            # print('w  bn.shape:', w.shape)

        elif name is 'res_bottleneck':
            # print('make layer res_bottleneck')
            # param[0] :c_out  |   param[1] :c_in
            self.make_layer('CBL', [param[0], param[1], 1, 1, 2, 0])  # down sample
            mid_channel = int(param[0] / 4)
            self.make_layer('CBL', [mid_channel, param[1], 1, 1, 1, 0])
            self.make_layer('CBL', [mid_channel, mid_channel, 3, 3, 2, 1])  # down sample
            self.make_layer('CBL', [param[0], mid_channel, 1, 1, 1, 0])

        elif name is 'res_identity':
            # print('make layer res_identity')
            # param[0] :c_out  |   param[1] :c_in
            self.make_layer('CBL', [param[0], param[1], 1, 1, 1, 0])
            mid_channel = int(param[0] / 4)
            self.make_layer('CBL', [mid_channel, param[1], 1, 1, 1, 0])
            self.make_layer('CBL', [mid_channel, mid_channel, 3, 3, 1, 1])
            self.make_layer('CBL', [param[0], mid_channel, 1, 1, 1, 0])

        elif name is 'res_conv2':
            self.make_layer('res_identity', [256, 64])
            self.make_layer('res_identity', [256, 256])
            self.make_layer('res_identity', [256, 256])

        elif name is 'res_conv3':
            self.make_layer('res_bottleneck', [512, 256])
            self.make_layer('res_identity', [512, 512])
            self.make_layer('res_identity', [512, 512])
            self.make_layer('res_identity', [512, 512])

        elif name is 'res_conv4':
            self.make_layer('res_bottleneck', [1024, 512])
            self.make_layer('res_identity', [1024, 1024])
            self.make_layer('res_identity', [1024, 1024])
            self.make_layer('res_identity', [1024, 1024])
            self.make_layer('res_identity', [1024, 1024])
            self.make_layer('res_identity', [1024, 1024])

        elif name is 'res_conv5':
            self.make_layer('res_bottleneck', [2048, 1024])
            self.make_layer('res_identity', [2048, 2048])
            self.make_layer('res_identity', [2048, 2048])

        elif name is 'CBL':
            self.make_layer('conv2d', param)
            self.make_layer('bn', param)


    def layer_forward(self, x, vars, name, param):

        if name is 'conv2d':
            w, b = vars[self.idx], vars[self.idx + 1]
            x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
            self.idx += 2
        elif name is 'linear':
            w, b = vars[self.idx], vars[self.idx + 1]
            x = F.linear(x, w, b)
            self.idx += 2
        elif name is 'adapted_head':
            w, b = vars[self.head_idx], vars[self.head_idx + 1]
            x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
            self.head_idx += 2

        elif name is 'bn':
            w, b = vars[self.idx], vars[self.idx + 1]
            running_mean, running_var = self.vars_bn[self.bn_idx], self.vars_bn[self.bn_idx+1]
            x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=True)
            self.idx += 2
            self.bn_idx += 2

        elif name is 'CBL':
            x = self.layer_forward(x, vars, 'conv2d', param)
            x = self.layer_forward(x, vars, 'bn', param)
            x = self.layer_forward(x, vars, 'leaky_relu', [])

        elif name is 'res_bottleneck':
            x_ = x
            x_ = self.layer_forward(x_, vars, 'CBL', [param[0], param[1], 1, 1, 2, 0])
            mid_channel = int(param[0] / 4)
            x = self.layer_forward(x, vars, 'CBL', [mid_channel, param[1], 1, 1, 1, 0])
            x = self.layer_forward(x, vars, 'CBL', [mid_channel, mid_channel, 3, 3, 2, 1])
            x = self.layer_forward(x, vars, 'conv2d', [param[0], mid_channel, 1, 1, 1, 0])
            x = self.layer_forward(x, vars, 'bn', [param[0]])
            x += x_
            x = self.layer_forward(x, vars, 'leaky_relu', [])

        elif name is 'res_identity':

            x_ = x
            # print('before  x_.shape:', x_.shape)
            # print('before x:', x.shape)
            x_ = self.layer_forward(x, vars, 'CBL', [param[0], param[1], 1, 1, 1, 0])

            mid_channel = int(param[0] / 4)
            x = self.layer_forward(x, vars, 'CBL', [mid_channel, param[1], 1, 1, 1, 0])
            x = self.layer_forward(x, vars, 'CBL', [mid_channel, mid_channel, 3, 3, 1, 1])
            x = self.layer_forward(x, vars, 'conv2d', [param[0], mid_channel, 1, 1, 1, 0])
            x = self.layer_forward(x, vars, 'bn', [param[0]])
            # print('x_.shape:', x_.shape)
            # print('x:', x.shape)
            x += x_
            # print('new x', x.shape)
            x = self.layer_forward(x, vars, 'leaky_relu', [])

        elif name is 'res_conv2':
            x = self.layer_forward(x, vars, 'res_identity', [256, 64])
            x = self.layer_forward(x, vars, 'res_identity', [256, 256])
            x = self.layer_forward(x, vars, 'res_identity', [256, 256])


        elif name is 'res_conv3':
            x = self.layer_forward(x, vars, 'res_bottleneck', [512, 256])
            x = self.layer_forward(x, vars, 'res_identity', [512, 512])
            x = self.layer_forward(x, vars, 'res_identity', [512, 512])
            x = self.layer_forward(x, vars, 'res_identity', [512, 512])

        elif name is 'res_conv4':
            x = self.layer_forward(x, vars, 'res_bottleneck', [1024, 512])
            x = self.layer_forward(x, vars, 'res_identity', [1024, 1024])
            x = self.layer_forward(x, vars, 'res_identity', [1024, 1024])
            x = self.layer_forward(x, vars, 'res_identity', [1024, 1024])
            x = self.layer_forward(x, vars, 'res_identity', [1024, 1024])
            x = self.layer_forward(x, vars, 'res_identity', [1024, 1024])


        elif name is 'res_conv5':

            x = self.layer_forward(x, vars, 'res_bottleneck', [2048, 1024])
            x = self.layer_forward(x, vars, 'res_identity', [2048, 2048])
            x = self.layer_forward(x, vars, 'res_identity', [2048, 2048])

        elif name is 'flatten':
            x = x.view(x.size(0), -1)
        elif name is 'relu':
            x = F.relu(x, inplace=param[0])
        elif name is 'leaky_relu':
            x = F.leaky_relu(x)
        elif name is 'tanh':
            x = F.tanh(x)
        elif name is 'sigmoid':
            x = torch.sigmoid(x)
        elif name is 'max_pool2d':
            x = F.max_pool2d(x, param[0], param[1], param[2])
        elif name is 'avg_pool2d':
            x = F.avg_pool2d(x, param[0], param[1], param[2])

        return x




    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars