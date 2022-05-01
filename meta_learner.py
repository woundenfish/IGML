import torch
from torch import nn
from torch.nn import functional as F
from dataset4 import MetaDataset
from learner import Learner
from copy import deepcopy
from more_focal_loss import more_focal_loss
import time

class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        # self.n_way = args.n_wayc
        self.k_spt = args.k_shot
        self.k_qry = args.k_query
        self.task_num = args.task_num
        self.update_step_test = args.update_step_test
        self.obj_weight = args.obj_weight
        self.pos_weight = args.pos_weight
        self.neg1_weight = args.neg_1_weight
        self.neg2_weight = args.neg_2_weight
        self.inner_pos_weight = args.inner_pos_weight
        self.inner_neg1_weight = args.inner_neg_1_weight
        self.inner_neg2_weight = args.inner_neg_2_weight
        self.update_step_valid = args.update_step_test

        self.net = Learner(config)
        self.meta_optim = torch.optim.Adam([
            {'params': self.net.vars},
            {'params': self.net.adapted_head}
        ], lr=self.meta_lr)

    def forward(self, support, support_label, query, query_label):
        """
        :param support:         [b, setsz, c_, h, w]
        :param support_label:   [b, setsz, 48, 12, 12]
        :param query:           [b, querysz, c_, h, w]
        :param query_label:     [b, querysz, 48, 12, 12]
        :return:
        """
        pred = self.net(support)
        loss = self.grasp_loss(pred, support_label, inner=True)
#             print('---------inner loss:', loss)
        grad = torch.autograd.grad(loss, self.net.adapted_head, retain_graph=True, create_graph=True)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.adapted_head)))
        pred = self.net(query, fast_weights)
        loss_q = self.grasp_loss(pred, query_label, inner=False)
#             print('-----------outer loss:', loss)
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        loss_inner = loss
        loss_outer = loss_q
        return loss_inner, loss_outer

    def validation(self, support, support_label, query, query_label):
        """

        :param support:         [1, setsz, c_, h, w]
        :param support_label: [12, 14, 14]
        :param query:           [1, querysz, c_, h, w]
        :param query_label:     [1, querysz, 12, 14, 14]
        :return:qqq
        """

        valid_path = '/home/gwk/Test_Data'
        valid_dataset = MetaDataset(valid_path, self.k_spt, self.k_qry)
        net = deepcopy(self.net)
        # t1 = time.time()
        with torch.no_grad():
            # t1 = time.time()
            original_pred = net(query)
            # t2 = time.time()
            # print('time: ', t2 - t1)
        # t2 = time.time()
        t1 = time.time()
        pred = net(support)
        # t2 = time.time()
        loss = self.grasp_loss(pred, support_label, inner=True)
        print('inner_loss', loss)
        grad = torch.autograd.grad(loss, net.adapted_head)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.adapted_head)))
        t2 = time.time()
        print('time: ', t2 - t1)
        for k in range(1, self.update_step_valid):
            # support, support_label, query_2, query_label_2 = valid_dataset.__getitem__(0)
            # support = support.to('cuda:0')
            # support_label = support_label.to('cuda:0')
            pred = net(support, fast_weights)
            loss = self.grasp_loss(pred, support_label, inner=False)
            # print('loss_inner: ', k, ' ,',  loss)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            # with torch.no_grad():
            #     pred = self.net(query, fast_weights)
            #     loss = self.grasp_loss(pred, query_label, inner=False)  # mean loss of every shot
            #     print('loss_outer: ', k, ' ,', loss)
        torch.save(fast_weights, '/home/gwk/model/head_0.pth')
        with torch.no_grad():
            pred = self.net(query, fast_weights)
            loss = self.grasp_loss(pred, query_label, inner=False)  # mean loss of every shot
            # print('loss_outer: ', k, ' ,', loss)

        return loss, pred, original_pred



    def grasp_loss(self, output, label, inner=True):
        # device = torch.device('cuda')
        if inner:
            loss_fn = more_focal_loss(alpha=self.inner_pos_weight,
                                      beta=self.inner_neg1_weight,
                                      gamma=self.inner_neg2_weight
                                    )
        else:
            loss_fn = more_focal_loss(alpha=self.pos_weight,
                                  beta=self.neg1_weight,
                                  gamma=self.neg2_weight
                                  )
        # loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        return loss_fn(output, label)

    # def reg_loss(self, output, label, loss_weight):
    #     reg_loss = 0
    #     for i in range(5):
    #         tensor_loss = F.smooth_l1_loss(output[8 * i : 8 * (i + 1)], label[8 * i : 8 * (i + 1)], reduction='none')
    #         tensor_loss *= loss_weight
    #         # reg_loss += tensor_loss.mean()
    #         reg_loss += tensor_loss.sum()
    #     return reg_loss




