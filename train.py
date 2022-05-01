import torch
from dataset import MetaDataset
from torch.utils.data import DataLoader
import argparse
from torch.nn import functional as F
import random
from meat_learner import Meta
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import cv2


def train_grasp():
    device = torch.device('cuda:0')
    args = Argparser1()

    config = [
        ('CBL', [64, 4, 7, 7, 2, 3]),
        ('max_pool2d', [3, 2, 1]),
        ('res_conv2', [256, 64]),
        ('res_conv3', [512, 256]),
        ('res_conv4', [1024, 512]),
        ('adapted_head', [256, 1024, 3, 3, 1, 1]),
        ('leaky_relu', [True]),
        ('adapted_head', [18, 256, 1, 1, 1, 0]),
        ('sigmoid', [])
    ]
    maml = Meta(args, config).to(device)
    #     check_point = torch.load('./model/train.pth')
    #     maml.load_state_dict(check_point['net'])
    #     maml.meta_optim.load_state_dict(check_point['optimizer'])

    train_path = '/home/gwk/Data'
    valid_path = '/home/gwk/Data'
    train_dataset = MetaDataset(train_path, args.k_shot, args.k_query)

    losses_train = []
    losses_valid = []
    for fake_epoch in range(0, args.epoch):
        print('---------------epoch-----------------:', fake_epoch)
        if fake_epoch % 30 == 20:
            state_dict = {'net': maml.state_dict(), 'optimizer': maml.meta_optim.state_dict(), 'epoch': fake_epoch}
            torch.save(state_dict, './model/train_18.pth')

        for i in random.sample(range(0, 99), 98):
            support_input, support_cate, query_input, query_cate = train_dataset.__getitem__(i)
            support_input = support_input.to('cuda:0')
            support_cate = support_cate.to('cuda:0')
            query_input = query_input.to('cuda:0')
            query_cate = query_cate.to('cuda:0')
            loss = maml(support_input, support_cate, query_input, query_cate)

        if fake_epoch % 30 == 20:
            for i in random.sample(range(0, 99), 3):
                support_input, support_cate, query_input, query_cate = train_dataset.__getitem__(i)
                support_input = support_input.to('cuda:0')
                support_cate = support_cate.to('cuda:0')
                query_input = query_input.to('cuda:0')
                query_cate = query_cate.to('cuda:0')
                loss, prediction, inner_prediction = maml.validation(support_input, support_cate, query_input, query_cate)
                print('-----in  validation----------------')

def show_output(masks, m):
    plt.figure('masks', figsize=(16, 6))
    for i in range(0, 18):
        plt.subplot(3, 6, i + 1)
        plt.imshow(masks[m, i])
    plt.show()

def plot_prediction(support_input, support_cate, query_input, query_cate, prediction, inner_prediction):
    support_input = support_input.cpu()
    support_cate = support_cate.cpu()
    query_input = query_input.cpu()
    query_cate = query_cate.cpu()
    prediction = prediction.cpu()
    inner_prediction = inner_prediction.cpu()
    support_input = support_input.permute(0, 2, 3, 1)
    query_input = query_input.permute(0, 2, 3, 1)
    support_cate_gt = torch.sum(support_cate, dim=1)
    query_cate_gt = torch.sum(query_cate, dim=1)
    print('------SUPPORT----------')
    plt.figure('support', figsize=(4, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(support_input[0, :, :, 0:3])
    plt.subplot(1, 2, 2)
    plt.imshow(support_cate_gt[0], cmap='gray')
    plt.show()
    show_output(inner_prediction, 0)
    print('------QUERY----------')
    for i in range(2):
        plt.figure('query', figsize=(4, 12))
        plt.subplot(1, 2, 1)
        plt.imshow(query_input[i, :, :, 0:3])
        plt.subplot(1, 2, 2)
        plt.imshow(query_cate_gt[i], cmap='gray')
        plt.show()
        show_output(prediction, i)
        print('------------------------')







class Argparser1():

    def __init__(self):
        self.epoch = 700
        self.k_shot = 1
        self.k_query = 4
        self.task_num = 1
        self.meta_lr = 0.0015
        self.update_lr = 0.06  # 0.06
        self.obj_weight = 1
        self.pos_weight = 15
        self.neg_1_weight = 2
        self.neg_2_weight = 10
        self.inner_pos_weight = 100
        self.inner_neg_1_weight = 1
        self.inner_neg_2_weight = 1
        self.update_step_test = 1


if __name__ == '__main__':
    train_grasp()

