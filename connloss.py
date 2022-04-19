import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import ReLU
import sys

class ConnLoss(nn.Module):
    def __init__(self, batch=True):
        super(ConnLoss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        self.down_time = 5 #m-1

    def downsample(self, tensor):
        # tensor: N x C x H x W
        # maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        return maxpool(tensor)

    def one_padding(self, tensor):
        n, h, w = tensor.size()
        tensor_big = torch.ones(n,h+2,w+2).cuda()
        for i in range(n):
            tensor_big[i][1:h+1, 1:w+1] = tensor[i]
        return tensor_big

    def conn_matrix(self, prob_mat, num_neighbor=8):
        # prob_mat: y_pred or y_true
        n, h, w = prob_mat.size()
        prob_mat_big = self.one_padding(prob_mat)
        cm_u = torch.zeros(n, h, w).cuda()
        cm_l = torch.zeros(n, h, w).cuda()
        cm_r = torch.zeros(n, h, w).cuda()
        cm_d = torch.zeros(n, h, w).cuda()
        for i in range(n):
            cm_u[i] = prob_mat[i] * prob_mat_big[i][0:h, 1:w+1]
            cm_l[i] = prob_mat[i] * prob_mat_big[i][1:h+1, 0:w]
            cm_r[i] = prob_mat[i] * prob_mat_big[i][1:h+1, 2:w+2]
            cm_d[i] = prob_mat[i] * prob_mat_big[i][2:h+2, 1:w+1]
        cm_add = cm_u + cm_l + cm_r + cm_d
        if num_neighbor==4:
            return cm_add/num_neighbor
        elif num_neighbor==8:
            cm_ul = torch.zeros(n, h, w).cuda()
            cm_ur = torch.zeros(n, h, w).cuda()
            cm_dl = torch.zeros(n, h, w).cuda()
            cm_dr = torch.zeros(n, h, w).cuda()
            for i in range(n):
                cm_ul[i] = prob_mat[i] * prob_mat_big[i][0:h, 0:w]
                cm_ur[i] = prob_mat[i] * prob_mat_big[i][0:h, 2:w+2]
                cm_dl[i] = prob_mat[i] * prob_mat_big[i][2:h+2, 0:w]
                cm_dr[i] = prob_mat[i] * prob_mat_big[i][2:h+2, 2:w+2]
            cm_add = cm_add + cm_ul + cm_ur + cm_dl + cm_dr
            return cm_add/num_neighbor
        else:
            raise SettingError("The num_neighbor should be 4 or 8!")

    def my_l1_loss(self, y_pred, y_true):
        loss = 0
        mask0 = y_true.eq(0).type(torch.cuda.FloatTensor)
        mask1 = y_true.gt(0).type(torch.cuda.FloatTensor)
        mask0_sum = torch.sum(mask0)
        mask1_sum = torch.sum(mask1)
        weight = (mask1_sum/mask0_sum).item()
        y_pred_w = torch.where(y_true>0,y_pred,y_pred*weight)
        loss = F.l1_loss(y_pred_w, y_true)
        return loss
    
    def compute_conn_loss(self, y_pred, y_true):
        conn_pred = self.conn_matrix(y_pred.squeeze(1))
        conn_true = self.conn_matrix(y_true.squeeze(1))
        loss = self.my_l1_loss(conn_pred.unsqueeze(1), conn_true.unsqueeze(1))
        return loss

    def __call__(self, y_pred, y_true):
        now_pred = y_pred
        now_true = y_true
        loss = self.compute_conn_loss(now_pred, now_true)
        alfa = 0.5
        for i in range(self.down_time):
            now_pred = self.downsample(now_pred)
            now_true = self.downsample(now_true)
            now_loss = self.compute_conn_loss(now_pred, now_true)
            loss = loss + now_loss*(alfa**(i+1))
        return loss/((1-alfa**(self.down_time+1))/(1-alfa))

