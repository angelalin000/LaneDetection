import torch
import config
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from utils import *
import operator
from config import args_setting
from torch.nn import MultiheadAttention

def generate_model(args):

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model =UNet_TSM(config.img_channel, config.class_num).to(device)
    return model


class TSM(nn.Module):
    def __init__(self, channels, n_segment=5):
        super(TSM, self).__init__()
        self.channels = channels
        self.fold_div = 5
        self.fold = channels // self.fold_div
        self.n_segment = n_segment

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = self.fold

        # Shift left
        x1 = torch.cat((x[:, :-1, :fold], torch.zeros_like(x[:, -1:, :fold])), dim=1)
        # Shift right
        x2 = torch.cat((torch.zeros_like(x[:, :1, fold: 2 * fold]), x[:, 1:, fold: 2 * fold]), dim=1)
        # No shift
        x3 = x[:, :, 2 * fold:]
        out = torch.cat((x1, x2, x3), dim=2).view(nt, c, h, w)
        return out


class UNet_TSM(nn.Module):
    def __init__(self, n_channels, n_classes, n_segment=5):
        super(UNet_TSM, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.tsm = TSM(channels=256, n_segment=n_segment)

    def forward(self, x):
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            x1 = self.inc(item)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            data.append(x5.unsqueeze(0))
        data = torch.cat(data, dim=0)
        N, T, C, H, W = data.size()
        data = data.view(N * T, C, H, W)
        # 應用 TSM 模塊進行時序信息的捕捉
        data = self.tsm(data)
        
        # 使用最後一個時間步長的輸出
        # print(data.size())
        data = data.view(N, T, C, H, W)
        x = data[-1, :, :, :, :]
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x, data[-1]

