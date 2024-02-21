'''
    Building Model  : MCNN 모델 정의
'''
'''metrix, config_optimizer 수정해보기'''

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

# sub model
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        # self.conv = []
        # if dilation==1:
        #     self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        # else:
        #     self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else nn.Identity()
        # if NL == 'relu' :
        #     self.relu = nn.ReLU(inplace=True)
        # elif NL == 'prelu':
        #     self.relu = nn.PReLU()
        # else:
        #     self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# Multi-column CNN 
# Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
class MCNN(LightningModule):
    def __init__(self, lr, max_steps, bn=False):
        super(MCNN, self).__init__()
        self.save_hyperparameters() # self.hparam을 사용하게 해줌; model에 입력받은 args
        self.use = 0    # default : MCNN 전체
        self.lr = lr
        self.crit = nn.MSELoss()
        
        self.branch1 = nn.Sequential(Conv2d( 1, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16,  8, 7, same_padding=True, bn=bn))
        
        self.branch2 = nn.Sequential(Conv2d( 1, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))
        
        self.branch3 = nn.Sequential(Conv2d( 1, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))
        
        self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=bn))  # MCNN output
        
        self.out1 = nn.Sequential(Conv2d( 8, 1, 1, same_padding=True, bn=bn))   # 각 column(CNN) ouput
        self.out2 = nn.Sequential(Conv2d( 10, 1, 1, same_padding=True, bn=bn))
        self.out3 = nn.Sequential(Conv2d( 12, 1, 1, same_padding=True, bn=bn))
        
    def forward(self, im_data):
        im_data = im_data.unsqueeze(1) # 차원 생성, (batch, h, w) -> (batch, n, h, w)
        
        '''self.use에 따라 필요없는 x1, x2, x3이 존재하는데..????????????????////'''
        x1 = self.branch1(im_data)  # shape [1, 8, 192, 256]
        x2 = self.branch2(im_data)  # shape [1, 10, 192, 256]
        x3 = self.branch3(im_data)  # shape [1, 12, 192, 256]
        # print(f'x1:{x1.shape}, x2:{x2.shape}, x3:{x3.shape}')
        
        if self.use == 0: # MCNN
            x = torch.cat((x1,x2,x3),1) # shape [1, 30, 192, 256]
            x = self.fuse(x)            # shape [1, 1, 192, 256]
            # print(f'x:{x.shape}')

        elif self.use == 1: # branch 1
            x = self.out1(x1)
        elif self.use == 2: # branch 2
            x = self.out2(x2)
        elif self.use == 3: # branch 3
            x = self.out3(x3)
        
        return x.squeeze(1) # 차원 제거 [1, 192, 256]
    
    '''train 시간이 너무 오래 걸리면 필요없는 metrix 빼도 될 듯'''
    def training_step(self, batch, batch_idx): # 단일배치 훈련
        self.train()
        x, y = batch    # img, dm
        
        pred = self(x)
        loss = self.crit(pred, y)
        
        pred_sum = torch.round(pred.sum(dim=(1,2))).int()
        gt_sum = torch.round(y.sum(dim=(1,2))).int()
        acc = (pred_sum == gt_sum).float().mean()
        
        mae = torch.abs(pred_sum - gt_sum).float().mean()
        
        self.log('train_loss', loss) # mse
        self.log('train_acc', acc)
        self.log('train_mae', mae)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.eval()
            x, y = batch
            
            pred = self(x)
            loss = self.crit(pred, y)
        
            pred_sum = torch.round(pred.sum(dim=(1,2))).int()
            gt_sum = torch.round(y.sum(dim=(1,2))).int()
            acc = (pred_sum == gt_sum).float().mean()

            mae = torch.abs(pred_sum - gt_sum).float().mean()
            
            self.log('val_loss', loss)
            self.log('val_acc', acc)
            self.log('val_mae', mae)
            
    '''더 공부해보기/!!!!!!!'''
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.hparams.max_steps, pct_start=0.1, cycle_momentum=False), # , total_steps=self.hparams.max_steps
            'interval': 'step', # step
            'frequency': 1
        }
        
        return [optimizer], [scheduler]