import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange


# Define necessary modules

class Depwise_Block(nn.Module):
    def __init__(self,in_channels,out_channels,dilation=1):
        super().__init__()
        self.out =  nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,groups=in_channels,dilation=dilation,padding='same'),
            nn.Conv2d(in_channels,out_channels,1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self,x):
        return self.out(x)

class Residual_Block(nn.Module):
    def __init__(self,in_channels,out_channels,num_conv=2,last=False):
        super().__init__()
        self.change_channels = nn.Conv2d(in_channels,out_channels,3,padding='same')
     
        model_list=[
           Depwise_Block(in_channels,out_channels)
        ]
        for _ in range(1,num_conv):
            model_list.extend([Depwise_Block(out_channels,out_channels)])
        merge_lst = [nn.Conv2d(2*out_channels,out_channels,3,padding='same'),] 
        if last == False:
            merge_lst.append(nn.ReLU())
        merge_lst.append(nn.BatchNorm2d(out_channels))
        self.merge = nn.Sequential(*merge_lst
        )
        self.out = nn.Sequential(
            *model_list
        )
    def forward(self,x):
        x0=self.change_channels(x)
        x1=self.out(x)
        return  self.merge(torch.cat((x0,x1),1))

class Inception_Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.branch_1= nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        self.branch_2= nn.Sequential(
            nn.Conv2d(in_channels,out_channels,(1,5),padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,out_channels,(5,1),padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        self.branch_3= Depwise_Block(in_channels,out_channels,2)
        self.branch_4= Depwise_Block(in_channels,out_channels,3)
        self.merge = Depwise_Block(2*out_channels,out_channels)

    def forward(self,x):
        b1=self.branch_1(x)
        b2=self.branch_2(x)
        b3=self.branch_3(x)
        b4=self.branch_4(x)
        return  self.merge(torch.cat(((b1+b2),(b3+b4)),1))

class SE_Block(nn.Module):
    def __init__(self,in_channels,out_channels,type_block='inception'):
        super().__init__()
        self.type_block = type_block
        self.change_channels = nn.Conv2d(in_channels,out_channels,1,bias=False)
        if type_block == 'inception':
            first_block= Inception_Block(in_channels,in_channels)
        else:
            first_block= Residual_Block(in_channels,in_channels)

        self.first_block=first_block
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.hidden = nn.Sequential(
            nn.Linear(in_channels,in_channels//4),
            nn.ReLU(),
            nn.Linear(in_channels//4,in_channels),
            nn.Sigmoid()
        )
    def forward(self,x):
        first_x = self.first_block(x)
        pool_x = self.global_pool(first_x).permute(0,2,3,1)
        hidden = self.hidden(pool_x)
        out = (first_x.permute(0,2,3,1)*hidden).permute(0,3,1,2)
        if self.type_block != 'inception':
            out=out+x
        return self.change_channels(out)

class Down_Block(nn.Module):
    def __init__(self,in_channels,out_channels,type_block='inception'):
        super().__init__()
        if type_block == 'inception':
            first = Inception_Block(in_channels,out_channels)
        else:
            first = Residual_Block(in_channels,out_channels)
        self.first = first
        self.down = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,2,2,bias=False),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self,x):
        x=self.first(x)
        return  self.down(x)
    
class Up_Block(nn.Module):
    def __init__(self,in_channels,out_channels,last=False,type_block='depwsie'):
        super().__init__()
        self.change_channels =  nn.Conv2d(in_channels,out_channels,1,bias=False)
        self.type_block=type_block
        if type_block == 'depwsie':
            out_lst = [Depwise_Block(out_channels,out_channels)]
        elif type_block == 'convTranspose':
            self.up = nn.Sequential(nn.ConvTranspose2d(in_channels,out_channels,2,2),nn.ReLU(),nn.BatchNorm2d(out_channels))
            out_lst =  [SE_Block(out_channels,out_channels,'residual')]
        elif type_block == 'residual':
            out_lst =  [Residual_Block(out_channels,out_channels,3,last=last)]
        else :
            out_lst =  [Inception_Block(out_channels,out_channels)]

        if (last == True) & (type_block != 'residual'):
            out_lst.append(nn.Sequential(nn.Conv2d(out_channels,out_channels,3,bias=False,padding='same')))
        self.out = nn.Sequential(
            *out_lst
        )
    def forward(self,x,x1):
        if self.type_block == 'convTranspose':
             scale_x = self.up(x)
        else:
            scale_x = self.change_channels(F.interpolate(x,scale_factor=2,mode='nearest'))
        return self.out(scale_x+x1)

class Midscope_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding='same', dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        return self.out(x)

class Widescope_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding='same', dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding='same', dilation=3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        return self.out(x)

class Seperate_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
    def forward(self, x):
        return self.out(x)

class Duck_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn_0 = nn.BatchNorm2d(in_channels)
        self.widescope = Widescope_Block(in_channels, out_channels, 15)
        self.midscope = Midscope_Block(in_channels, out_channels, 7)
        self.res_1 = Residual_Block(in_channels, out_channels, 5)
        self.res_2 = nn.Sequential(
            Residual_Block(in_channels, out_channels, 9),
            Residual_Block(out_channels, out_channels, 9),
        )
        self.res_3 = nn.Sequential(
            Residual_Block(in_channels, out_channels, 13),
            Residual_Block(out_channels, out_channels, 13),
            Residual_Block(out_channels, out_channels, 13)
        )
        self.sep = Seperate_Block(in_channels, out_channels, 6)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn_0(x)
        widescope = self.widescope(x)
        midscope = self.midscope(x)
        res_1 = self.res_1(x)
        res_2 = self.res_2(x)
        res_3 = self.res_3(x)
        sep = self.sep(x)
        return self.bn(widescope + midscope + res_1 + res_2 + res_3 + sep)

class Down_Block_Duck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, 2),
            nn.ReLU(),
        )
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, 2),
            nn.ReLU(),
        )
        self.merge = nn.Conv2d(2 * out_channels, out_channels, 3, padding='same')
        self.duck = Duck_Block(out_channels, out_channels)

    def forward(self, x, duck_x):
        down_x = self.down_0(x)
        down_duck_x = self.down_1(duck_x)
        merge = self.merge(torch.cat((down_x, down_duck_x), 1))
        duck_out = self.duck(merge)
        return down_x, duck_out

class Bneck_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, 2),
            nn.ReLU(),
        )
        self.down_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, 2),
            nn.ReLU(),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels)
        )
        self.res = Residual_Block(out_channels, out_channels, 3)

    def forward(self, x, duck_x):
        down_x = self.down_0(x)
        down_duck_x = self.down_1(duck_x)
        merge = self.merge(torch.cat((down_x, down_duck_x), 1))
        res = self.res(merge)
        return res

class Up_Block_Duck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.merge = nn.Conv2d(2 * in_channels, in_channels, 3, padding='same')
        self.duck = Duck_Block(in_channels, out_channels)

    def forward(self, x, x1):
        scale_x = F.interpolate(x, scale_factor=2, mode='nearest')
        merge = self.merge(torch.cat((scale_x, x1), 1))
        duck_out = self.duck(merge)
        return duck_out

