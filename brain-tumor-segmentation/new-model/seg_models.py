import torch
import torch.nn as nn
from modules import Down_Block,SE_Block,Up_Block,Down_Block_Duck,Up_Block_Duck,Residual_Block,Bneck_Block,Duck_Block

class ClothSegmentation(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.change_channels = nn.Sequential(
            nn.Conv2d(in_channels,4,3,padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(4)
        )
        self.down_1 = Down_Block(4,8,'residual') #128
        self.down_2 = Down_Block(8,16,'residual') #64
        self.down_3 = Down_Block(16,32) #32
        self.down_4 = Down_Block(32,64) #16
        self.bneck = nn.Sequential(
            Down_Block(64,128,'residual'),
            SE_Block(128,256,'inception'),
            SE_Block(256,128,'residual'),
        ) #8
        self.up_1 = Up_Block(128,64) #16
        self.up_2 = Up_Block(64,32) #32
        self.up_3 = Up_Block(32,16) #64
        self.up_4 = Up_Block(16,8) #128

        self.out = Up_Block(8,4,True,'residual')
        self.fc = nn.Sequential(
            nn.Linear(4, 20),
            nn.ReLU(),
            nn.Linear(20, 2, bias=False)
            )
    def forward(self,x):
        change_channels = self.change_channels (x)
        down_1 = self.down_1(change_channels)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)

        bneck =  self.bneck(down_4)
        up_1 = self.up_1(bneck,down_4)
        up_2 = self.up_2(up_1,down_3)
        up_3 = self.up_3(up_2,down_2)
        up_4 = self.up_4(up_3,down_1)
        out=self.out(up_4,change_channels)
        return self.fc(out.permute(0,2,3,1)).permute(0,3,1,2)


class AgnosticModel(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down_1 = Down_Block(in_channels,8) #128
        self.down_2 = Down_Block(8,16) #64
        self.down_3 = Down_Block(16,32) #32
        self.down_4 = Down_Block(32,64) #16
        self.bneck = nn.Sequential(
            Down_Block(64,128),
            SE_Block(128,256,'residual'),
            SE_Block(256,128,'residual'),
        ) #8
        self.up_1 = Up_Block(128,64,type_block='residual') #16
        self.up_2 = Up_Block(64,32,type_block='convTranspose') #32
        self.up_3 = Up_Block(32,16,type_block='residual') #64
        self.up_4 = Up_Block(16,8,type_block='convTranspose') #128

        self.out = Up_Block(8,8)
        self.fc = nn.Sequential(
            nn.Linear(8,16),
            nn.Mish(),
            nn.Linear(16,3),
        )
    def forward(self,x):
        down_1 = self.down_1(x)
        down_2 = self.down_2(down_1)
        down_3 = self.down_3(down_2)
        down_4 = self.down_4(down_3)

        bneck =  self.bneck(down_4)
        up_1 = self.up_1(bneck,down_4)
        up_2 = self.up_2(up_1,down_3)
        up_3 = self.up_3(up_2,down_2)
        up_4 = self.up_4(up_3,down_1)
        out = self.out(up_4,x)
        return self.fc(out.permute(0,2,3,1)).permute(0,3,1,2)
class Duck_Net(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self._change_channels = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding='same'),
            nn.Mish()
        )

        self.duck_0 = Duck_Block(1, 1)
        self.down_0 = Down_Block_Duck(1, 2)
        self.down_1 = Down_Block_Duck(2, 4)
        self.down_2 = Down_Block_Duck(4, 8)
        self.bneck = Bneck_Block(8, 16)
        self.res = Residual_Block(16, 8, 3)
        self.up_1 = Up_Block_Duck(8, 4)
        self.up_2 = Up_Block_Duck(4, 2)
        self.up_3 = Up_Block_Duck(2, 1)
        self.final_up = Up_Block_Duck(1, 4)
        self.fc = nn.Linear(4, 2, bias=False)

    def forward(self, x):

        x = self._change_channels(x)
        duck_first = self.duck_0(x)
        down_0, duck_0 = self.down_0(x, duck_first)
        down_1, duck_1 = self.down_1(down_0, duck_0)
        down_2, duck_2 = self.down_2(down_1, duck_1)
        bneck = self.bneck(down_2, duck_2)
        bneck = self.res(bneck)
        up_1 = self.up_1(bneck, duck_2)
        up_2 = self.up_2(up_1, duck_1)
        up_3 = self.up_3(up_2, duck_0)
        fn = self.final_up(up_3, duck_first)
        fn = fn.permute(0, 2, 3, 1)
        return self.fc(fn).permute(0, 3, 1, 2)