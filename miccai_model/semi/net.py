import torch
from torch import nn
import torch.nn.functional as F
import math
class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.LeakyReLU(inplace=True))

            # ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.leaky_relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.LeakyReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x



class Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=32, normalization='none', has_dropout=False, deep_supervision=False,max_n_filters=320):#,patch_size = [96,160,160],max_pool=6):
        super(Net, self).__init__()
        self.deep_supervision = deep_supervision
        self.has_dropout = has_dropout

        self.block_en1 = ConvBlock(2, n_channels, n_filters, normalization=normalization)
        self.block_en1_dw = DownsamplingConvBlock(n_filters, n_filters* 2, stride=(1,2,2),normalization=normalization)

        self.block_en2 = ConvBlock(2, n_filters* 2, n_filters * 2, normalization=normalization)
        self.block_en2_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_en3 = ConvBlock(2, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_en3_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)
        
        self.block_en4 = ConvBlock(2, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_en4_dw = DownsamplingConvBlock(n_filters * 8, max_n_filters, normalization=normalization)

        self.block_en5 = ConvBlock(2, max_n_filters, max_n_filters, normalization=normalization)
        self.block_en5_dw = DownsamplingConvBlock(max_n_filters, max_n_filters, stride=(1,2,2),normalization=normalization)

        self.block_b6 = ConvBlock(2,max_n_filters, max_n_filters, normalization=normalization)
        self.block_de6_up = UpsamplingDeconvBlock(max_n_filters, max_n_filters,stride=(1,2,2), normalization=normalization)

        self.block_de5= ConvBlock(2, 2*max_n_filters, max_n_filters, normalization=normalization)
        self.block_de5_up = UpsamplingDeconvBlock(max_n_filters, n_filters * 8, normalization=normalization)

        self.block_de4 = ConvBlock(2, 2*n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_de4_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_de3 = ConvBlock(2, 2*n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_de3_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_de2 = ConvBlock(2, 2*n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_de2_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, stride=(1,2,2),normalization=normalization)

        self.block_de1 = ConvBlock(2, 2*n_filters, n_filters, normalization=normalization)

        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv1 = nn.Conv3d(n_filters * 2, n_classes, 1, padding=0)
        self.out_conv2 = nn.Conv3d(n_filters * 4, n_classes, 1, padding=0)
        self.out_conv3 = nn.Conv3d(n_filters * 8, n_classes, 1, padding=0)
        # we don't do supervision on the lowest 2 outputs.

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()
        self.__init_weight()

    def encoder(self, input):
        x1 = self.block_en1(input)
        x1_dw = self.block_en1_dw(x1)

        x2 = self.block_en2(x1_dw)
        x2_dw = self.block_en2_dw(x2)

        x3 = self.block_en3(x2_dw)
        x3_dw = self.block_en3_dw(x3)

        x4 = self.block_en4(x3_dw)
        x4_dw = self.block_en4_dw(x4)

        x5 = self.block_en5(x4_dw)      
        x5_dw = self.block_en5_dw(x5)

        x6= self.block_b6(x5_dw)

        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x6 = self.dropout(x6)

        res = [x1, x2, x3, x4, x5,x6]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]
        x6= features[5]

        B,C,Z,X,Y = x1.size()
        x6_up = self.block_de6_up(x6)
        x6_up = torch.cat((x6_up, x5), 1)#x6_up + x5
        
        x5d = self.block_de5(x6_up)
        x5_up = self.block_de5_up(x5d)
        x5_up = torch.cat((x5_up, x4), 1)#x5_up + x4
        
        x4d = self.block_de4(x5_up)
        x4_up = self.block_de4_up(x4d)
        x4_up = torch.cat((x4_up, x3), 1)#x4_up + x3
        
        x3d = self.block_de3(x4_up)
        x3_up = self.block_de3_up(x3d)
        x3_up = torch.cat((x3_up, x2), 1)#x3_up + x2
                
                
        x2d = self.block_de2(x3_up)
        x2_up = self.block_de2_up(x2d)
        x2_up = torch.cat((x2_up, x1), 1)#x2_up + x1

        x1d = self.block_de1(x2_up)

        if self.has_dropout:
            x1d = self.dropout(x1d)
        if(self.deep_supervision == True):
            out0 = self.out_conv(x1d)
            out1 =F.interpolate(self.out_conv1(x2d), scale_factor=(1,2,2), mode='trilinear')
            out2 = F.interpolate(self.out_conv2(x3d), scale_factor=(2,4,4), mode='trilinear')#nn.functional.interpolate(, scale_factor=4, mode='trilinear')
            out3 = F.interpolate(self.out_conv3(x4d), scale_factor=(4,8,8), mode='trilinear')
            out = [out0, out1, out2, out3]
        else:
            out = self.out_conv(x1d)
        return out


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                torch.nn.init.kaiming_normal_(m.weight,a=1e-2)

if __name__ == '__main__':

    model = Net(n_channels=1, n_classes=7)
