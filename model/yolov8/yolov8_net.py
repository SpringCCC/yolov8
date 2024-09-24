import numpy as np
import torch
import torch.nn as nn
import pkg_resources as pkg
from .backbone import Backbone, C2f, Conv
import logging


def dist2bbox(distance, anchor_points,xywh=True,dim=1):
   
    lt,rb = distance.chunk(2,dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) /2
        wh = x2y2 - x1y1
        return torch.cat((c_xy,wh),dim)
    return torch.cat((x1y1,x2y2),dim)

def check_version(current: str = "0.0.0",
                  minimum: str = "0.0.0",
                  name: str = "version ",
                  pinned: bool = False) -> bool:
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    return result
TORCH_1_10 = check_version(torch.__version__, '1.10.0')


def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w  = feats[i].shape
        sx          = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy          = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx      = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def fuse_conv_and_bn(conv, bn):
    # 混合Conv2d + BatchNorm2d 减少计算量
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)
    # 准备kernel
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    # 准备bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fusedconv

class DFL(nn.Module):
    # DFL模块
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv   = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.c1     = c1
        self.reset_dfl()

    def reset_dfl(self):
        x = torch.arange(self.c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, self.c1, 1, 1))


    def forward(self, x):
        # bs, self.reg_max * 4, 8400
        b, c, a = x.shape
        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloV8(nn.Module):
    def __init__(self, num_classes=20, phi='n', pretrained=False):
        super(YoloV8, self).__init__()
        depth_dict          = {'n' : 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.00,}
        width_dict          = {'n' : 0.25, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        deep_width_dict     = {'n' : 1.00, 's' : 1.00, 'm' : 0.75, 'l' : 0.50, 'x' : 0.50,}
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是3, 640, 640
        #-----------------------------------------------#

        #---------------------------------------------------#   
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   256, 80, 80
        #   512, 40, 40
        #   1024 * deep_mul, 20, 20
        #---------------------------------------------------#
        self.backbone   = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)

        #------------------------加强特征提取网络------------------------# 
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        self.conv3_for_upsample1    = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        # 768, 80, 80 => 256, 80, 80
        self.conv3_for_upsample2    = C2f(base_channels * 8 + base_channels * 4, base_channels * 4, base_depth, shortcut=False)
        
        # 256, 80, 80 => 256, 40, 40
        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        # 512 + 256, 40, 40 => 512, 40, 40
        self.conv3_for_downsample1  = C2f(base_channels * 8 + base_channels * 4, base_channels * 8, base_depth, shortcut=False)

        # 512, 40, 40 => 512, 20, 20
        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        # 1024 * deep_mul + 512, 20, 20 =>  1024 * deep_mul, 20, 20
        self.conv3_for_downsample2  = C2f(int(base_channels * 16 * deep_mul) + base_channels * 8, int(base_channels * 16 * deep_mul), base_depth, shortcut=False)
        #------------------------加强特征提取网络------------------------# 
        
        ch              = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
        self.shape      = None
        self.nl         = len(ch)
        # self.stride     = torch.zeros(self.nl)
        self.stride     = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
        self.reg_max    = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no         = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.num_classes = num_classes
        self.head = Head(self.num_classes, ch)
        if not pretrained:
            weights_init(self)
        self.head.dfl.reset_dfl()
        
        

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self
    
    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)
        
        #------------------------加强特征提取网络------------------------# 
        # 1024 * deep_mul, 20, 20 => 1024 * deep_mul, 40, 40
        P5_upsample = self.upsample(feat3)
        # 1024 * deep_mul, 40, 40 cat 512, 40, 40 => 1024 * deep_mul + 512, 40, 40
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        P4          = self.conv3_for_upsample1(P4)

        # 512, 40, 40 => 512, 80, 80
        P4_upsample = self.upsample(P4)
        # 512, 80, 80 cat 256, 80, 80 => 768, 80, 80
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 768, 80, 80 => 256, 80, 80
        P3          = self.conv3_for_upsample2(P3)

        # 256, 80, 80 => 256, 40, 40
        P3_downsample = self.down_sample1(P3)
        # 512, 40, 40 cat 256, 40, 40 => 768, 40, 40
        P4 = torch.cat([P3_downsample, P4], 1)
        # 768, 40, 40 => 512, 40, 40
        P4 = self.conv3_for_downsample1(P4)

        # 512, 40, 40 => 512, 20, 20
        P4_downsample = self.down_sample2(P4)
        # 512, 20, 20 cat 1024 * deep_mul, 20, 20 => 1024 * deep_mul + 512, 20, 20
        P5 = torch.cat([P4_downsample, feat3], 1)
        # 1024 * deep_mul + 512, 20, 20 => 1024 * deep_mul, 20, 20
        P5 = self.conv3_for_downsample2(P5)
        #------------------------加强特征提取网络------------------------# 
        # P3 256, 80, 80
        # P4 512, 40, 40
        # P5 1024 * deep_mul, 20, 20
        # P3 256, 80, 80 => num_classes + self.reg_max * 4, 80, 80
        # P4 512, 40, 40 => num_classes + self.reg_max * 4, 40, 40
        # P5 1024 * deep_mul, 20, 20 => num_classes + self.reg_max * 4, 20, 20
        x = self.head([P3, P4, P5])
        return x
        
    


class Head(nn.Module):
    
    def __init__(self,nc=20,ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = 16
        self.no = nc + self.reg_max * 4  # 类别信息 + 框信息
        self.stride = torch.Tensor([8,16,32])
        self.reg_max    = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        c2, c3   = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], nc)  # channels
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max)
        self.shape = None


    def decode_x(self, x):
        shape = x[0].shape
        self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
        
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=False, dim=1) * self.strides # 真实图像尺寸（相当于imgsz的）
        y = torch.cat((dbox, cls.sigmoid()), 1) # b, 84, 8400
        return y
        
    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        y = self.decode_x(x)
        return y, x
