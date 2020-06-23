import torch.nn as nn
import torch
from torchvision.ops.boxes import nms as nms_torch

from efficientnet import EfficientNet as EffNet
from efficientnet.utils import MemoryEfficientSwish, Swish
from efficientnet.utils import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding
from efficientnet.utils import calculate_output_image_size
import struct
import ctypes

def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117, Lichao
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False, image_size=None):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False, image_size=image_size)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1, image_size=image_size)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPN(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True, image_size=None):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        p3_image_size = calculate_output_image_size(image_size, 1)
        p4_image_size = calculate_output_image_size(image_size, 2)
        p5_image_size = calculate_output_image_size(image_size, 4)
        p6_image_size = calculate_output_image_size(image_size, 8)
        p7_image_size = calculate_output_image_size(image_size, 16)

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p6_image_size)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p5_image_size)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p4_image_size)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p3_image_size)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p4_image_size)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p5_image_size)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p6_image_size)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p7_image_size)

        # Feature scaling layers
        # self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Feature scaling layers, for onnx
        self.p6_upsample = nn.Upsample(size=(p6_image_size[0], p6_image_size[1]))
        self.p5_upsample = nn.Upsample(size=(p5_image_size[0], p5_image_size[1]))
        self.p4_upsample = nn.Upsample(size=(p4_image_size[0], p4_image_size[1]))
        self.p3_upsample = nn.Upsample(size=(p3_image_size[0], p3_image_size[1]))

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2, image_size=p3_image_size)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2, image_size=p4_image_size)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2, image_size=p5_image_size)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2, image_size=p6_image_size)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1, image_size=p5_image_size),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1, image_size=p4_image_size),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1, image_size=p3_image_size),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1, image_size=p5_image_size),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2, image_size=p5_image_size)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2, image_size=p6_image_size)
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1, image_size=p4_image_size),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1, image_size=p5_image_size),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class BiFPN_infer(nn.Module):
    """
    modified by Zylo117, Lichao
    """

    def __init__(self, num_channels, conv_channels, first_time=False, index=0, epsilon=1e-4, onnx_export=False, attention=True, image_size=None):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN_infer, self).__init__()
        self.epsilon = epsilon
        p3_image_size = calculate_output_image_size(image_size, 1)
        p4_image_size = calculate_output_image_size(image_size, 2)
        p5_image_size = calculate_output_image_size(image_size, 4)
        p6_image_size = calculate_output_image_size(image_size, 8)
        p7_image_size = calculate_output_image_size(image_size, 16)

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p6_image_size)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p5_image_size)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p4_image_size)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p3_image_size)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p4_image_size)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p5_image_size)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p6_image_size)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export, image_size=p7_image_size)

        # Feature scaling layers, for onnx
        self.p6_upsample = nn.Upsample(size=(p6_image_size[0], p6_image_size[1]))
        self.p5_upsample = nn.Upsample(size=(p5_image_size[0], p5_image_size[1]))
        self.p4_upsample = nn.Upsample(size=(p4_image_size[0], p4_image_size[1]))
        self.p3_upsample = nn.Upsample(size=(p3_image_size[0], p3_image_size[1]))

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2, image_size=p3_image_size)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2, image_size=p4_image_size)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2, image_size=p5_image_size)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2, image_size=p6_image_size)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1, image_size=p5_image_size),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1, image_size=p4_image_size),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1, image_size=p3_image_size),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1, image_size=p5_image_size),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2, image_size=p5_image_size)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2, image_size=p6_image_size)
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1, image_size=p4_image_size),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1, image_size=p5_image_size),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        # hard code
        self.weight_p6_1 = []
        self.weight_p5_1 = []
        self.weight_p4_1 = []
        self.weight_p3_1 = []
        self.weight_p6_2 = []
        self.weight_p5_2 = []
        self.weight_p4_2 = []
        self.weight_p3_2 = []
        self.weight_p6_1.clear()
        self.weight_p5_1.clear()
        self.weight_p4_1.clear()
        self.weight_p3_1.clear()
        self.weight_p3_2.clear()
        self.weight_p4_2.clear()
        self.weight_p5_2.clear()
        self.weight_p6_2.clear()


        self.weight_p6_1.append([self.hex2float("0x3eb37115"), self.hex2float("0x3f26449d")])
        self.weight_p5_1.append([self.hex2float("0x3f35db7c"), self.hex2float("0x3e944361")])
        self.weight_p4_1.append([self.hex2float("0x3ef7b104"), self.hex2float("0x3f042548")])
        self.weight_p3_1.append([self.hex2float("0x3f01447d"), self.hex2float("0x3efd72c4")])
        self.weight_p3_2.append([self.hex2float("0x3eeee6a9"), self.hex2float("0x3f088a97"), self.hex2float("0x0")])
        self.weight_p4_2.append([self.hex2float("0x3f68799b"), self.hex2float("0x3dbc1ac5"), self.hex2float("0x0")])
        self.weight_p5_2.append([self.hex2float("0x3f2d33bc"), self.hex2float("0x3e830c60"), self.hex2float("0x3d8a2074")])
        self.weight_p6_2.append([self.hex2float("0x3f14da23"), self.hex2float("0x3ed64757")])
        
        self.weight_p6_1.append([self.hex2float("0x3ecb13c4"), self.hex2float("0x3f1a72e7")])
        self.weight_p5_1.append([self.hex2float("0x3ef1e158"), self.hex2float("0x3f070c81")])
        self.weight_p4_1.append([self.hex2float("0x3f1d2519"), self.hex2float("0x3ec5b0b8")])
        self.weight_p3_1.append([self.hex2float("0x3f3cc1ce"), self.hex2float("0x3e867798")])
        self.weight_p3_2.append([self.hex2float("0x3e89c622"), self.hex2float("0x3f0374f0"), self.hex2float("0x3e5e984e")])
        self.weight_p4_2.append([self.hex2float("0x3f18653f"), self.hex2float("0x3ea934b8"), self.hex2float("0x3d97f1c3")])
        self.weight_p5_2.append([self.hex2float("0x3eb48c13"), self.hex2float("0x3ce0c706"), self.hex2float("0x3f1eb138")])
        self.weight_p6_2.append([self.hex2float("0x3f1e94c7"), self.hex2float("0x3ec2d0f0")])
        
        self.weight_p6_1.append([self.hex2float("0x3f257965"), self.hex2float("0x3eb506a4")])
        self.weight_p5_1.append([self.hex2float("0x3efb61a4"), self.hex2float("0x3f024c0e")])
        self.weight_p4_1.append([self.hex2float("0x3f1dea5d"), self.hex2float("0x3ec425bc")])
        self.weight_p3_1.append([self.hex2float("0x3f4d9d5e"), self.hex2float("0x3e497faa")])
        self.weight_p3_2.append([self.hex2float("0x3e8a4141"), self.hex2float("0x3f1ae133"), self.hex2float("0x3dffe0d3")])
        self.weight_p4_2.append([self.hex2float("0x3ec3e7c6"), self.hex2float("0x3e5d1e5c"), self.hex2float("0x3ecd8542")])
        self.weight_p5_2.append([self.hex2float("0x3dc8c0d2"), self.hex2float("0x3ea09642"), self.hex2float("0x3f169a64")])
        self.weight_p6_2.append([self.hex2float("0x3f13df92"), self.hex2float("0x3ed83b12")])
        
        self.weight_p6_1.append([self.hex2float("0x3f222c3e"), self.hex2float("0x3ebba1d2")])
        self.weight_p5_1.append([self.hex2float("0x3effbf80"), self.hex2float("0x3f001d6b")])
        self.weight_p4_1.append([self.hex2float("0x3f0b99a2"), self.hex2float("0x3ee8c763")])
        self.weight_p3_1.append([self.hex2float("0x3f39cbce"), self.hex2float("0x3e8c63a4")])
        self.weight_p3_2.append([self.hex2float("0x3e86f76f"), self.hex2float("0x3f055105"), self.hex2float("0x3e5cc5a8")])
        self.weight_p4_2.append([self.hex2float("0x3e914b9d"), self.hex2float("0x3e5efc1a"), self.hex2float("0x3eff3271")])
        self.weight_p5_2.append([self.hex2float("0x3d77ebf7"), self.hex2float("0x3eb7a769"), self.hex2float("0x3f14ab39")])
        self.weight_p6_2.append([self.hex2float("0x3eebaf36"), self.hex2float("0x3f0a2567")])
        
        self.weight_p6_1.append([self.hex2float("0x3f0539b6"), self.hex2float("0x3ef58761")])
        self.weight_p5_1.append([self.hex2float("0x3ef02351"), self.hex2float("0x3f07ebb2")])
        self.weight_p4_1.append([self.hex2float("0x3f1215f7"), self.hex2float("0x3edbcf17")])
        self.weight_p3_1.append([self.hex2float("0x3f1457ea"), self.hex2float("0x3ed74c5e")])
        self.weight_p3_2.append([self.hex2float("0x3e256b25"), self.hex2float("0x3ee3f088"), self.hex2float("0x3ec9564b")])
        self.weight_p4_2.append([self.hex2float("0x3e4438cf"), self.hex2float("0x3ec84ce1"), self.hex2float("0x3ed59320")])
        self.weight_p5_2.append([self.hex2float("0x3dcd12a0"), self.hex2float("0x3ed2c85e"), self.hex2float("0x3ef9eed7")])
        self.weight_p6_2.append([self.hex2float("0x3e48eee4"), self.hex2float("0x3f4dc0c3")])

        self.weight_p6_1.append([self.hex2float("0x3f04ccb5"), self.hex2float("0x3ef66127")])
        self.weight_p5_1.append([self.hex2float("0x3eed880d"), self.hex2float("0x3f093941")])
        self.weight_p4_1.append([self.hex2float("0x3effb230"), self.hex2float("0x3f002414")])
        self.weight_p3_1.append([self.hex2float("0x3f31c26e"), self.hex2float("0x3e9c7656")])
        self.weight_p3_2.append([self.hex2float("0x3e72bab6"), self.hex2float("0x3edf6449"), self.hex2float("0x3ea73aae")])
        self.weight_p4_2.append([self.hex2float("0x3e940808"), self.hex2float("0x3e9dbad6"), self.hex2float("0x3ece3952")])
        self.weight_p5_2.append([self.hex2float("0x3b9f33c6"), self.hex2float("0x3ec6a2f2"), self.hex2float("0x3f1b6db1")])
        self.weight_p6_2.append([self.hex2float("0x3ec3ff9b"), self.hex2float("0x3f1dfd70")])

        self.weight_p6_1.append([self.hex2float("0x3ecaf1b7"), self.hex2float("0x3f1a840d")])
        self.weight_p5_1.append([self.hex2float("0x3ecbb94f"), self.hex2float("0x3f1a208d")])
        self.weight_p4_1.append([self.hex2float("0x3ef6e61f"), self.hex2float("0x3f048a8c")])
        self.weight_p3_1.append([self.hex2float("0x3f00d2c1"), self.hex2float("0x3efe5675")])
        self.weight_p3_2.append([self.hex2float("0x3e729312"), self.hex2float("0x3e86843d"), self.hex2float("0x3f00174b")])
        self.weight_p4_2.append([self.hex2float("0x3e5d212b"), self.hex2float("0x3e92145c"), self.hex2float("0x3eff5768")])
        self.weight_p5_2.append([self.hex2float("0x3df85e54"), self.hex2float("0x3e9e4930"), self.hex2float("0x3f11cd8a")])
        self.weight_p6_2.append([self.hex2float("0x3e4986ba"), self.hex2float("0x3f4d9acd")])

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.idx = index
        self.attention = attention


    def float_to_hex(self, f):
        return hex(struct.unpack('<I', struct.pack('<f', f))[0])
    
    
    def hex2float(self, h):
        i = int(h, 16)
        cp = ctypes.pointer(ctypes.c_int(i))
        fp = ctypes.cast(cp, ctypes.POINTER(ctypes.c_float))
        return fp.contents.value
    
    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        # p6_w1 = self.p6_w1_relu(self.p6_w1)
        # weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # export = weight.cpu().detach().numpy()
        # print("self.weight_p6_1.append([", self.float_to_hex(export[0]), ", ", self.float_to_hex(export[1]), " ])")
        # Connections for P6_0 and P7_0 to P6_1 respectively
        # p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))
        p6_up = self.conv6_up(self.swish(self.weight_p6_1[self.idx][0] * p6_in + self.weight_p6_1[self.idx][1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        # p5_w1 = self.p5_w1_relu(self.p5_w1)
        # weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # export = weight.cpu().detach().numpy()
        # print("self.weight_p5_1.append([", self.float_to_hex(export[0]), ", ", self.float_to_hex(export[1]), " ])")
        # Connections for P5_0 and P6_1 to P5_1 respectively
        # p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))
        p5_up = self.conv5_up(self.swish(self.weight_p5_1[self.idx][0] * p5_in + self.weight_p5_1[self.idx][1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_1 to P4_1
        # p4_w1 = self.p4_w1_relu(self.p4_w1)
        # weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # export = weight.cpu().detach().numpy()
        # print("self.weight_p4_1.append([", self.float_to_hex(export[0]), ", ", self.float_to_hex(export[1]), " ])")
        # Connections for P4_0 and P5_1 to P4_1 respectively
        # p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))
        p4_up = self.conv4_up(self.swish(self.weight_p4_1[self.idx][0] * p4_in + self.weight_p4_1[self.idx][1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        # p3_w1 = self.p3_w1_relu(self.p3_w1)
        # weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # export = weight.cpu().detach().numpy()
        # print("self.weight_p3_1.append([", self.float_to_hex(export[0]), ", ", self.float_to_hex(export[1]), " ])")
        # Connections for P3_0 and P4_1 to P3_2 respectively
        # p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))
        p3_out = self.conv3_up(self.swish(self.weight_p3_1[self.idx][0] * p3_in + self.weight_p3_1[self.idx][1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        # p4_w2 = self.p4_w2_relu(self.p4_w2)
        # weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # export = weight.cpu().detach().numpy()
        # print("self.weight_p3_2.append([", self.float_to_hex(export[0]), ", ", self.float_to_hex(export[1]), ", ", self.float_to_hex(export[2]), " ])")
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        # p4_out = self.conv4_down(
        #     self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))
        p4_out = self.conv4_down(
            self.swish(self.weight_p3_2[self.idx][0] * p4_in + self.weight_p3_2[self.idx][1] * p4_up + self.weight_p3_2[self.idx][2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        # p5_w2 = self.p5_w2_relu(self.p5_w2)
        # weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # export = weight.cpu().detach().numpy()
        # print("self.weight_p4_2.append([", self.float_to_hex(export[0]), ", ", self.float_to_hex(export[1]), ", ", self.float_to_hex(export[2]), " ])")
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        # p5_out = self.conv5_down(
        #     self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))
        p5_out = self.conv5_down(
            self.swish(self.weight_p4_2[self.idx][0] * p5_in + self.weight_p4_2[self.idx][1] * p5_up + self.weight_p4_2[self.idx][2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        # p6_w2 = self.p6_w2_relu(self.p6_w2)
        # weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # export = weight.cpu().detach().numpy()
        # print("self.weight_p5_2.append([", self.float_to_hex(export[0]), ", ", self.float_to_hex(export[1]), ", ", self.float_to_hex(export[2]), " ])")
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        # p6_out = self.conv6_down(
        #     self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))
        p6_out = self.conv6_down(
            self.swish(self.weight_p5_2[self.idx][0] * p6_in + self.weight_p5_2[self.idx][1] * p6_up + self.weight_p5_2[self.idx][2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        # p7_w2 = self.p7_w2_relu(self.p7_w2)
        # weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # export = weight.cpu().detach().numpy()
        # print("self.weight_p6_2.append([", self.float_to_hex(export[0]), ", ", self.float_to_hex(export[1]), " ])")
        # Connections for P7_0 and P6_2 to P7_2
        # p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))
        p7_out = self.conv7_down(self.swish(self.weight_p6_2[self.idx][0] * p7_in + self.weight_p6_2[self.idx][1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        print("not attention!!")
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class Regressor(nn.Module):
    """
    modified by Zylo117, Lichao
    """

    def __init__(self, in_channels, num_anchors, num_layers, onnx_export=False, image_size=None, compound_coef=0, batch_size=1):
        super(Regressor, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.compound_coef = compound_coef
        self.index = [0,1,2,3,4]
        self.feat_shape = [
                [36864, 9216, 2304, 576, 144],
                [57600, 14400, 3600, 900, 225],
                [82944, 20736, 5184, 1296, 324],
                [112896, 28224, 7056, 1764, 441],
                [147456, 36864, 9216, 2304, 576],
                [230400, 57600, 14400, 3600, 900],
                [230400, 57600, 14400, 3600, 900],
                [331776, 82944, 20736, 5184, 1296]
                ]

        temp_image_size = image_size
        conv_list = []
        for _ in range(5):
            c_list = nn.ModuleList([SeparableConvBlock(in_channels, in_channels, norm=False, activation=False, image_size=image_size) for i in range(num_layers)])
            image_size = calculate_output_image_size(image_size, 2)
            conv_list.append(c_list)
        self.conv_list = nn.ModuleList(conv_list)

        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])

        image_size = temp_image_size
        h_list = []
        for _ in range(5):
            h_list.append(SeparableConvBlock(in_channels, num_anchors * 4, norm=False, activation=False, image_size=image_size))
            image_size = calculate_output_image_size(image_size, 2)
        self.headers = nn.ModuleList(h_list)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for ids, feat, bn_list, header, convs in zip(self.index, inputs, self.bn_list, self.headers, self.conv_list):
            for _, bn, conv in zip(range(self.num_layers), bn_list, convs):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            
            feat = header(feat)

            feat = feat.permute(0, 2, 3, 1)
            # feat = feat.contiguous().view(self.batch_size, -1, 4)
            # feat = feat.contiguous().view(self.batch_size, self.feat_shape[self.compound_coef][ids], 4)
            feat = feat.reshape(self.batch_size, self.feat_shape[self.compound_coef][ids], 4)

            feats.append(feat)
        feats = torch.cat(feats, dim=1)

        return feats


class Classifier(nn.Module):
    """
    modified by Zylo117, Lichao
    """

    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False, image_size=None, compound_coef=0, batch_size=1):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.batch_size = batch_size
        
        self.compound_coef = compound_coef
        self.index = [0,1,2,3,4]
        self.feat_shape = [
                [36864, 9216, 2304, 576, 144],
                [57600, 14400, 3600, 900, 225],
                [82944, 20736, 5184, 1296, 324],
                [112896, 28224, 7056, 1764, 441],
                [147456, 36864, 9216, 2304, 576],
                [230400, 57600, 14400, 3600, 900],
                [230400, 57600, 14400, 3600, 900],
                [331776, 82944, 20736, 5184, 1296]
                ]
        self.feats_size = [
                [
                    [4,64,64],
                    [4,32,32],
                    [4,16,16],
                    [4,8,8],
                    [4,4,4]
                    ],
                [
                    [4,80,80],
                    [4,40,40],
                    [4,20,20],
                    [4,10,10],
                    [4,5,5]
                    ],
                [
                    [4,96,96],
                    [4,48,48],
                    [4,24,24],
                    [4,12,12],
                    [4,6,6]
                    ],
                [
                    [4,112,112],
                    [4,56,56],
                    [4,28,28],
                    [4,14,14],
                    [4,7,7]
                    ],
                [
                    [4,128,128],
                    [4,64,64],
                    [4,32,32],
                    [4,16,16],
                    [4,8,8]
                    ],
                [
                    [4,160,160],
                    [4,80,80],
                    [4,40,40],
                    [4,20,20],
                    [4,10,10]
                    ],
                [
                    [4,160,160],
                    [4,80,80],
                    [4,40,40],
                    [4,20,20],
                    [4,10,10]
                    ],
                [
                    [4,192,192],
                    [4,96,96],
                    [4,48,48],
                    [4,24,24],
                    [4,12,12]
                    ]

            ]

        temp_image_size = image_size
        conv_list = []
        for _ in range(5):
            c_list = nn.ModuleList([SeparableConvBlock(in_channels, in_channels, norm=False, activation=False, image_size=image_size) for i in range(num_layers)])
            image_size = calculate_output_image_size(image_size, 2)
            conv_list.append(c_list)
        self.conv_list = nn.ModuleList(conv_list)

        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(5)])

        image_size = temp_image_size
        h_list = []

        for _ in range(5):
            h_list.append(SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False, image_size=image_size))
            image_size = calculate_output_image_size(image_size, 2)
        self.headers = nn.ModuleList(h_list)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for ids, feat, bn_list, header, convs in zip(self.index, inputs, self.bn_list, self.headers, self.conv_list):
            for _, bn, conv in zip(range(self.num_layers), bn_list, convs):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            
            feat = header(feat)

            feat = feat.permute(0, 2, 3, 1)

            # feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors, self.num_classes)
            # feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)
            # feat = feat.contiguous().view(self.batch_size, self.feats_size[self.compound_coef][ids][1], self.feats_size[self.compound_coef][ids][2], self.num_anchors, self.num_classes)
            # feat = feat.contiguous().view(self.batch_size, -1, self.num_classes)
            # feat = feat.contiguous().view(self.batch_size, self.feat_shape[self.compound_coef][ids], self.num_classes)
            feat = feat.reshape(self.batch_size, self.feat_shape[self.compound_coef][ids], self.num_classes)
            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()
        return feats


class EfficientNet(nn.Module):
    """
    modified by Zylo117, Lichao
    """

    def __init__(self, compound_coef, load_weights=False):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(f'efficientnet-b{compound_coef}', load_weights=load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == (2, 2):
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[1:]


if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
