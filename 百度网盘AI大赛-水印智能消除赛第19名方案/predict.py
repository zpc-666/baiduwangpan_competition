# 代码示例
# python predict.py [src_image_dir] [results]

import os
import sys
import glob
import json
import cv2

# https://aistudio.baidu.com/aistudio/projectdetail/3340332?channelType=0&channel=0
"""
paddlepaddle-gpu==2.2.1
time:2021.07.16 9:00
author:CP
backbone：U-net
"""
import paddle
from paddle import nn
import paddle.nn.functional as F

class CALayer(nn.Layer):
    def __init__(self, channels, reduction=16):
        super(CALayer, self).__init__()

        mid_c = max(channels//reduction, 16)
        self.conv1 = nn.Sequential(
            nn.Conv2D(channels, mid_c, 1),
            nn.ReLU(),
            nn.Conv2D(mid_c, channels, 1),
            nn.Sigmoid(),
            )

    def forward(self, x):
        y = x.mean(axis=(-1, -2), keepdim=True)
        y = self.conv1(y)
        return y

class Encoder(nn.Layer):#下采样：两层卷积，两层归一化，最后池化。
    def __init__(self, num_channels, num_filters):
        super(Encoder,self).__init__()#继承父类的初始化
        self.conv1 = nn.Conv2D(in_channels=num_channels,
                              out_channels=num_filters,
                              kernel_size=3,#3x3卷积核，步长为1，填充为1，不改变图片尺寸[H W]
                              stride=1,
                              padding=1,
                              bias_attr=False)
        self.bn1   = nn.BatchNorm(num_filters)#归一化，并使用了激活函数
        
        self.conv2 = nn.Conv2D(in_channels=num_filters,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias_attr=False)
        self.bn2   = nn.BatchNorm(num_filters)
        
        self.pool  = nn.MaxPool2D(kernel_size=2,stride=2,padding="SAME")#池化层，图片尺寸减半[H/2 W/2]

        if num_channels!=num_filters:
            self.downsample = nn.Sequential(
                nn.Conv2D(num_channels, num_filters, 1, bias_attr=False),
                nn.BatchNorm2D(num_filters)
            )
        else:
            self.downsample = lambda x: x
        
    def forward(self,inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x+self.downsample(inputs), 0.2)
        x_conv = x           #两个输出，灰色 ->
        x_pool = self.pool(x)#两个输出，红色 | 
        return x_conv, x_pool
    
    
class Decoder(nn.Layer):#上采样：一层反卷积，两层卷积层，两层归一化
    def __init__(self, num_channels, num_filters):
        super(Decoder,self).__init__()
        self.up = nn.Conv2DTranspose(in_channels=num_channels,
                                    out_channels=num_filters,
                                    kernel_size=2,
                                    stride=2,
                                    padding=0,
                                    bias_attr=False)#图片尺寸变大一倍[2*H 2*W]
        self.up_bn   = nn.BatchNorm(num_filters)

        self.conv1 = nn.Conv2D(in_channels=num_filters*2,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias_attr=False)
        self.bn1   = nn.BatchNorm(num_filters)
        
        self.conv2 = nn.Conv2D(in_channels=num_filters,
                              out_channels=num_filters,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias_attr=False)
        self.bn2   = nn.BatchNorm(num_filters)

        if num_channels!=num_filters:
            self.upsample = nn.Sequential(
                nn.Conv2D(num_filters*2, num_filters, 1, bias_attr=False),
                nn.BatchNorm2D(num_filters)
            )
        else:
            self.downsample = lambda x: x
        
    def forward(self,input_conv,input_pool):
        x = self.up_bn(self.up(input_pool))
        x = F.leaky_relu(x, 0.2)
        h_diff = (input_conv.shape[2]-x.shape[2])
        w_diff = (input_conv.shape[3]-x.shape[3])
        pad = nn.Pad2D(padding=[h_diff//2, h_diff-h_diff//2, w_diff//2, w_diff-w_diff//2])
        x = pad(x)                                #以下采样保存的feature map为基准，填充上采样的feature map尺寸
        x = paddle.concat(x=[input_conv,x],axis=1)#考虑上下文信息，in_channels扩大两倍
        x_sc = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x+x_sc, 0.2)
        return x
    
class UNet(nn.Layer):
    def __init__(self,num_classes=3):
        super(UNet,self).__init__()
        self.down1 = Encoder(num_channels=  3, num_filters=64) #下采样
        self.down2 = Encoder(num_channels= 64, num_filters=128)
        self.down3 = Encoder(num_channels=128, num_filters=256)
        self.down4 = Encoder(num_channels=256, num_filters=512)
        
        self.mid_conv1 = nn.Sequential(
            nn.Conv2D(512,1024,1, bias_attr=False),
            nn.BatchNorm(1024),
            nn.LeakyReLU(0.2)
        )

        self.mid_conv2 = nn.Sequential(
            nn.Conv2D(512,1024,3, padding=1, bias_attr=False),
            nn.BatchNorm(1024),
            nn.LeakyReLU(0.2)
        )

        self.ca_layer1 = CALayer(1024, 32)

        self.mid_conv3 = nn.Sequential(
            nn.Conv2D(1024,1024,1, bias_attr=False),
            nn.BatchNorm(1024),
            nn.LeakyReLU(0.2)
        )

        self.up4 = Decoder(1024,512)                           #上采样
        self.up3 = Decoder(512,256)
        self.up2 = Decoder(256,128)
        self.up1 = Decoder(128,64)
        
        self.last_conv1 = nn.Conv2D(64,num_classes,1)           #1x1卷积，softmax做分类
        self.last_conv2 = nn.Conv2D(64,num_classes,3, padding=1)

        self.ca_layer2 = CALayer(num_classes)
        
    def forward(self,inputs):
        x1, x = self.down1(inputs)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        
        x_m1 = self.mid_conv1(x)
        x_m2 = self.mid_conv2(x)
        attn = self.ca_layer1(x_m1+x_m2)
        x = x_m1*attn+x_m2*(1.-attn)
        x = self.mid_conv3(x)
        
        x = self.up4(x4, x)
        x = self.up3(x3, x)
        x = self.up2(x2, x)
        x = self.up1(x1, x)
        
        out1 = self.last_conv1(x)
        out2 = self.last_conv2(x)
        attn = self.ca_layer2(out1+out2)
        x = out1*attn+out2*(1.-attn)
        
        return inputs-x

# 查看网络各个节点的输出信息
#paddle.summary(UNet(), (1, 3, 600, 600))

def process(src_image_dir, save_dir):
    model = UNet()
    param_dict = paddle.load('./model_best.pdparams')
    model.load_dict(param_dict)

    model.eval()

    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    for image_path in image_paths:
        # do something
        img = cv2.imread(image_path)
        h, w, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = paddle.vision.transforms.resize(img, (512,512), interpolation='bilinear')
        img = img.transpose((2,0,1))
        img = img/255

        img = paddle.to_tensor(img).astype('float32')
        img = img.reshape([1]+img.shape)
        pre = model(img)[0].numpy()

        pre[pre>0.9]=1
        pre[pre<0.1]=0

        pre = pre*255
        pre = pre.transpose((1,2,0))
        pre = paddle.vision.transforms.resize(pre, (h,w), interpolation='bilinear')
        out_image = cv2.cvtColor(pre, cv2.COLOR_RGB2BGR)

        # 保存结果图片
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, out_image)
        

if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    process(src_image_dir, save_dir)


