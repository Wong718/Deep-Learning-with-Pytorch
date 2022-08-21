# Day3

## 模型的定义

### 模型定义的三种方式

**Sequential：只需要将模型的层按序排列起来即可**

- 直接排列
    
    ```python
    import torch.nn as nn
    net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10), 
            )
    print(net)
    ```
    
- 使用OrderedDict
    
    ```python
    import collections
    import torch.nn as nn
    net2 = nn.Sequential(collections.OrderedDict([
              ('fc1', nn.Linear(784, 256)),
              ('relu1', nn.ReLU()),
              ('fc2', nn.Linear(256, 10))
              ]))
    print(net2)
    ```
    

**Modulelist:`ModuleList`接收一个子模块（或层，需属于`nn.Module`类）的列表作为输入，然后也可以类似`List`
那样进行append和extend操作。**

```python
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # 类似List的append操作
print(net[-1])  # 类似List的索引访问
print(net)
```

注意：`ModuleList`中元素的先后顺序并不代表其在网络中的真实位置顺序，需要经过forward函数指定各个层的先后顺序后才算完成了模型的定义。

```python
class model(nn.Module):
  def __init__(self, ...):
    super().__init__()
    self.modulelist = ...
    ...
    
  def forward(self, x):
    for layer in self.modulelist:
      x = layer(x)
    return x
```

**ModuleDict：`ModuleDict`能够更方便地为神经网络的层添加名称**

```python
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)
```

- ModuleList和ModuleDict在某个完全相同的层需要重复出现多次时，非常方便实现，可以”一行顶多行“；

## 利用模型块快速搭建复杂的网络

**U-Net模型块分析**

**U-Net模型块实现**

这里的基础部件对应上一节分析的四个模型块，根据功能我们将其命名为：DoubleConv, Down, Up, OutConv。下面给出U-Net中模型块的PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
```

### 利用模块组装U-Net

```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
```

## PyTorch修改模型

### 修改ResNet50

```python
import torchvision.models as models
net = models.resnet50()
print(net)
```

假设我们要用这个resnet模型去做一个10分类的问题，就应该修改模型的fc层，将其输出节点数替换为10。另外，我们觉得一层全连接层可能太少了，想再加一层。可以做如下修改：

```python
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 128)),
                          ('relu1', nn.ReLU()), 
                          ('dropout1',nn.Dropout(0.5)),
                          ('fc2', nn.Linear(128, 10)),
                          ('output', nn.Softmax(dim=1))
                          ]))
    
net.fc = classifier
```

这里的操作相当于将模型（net）最后名称为“fc”的层替换成了名称为“classifier”的结构，该结构是我们自己定义的。

### **添加外部输入**

我们希望利用已有的模型结构，在倒数第二层增加一个额外的输入变量add_variable来辅助预测。

```python
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_add = nn.Linear(1001, 10, bias=True)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x, add_variable):
        x = self.net(x)
        x = torch.cat((self.dropout(self.relu(x)), add_variable.unsqueeze(1)),1)
        x = self.fc_add(x)
        x = self.output(x)
        return x
```

- 这里的实现要点是通过torch.cat实现了tensor的拼接。torchvision中的resnet50输出是一个1000维的tensor，我们通过修改forward函数（配套定义一些层），先将1000维的tensor通过激活函数层和dropout层，再和外部输入变量"add_variable"拼接，最后通过全连接层映射到指定的输出维度10。
- "add_variable"进行unsqueeze操作是为了和net输出的tensor保持维度一致

```python
import torchvision.models as models
net = models.resnet50()
model = Model(net).cuda()
outputs = model(inputs, add_var)
```

### **添加额外输出**

我们依然以resnet50做10分类任务为例，在已经定义好的模型结构上，同时输出1000维的倒数第二层和10维的最后一层结果。

```python
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 10, bias=True)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x, add_variable):
        x1000 = self.net(x)
        x10 = self.dropout(self.relu(x1000))
        x10 = self.fc1(x10)
        x10 = self.output(x10)
        return x10, x1000
```

```python
import torchvision.models as models
net = models.resnet50()
model = Model(net).cuda()
out10, out1000 = model(inputs, add_var)
```

## **PyTorch模型保存与读取**

```python
from torchvision import models
model = models.resnet152(pretrained=True)

# 保存整个模型
torch.save(model, save_dir)
# 保存模型权重
torch.save(model.state_dict, save_dir)
# 保存+读取整个模型
torch.save(model, save_dir)
loaded_model = torch.load(save_dir)
loaded_model.cuda()
# 保存+读取模型权重
torch.save(model.state_dict(), save_dir)
loaded_dict = torch.load(save_dir)
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model.state_dict = loaded_dict
loaded_model.cuda()
```