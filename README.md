# Faster RCNN

## 原理

![frcnn.jpg](https://i.loli.net/2020/10/21/XcVnj2xTdE7Iqhu.jpg)

Faster RCNN其实主要可以分为四个内容：

1. Conv layers。作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。
2. Region Proposal Networks。RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。
3. Roi Pooling。该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。
4. Classification。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。

## Conv layers

实际上就是传统图像分类中的经典网络，比如VGG、ResNet。在目标检测中是作为特征提取的骨干网络，它不直接参与框的预测，而是输出特征层。所以，它的最后一层不是输出类别数，而是输出一个宽高可变的特征层。

## Region Proposal Networks

上图绿框内展示了RPN网络的具体结构。可以看到RPN网络实际分为2条线，上面一条通过softmax分类anchors获得前景和背景分类（通道数为18是2x9，一共有9个先验框，2是采用多分类交叉熵，若采用二元交叉熵就是1），下面一条用于计算对于anchors的bounding box regression偏移量，以获得精确的proposal（这里同理是4x9，4则是代表候选框在rpn上的坐标）。而最后的Proposal层则负责综合positive anchors和对应bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。其实整个网络到了Proposal Layer这里，就完成了相当于目标定位的功能。

## Roi Pooling

将输入的特征层看作是图像，用rpn生成的候选框截取的图像，然后resize成 pool_size * pool_size的大小。这样处理后，即使大小不同的proposal输出结果都是固定大小，实现了固定长度输出。

## Classification

从RoI Pooling层获取到固定大小的proposal feature maps后，送入后续网络，可以看到做了如下2件事：

1. 通过全连接和softmax对proposals进行分类，这实际上已经是识别的范畴了
2. 再次对proposals进行bounding box regression，获取更高精度的rect box



## 快速开始

1. 下载代码

```python
https://github.com/Runist/Faster_RCNN.git
```

2. 安装依赖库

```python
$ pip install -r requirements.txt
```

3. 下载权重文件

```python
$ wget https://github.com/Runist/Faster_RCNN/releases/download/v1.0/faster_rcnn.h5   
```

4. 将config/config.py中的文件信息修改至你的路径下

5. 修改predict/predcit.py的图像路径，运行预测代码

```python
$ python predict.py
```

<img src="https://i.loli.net/2020/09/04/pRtZ5FYhNc72olu.png" alt="show.png" align=center style="zoom: 200%;" />



## config.py - 配置文件

较为常用的配置文件一般是cfg、json格式的文件，因为没有原作者的框架复杂，所以在配置文件采用的是py格式，也方便各个文件的调用。在config.py中也有了较为详细的注释，如果有不懂欢迎在issues中提问。



## 如何训练自己的数据集

你的数据集放在什么位置都可以。但是你需要在config.py的annotation_path指定你的图片与真实框信息，如：D:/Dataset/VOCdevkit/VOC2012/JPEGImages/2007_000039.jpg 156,89,344,279,19

如果想要使用ImageNet的预训练权重

```python
$ wget https://github.com/Runist/Faster_RCNN/releases/download/v1.0/faster_rcnn.h5  
```

配置好config.py文件后，运行如下代码

```python
$ python train.py
```