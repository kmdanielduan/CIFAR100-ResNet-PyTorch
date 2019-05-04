# CS 398 Deep Learning @ UIUC

## Homework 5 Deep Residual Neural Network for CIFAR100

Name: Yawen Duan		UIN: 655877290

### **HW5 Description:**

You will learn how to build very deep convolutional networks, using Residual Networks (ResNets). In theory, very deep networks can represent very complex functions; but in practice, they are hard to train. Residual Networks, introduced by He et al., allow you to train much deeper networks than were previously practically feasible. In this assignment, you will implement the basic building blocks of ResNets, and then put together these building blocks to implement and train the neural network for image classification on CIFAR100 dataset via Pytorch. Moreover, in this assignment, you will also learn how to load the pre-trained ResNets which was trained on ImageNet dataset and train it on CIFAR100.

### Implementation

#### Part 1

In my code, I defined an class object  `ResNet` to obtain the model to be used. The hyperparameters are listed as follows:

```python
batch_size = 50
learning_rate = 0.001
num_workers = 4
# Note that we can obtain the pretrained model to reduce training time
load_pretrained_model = True
epochs = 50
pretrained_epoch = 15
```

The model consists of multiple basic block layers other layers, with max-pooling and dropouts between some layers as described in the homework description. The structure of the model can be shown as follows:

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, repeat = 1):
        super(BasicBlock, self).__init__()
        
        self.conv_init = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=padding, bias=True)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.convShortcut_init = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                      padding=0, bias=True)
        self.convShortcut = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1,
                                      padding=0, bias=True)
        self.repeat = repeat
        self.equalInOut = (in_channels == out_channels)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
    def forward(self, x):
        out = x.clone()
        y = x.clone()
        out = self.conv_init(out)
        out = F.relu(self.bn(out))
        out = self.bn(self.conv(out))
        if self.equalInOut:
            out = torch.add(out, x)
        else:
            self.convShortcut.in_channels = y.size()[1]
            out = torch.add(self.convShortcut_init(y), out)
        out = F.relu(out)     
        
        for i in range(1, self.repeat):
            y = out.clone()
            out = self.conv(out)
            i += 1
            out = F.relu(self.bn(out))
            out = self.bn(self.conv(out))
            self.convShortcut.in_channels = y.size()[1]
            out = torch.add(self.convShortcut(y), out)
            out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        nChannels = [32 ,32, 64, 128, 256]
        nStride = [1,2,2,2]
        nRepeat = [2,4,4,2]
        
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # batch normal layer
        self.bn = nn.BatchNorm2d(nChannels[0]) 
        # dropout layer
        self.Dropout = nn.Dropout(0.25)
        # 1st block
        self.block1 = BasicBlock(in_channels = nChannels[0], out_channels = nChannels[1], 
                                 stride = nStride[0], padding = 1, repeat = nRepeat[0])
        # 2nd block
        self.block2 = BasicBlock(in_channels = nChannels[1], out_channels = nChannels[2], 
                                 stride = nStride[1], padding = 1, repeat = nRepeat[1])
        # 3rd block
        self.block3 = BasicBlock(in_channels = nChannels[2], out_channels = nChannels[3], 
                                 stride = nStride[2], padding = 1, repeat = nRepeat[2])
        # 4th block
        self.block4 = BasicBlock(in_channels = nChannels[3], out_channels = nChannels[4], 
                                 stride = nStride[3], padding = 1, repeat = nRepeat[3])
        # maxpooling layer
        self.pool = nn.MaxPool2d(4,4)
        
        # classifier
        self.fc = nn.Linear(in_features=nChannels[4], out_features=100)
        self.nChannels = nChannels[-1]


    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn(out))
        out = self.Dropout(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool(out)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out
```

In this model, I apply data augmentation to the dataset through random horizontal flip. This model is trained with ADAM optimizer.



#### Part 2

In this part, I applied upsampling to the training dataset and the testing dataset. And during the training loop, I only train the layer 4 and the fully connected layer. 



### Test Result 

#### Part 1

The test result achieved an accuracy score of above 64% after 15 epochs.

#### Part 2

The test result achieved an accuracy score of above 76% after 15 epochs.