import torch
import torchvision

from torch.autograd import Variable
import torch.optim as optim


# Hyperparameters
batch_size = 50
learning_rate = 0.001
num_workers = 4
load_pretrained_model = False
epochs = 50
pretrained_epoch = 15

# CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


DIM = 224

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(DIM, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize(DIM, interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# CIFAR100 
trainset = CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainset_loader = DataLoader(trainset, batch_size=batch_size, 
                                    shuffle=True, num_workers=num_workers)

testset = CIFAR100(root='./data', train=False, download=True, transform=transform_train)
testset_loader = DataLoader(testset, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)

def save_model(net, epoch):
    PATH = "./pretrained_models/model_epoch" + str(epoch+1) + ".pth"
    torch.save(net.state_dict(),PATH)

def load_model(net, pretrained_epoch):
    PATH = "./pretrained_models/model_epoch" + str(pretrained_epoch) + ".pth"
    net.load_state_dict(torch.load(PATH))
    net.eval()

# define the test accuracy function
def test_accuracy(net, testset_loader, epoch):
    # Test the model
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testset_loader:
            images, labels = data
            images, labels = Variable(images).to(device), labels.to(device)
            output = net(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    print('Accuracy Test -- Epoch '+str(epoch+1)+': ' + str(100 * correct / total))


def l_rate(base_rate, epoch,schedule = False):
    if schedule == False:
        return base_rate
    else:
        if epoch < 10:
            return(base_rate)
        elif epoch >= 10 and epoch < 15:
            return(base_rate/10)
        elif epoch >= 15 and epoch < 25:
            return(base_rate/20)
        else:
            return(base_rate/50)


import torch.nn as nn
import torchvision.models as models
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512,100)


import torch.optim as optim
criterion = nn.CrossEntropyLoss()

model.to(device)

optimizer = optim.SGD([model.layer4.parameters(),model.fc.parameters()], 
lr = 0.001, momentum=0.9)

start_epoch = 0
if load_pretrained_model == True:
    load_model(model, pretrained_epoch)
    model.to(device)
    start_epoch = pretrained_epoch


for epoch in range(start_epoch,epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    tmp_loss = 0.0
    for i, data in enumerate(trainset_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = Variable(inputs).to(device),Variable(labels).to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        with torch.no_grad():
            h = model.conv1(inputs)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)
            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
        h = model.layer4(h)
        h = model.avgpool(h)
        h = h.view(h.size(0), -1)
        outputs = model.fc(h)
        
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        tmp_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, tmp_loss / 2000))
            tmp_loss = 0.0
    # print the loss after every epoch
    print('Epoch ' + str(epoch + 1) + ': loss = ' + str(running_loss / 50000))    
    if (epoch + 1)%5 == 0:
        # Test for accuracy after every 5 epochs
        test_accuracy(model, testset_loader, epoch)
        # Save model after every 5 epochs
        save_model(model, epoch)
    elif epoch == epochs - 1:
        test_accuracy(model, testset_loader, epoch)
        save_model(model, epoch)

print('Finished Training')