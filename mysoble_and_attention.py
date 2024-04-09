from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import time
from torchvision.datasets import ImageFolder
import numpy as np
from tqdm import tqdm
from my_sobel.sheartransform import calculate_non_black_ratio,split_image_into_blocks,process_blocks
from torchvision.transforms import Lambda
torch.cuda.set_device(0)
from torch.utils.data import Subset
from PIL import Image
trans_size=(128,164)
import random
class RandomBrightness:
    def __init__(self, brightness_range):
        self.min_brightness, self.max_brightness = brightness_range

    def __call__(self, img):
        brightness_factor = random.uniform(self.min_brightness, self.max_brightness)
        img_array = np.array(img).astype(float)
        img_array *= brightness_factor
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
transformations_split_train = transforms.Compose([
    transforms.Resize(trans_size),
    transforms.RandomRotation(30),  # 添加随机旋转，最大旋转角度为30度
    RandomBrightness((0.01, 1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transformations_split_test = transforms.Compose([
    transforms.Resize(trans_size),
    transforms.RandomRotation(30),
    RandomBrightness((0.01, 1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def split_image_train(img1):
    width = img1.size[0]
    half_width = width // 2
    left_img = img1.crop((0, 0, half_width, img1.size[1]))
    right_img = img1.crop((half_width, 0, width, img1.size[1]))
    left_img1 = transformations_split_train(left_img)
    right_img1 = transformations_split_train(right_img)

    return left_img1, right_img1
def split_image_test(img2):
    width = img2.size[0]
    half_width = width // 2
    left_img = img2.crop((0, 0, half_width, img2.size[1]))
    right_img = img2.crop((half_width, 0, width, img2.size[1]))
    left_img2 = transformations_split_test(left_img)
    right_img2 = transformations_split_test(right_img)
    return left_img2, right_img2

transformations = transforms.Compose([
    Lambda(split_image_train),
])
brightness_decrease = transforms.Compose([
    Lambda(split_image_test),
])

root = "/home/pyquan/sysb/sysb_data/pin_jie/pinjie_end"
batch_size = 32
def data_loader():
    dataset = ImageFolder(root)
    classes_num = len(dataset.classes)
    train_size = int(0.85 * len(dataset)) 
    torch.manual_seed(0)
    indices = np.random.permutation(len(dataset))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_dataset = ImageFolder(root, transform=transformations)
    train_set = Subset(train_dataset, train_indices)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = ImageFolder(root, transform=brightness_decrease)
    test_set = Subset(test_dataset, test_indices)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)    
    print("The number of images in the training set is: ", len(train_loader)*batch_size)
    print("The number of images in the test set is: ", len(test_loader)*batch_size)
    print("The number of batches per epoch is: ", len(train_loader))
    return train_loader,test_loader,classes_num,dataset
class ImageBlockProcessingModule(nn.Module):
    def __init__(self,pixels_per_block,):
        super(ImageBlockProcessingModule, self).__init__()
        self.pixels_per_block=pixels_per_block
    def forward(self, input):
        image_gray = calculate_non_black_ratio(input)
        blocks = split_image_into_blocks(image_gray,self.pixels_per_block,)
        output = process_blocks(blocks, 4)
        output = output.to(input.device)
        return output
from torchvision.models import mobilenet_v2
import torch.nn.functional as F
from torch.nn import MultiheadAttention
class Network(nn.Module):
    def __init__(self,classes_num,pixels_per_block):
        super(Network, self).__init__()
        self.image_block_processing = ImageBlockProcessingModule(pixels_per_block)
        # 亮度估计
        self.brightness_estimation_net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),  
            nn.ReLU(),
            nn.MaxPool2d(4),  # 使用更大的池化窗口
            nn.Conv2d(16, 3, 5, stride=2, padding=2),  
            nn.Sigmoid()
        )
        self.mobilenet_v2_net = mobilenet_v2(pretrained=True)
        self.conv = nn.Conv2d(self.mobilenet_v2_net.features[-1].out_channels, 128, kernel_size=1)
        in_features = 128  
        self.mobilenet_v2_net.classifier[1] = nn.Linear(256, classes_num)
        self.attention = MultiheadAttention(embed_dim=in_features, num_heads=4, dropout=0.05)
    def forward(self, input1,input2):
        edge1 = self.image_block_processing(input1)
        edge2 = self.image_block_processing(input2)
        # 获取input1的形状
        input_shape = input1.shape[2:]
        # 使用插值使得edge1和input1的形状相同
        edge1 = F.interpolate(edge1, size=input_shape, mode='bilinear', align_corners=False)
        edge2 = F.interpolate(edge2, size=input_shape, mode='bilinear', align_corners=False)

        brightness1 = self.brightness_estimation_net(input1)
        brightness2 = self.brightness_estimation_net(input2)
        brightness1 = brightness1.mean(dim=(2, 3))
        # 将brightness扩展到与edge和input相同的形状
        brightness1 = brightness1.unsqueeze(-1).unsqueeze(-1)
        brightness1 = brightness1.expand_as(input1)
        brightness2 = brightness2.mean(dim=(2, 3))
        brightness2 = brightness2.unsqueeze(-1).unsqueeze(-1)
        brightness2 = brightness2.expand_as(input2)
        edge1 = (1 - brightness1) *edge1 +  brightness1*input1
        edge2 = (1 - brightness2) *edge2 +  brightness2*input2
        output1 = self.mobilenet_v2_net.features(edge1)
        output1 = self.conv(output1)
        output2 = self.mobilenet_v2_net.features(edge2)
        output2 = self.conv(output2)
        output1 = output1.mean([2, 3])  # 平池化
        output2 = output2.mean([2,3])  # 增加一个维度，以适应多头自注意力层的输入格式
        output1 = output1.unsqueeze(0)  # 增加一个维度，以适应多头自注意力层的输入格式
        output1, _ = self.attention(output1, output1, output1)  # 对输出进行多头自注意力计算
        output1 = output1.squeeze(0)  # 去掉多余的维度
        output2 = output2.unsqueeze(0) 
        output2, _ = self.attention(output2, output2, output2)  
        output2 = output2.squeeze(0)  
        output = torch.cat((output1,output2),1)
        output = self.mobilenet_v2_net.classifier(output)
        return output
from torch.cuda.amp import GradScaler, autocast

        
def saveUsedWeights(model, epoch, classes):
    model_info = {'model': model.state_dict(), 'classes': classes}
    torch.save(model_info, 'model/model.pth')
    print('已保存第{}轮模型'.format(epoch+1))
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    with torch.no_grad():
        for data in test_loader:
            (left_images, right_images), labels = data
            left_images = left_images.cuda()
            right_images = right_images.cuda()
            labels = labels.cuda()
            outputs = model(left_images, right_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    accuracy = (100.0 * accuracy / total)
    model.train()
    return accuracy
import matplotlib.pyplot as plt
import numpy as np
def testModelAccuracy(classes_num):
    class_correct = list(0. for i in range(classes_num))
    class_total = list(0. for i in range(classes_num))
    confusion_matrix = np.zeros((classes_num, classes_num))
    with torch.no_grad():
        for data in test_loader:
            (left_images, right_images), labels = data
            left_images = left_images.cuda()
            right_images = right_images.cuda()
            labels = labels.cuda()
            outputs = model(left_images, right_images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                confusion_matrix[label.long(), predicted[i].long()] += 1
    for i in range(classes_num):
        if class_total[i] != 0:
            accuracy = 100 * class_correct[i] / class_total[i]
        else:
            accuracy = 0
        print('Accuracy of %5s : %2d %%' % (classes[i], accuracy))
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], '.2f'),
                    ha="center", va="center",
                    color="red" if confusion_matrix[i, j] > 0.5 else "black")

    plt.savefig('confusion_matrix.svg', format='svg')

def train(num_epochs,train_loader,dataset):
    best_accuracy = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("模型将在", device, "上运行")
    model.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        iteration_count = 0
        for i, ((left_images, right_images), labels) in tqdm(enumerate(train_loader, 0), desc="Processing images"):
            left_images = left_images.to(device)
            right_images = right_images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            left_images = left_images.cuda()
            right_images = right_images.cuda()
            labels = labels.cuda()
            outputs = model(left_images, right_images)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            running_acc += correct / (labels.size(0))

            running_loss += loss.item()

            iteration_count += 1

            if iteration_count % 1000 == 0 or i == len(train_loader)-1:
                average_loss = running_loss / iteration_count
                average_acc = running_acc / iteration_count
                print('[%d, %5d] 平均损失: %.3f, 平均准确率: %.3f' % (epoch + 1, iteration_count, average_loss, average_acc))
                running_loss = 0.0
                running_acc = 0.0
                iteration_count = 0
            
        accuracy = testAccuracy()
        print('对于第', epoch + 1, '轮，整个测试集的准确率为 %.3f %%' % (accuracy))
        if accuracy > best_accuracy:
            saveUsedWeights(model, epoch, dataset.classes)
            best_accuracy = accuracy


import time

def testBatch():
    class_correct = [0.0] * len(classes)
    class_total = [0.0] * len(classes)
    accuracy_dict = {}
    total_images = 0
    total_time = 0.0
    
    with torch.no_grad():
        start_time = time.time()
        for data in test_loader:
            
            (left_images, right_images), labels = data
            left_images = left_images.cuda()
            right_images = right_images.cuda()
            labels = labels.cuda()
            outputs = model(left_images, right_images)
            
            
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
            
            total_images += len(labels)
        end_time = time.time()
        total_time = end_time - start_time
    for i in range(len(classes)):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] != 0 else 0.0
        accuracy_dict[classes[i]] = accuracy
        print('类别 %5s 的准确度: %.3f %%' % (classes[i], accuracy))
    
    avg_time_per_image = total_time / total_images
    print("平均每张图片推理时间: {:.8f} 秒".format(avg_time_per_image))
    return accuracy_dict





if __name__ == "__main__":
   

    train_loader,test_loader,classes_num,dataset = data_loader()

    device = torch.device('cuda')  
    pixels_per_block=4
    
    model = Network(classes_num,pixels_per_block)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)

    model.train()
    startime = time.time() 
    train_number=40
    train(train_number,train_loader,dataset)
    print('Finished Training')
    endtime = time.time()
    train_duration = endtime-startime
    print('Running time: %s Seconds'%(train_duration))
    
    
    path = "model/model.pth"
    model_info = torch.load(path)
    classes = model_info['classes']
    classes_num = len(classes)
    # 获取模型和类别的顺序
    model = Network(classes_num,pixels_per_block)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    model = model.to(device)
    model.load_state_dict(model_info['model'])
    model.eval()
    testModelAccuracy(classes_num) 
    start_time = time.time()
    accuracy = testBatch()
    end_time = time.time()
    test_duration = end_time - start_time
    print('Running time: %s Seconds'%(test_duration))

    



