import torch
import torch.nn as nn
from torchvision.models import resnet18
from loss import loss_func1,loss_func2,loss_func3
from torch.utils.data import DataLoader
from PIL import Image
# from preprocess import setup_seed
model1 = resnet18(pretrained=True)
model2 = resnet18(pretrained=True)
import pprint
classifier = nn.Sequential()
model1.fc = classifier
model2.fc = classifier
model3 = nn.Linear(1024,5,bias=True)

model1.load_state_dict(torch.load("./epoch20_parameters_model1_twostreamResnet18_MSELoss.pkl"))
model2.load_state_dict(torch.load("./epoch20_parameters_model2_twostreamResnet18_MSELoss.pkl"))
model3.load_state_dict(torch.load("./epoch20_parameters_model3_twostreamResnet18_MSELoss.pkl"))


from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn, segfn, label = self.imgs[index]
        img1 = Image.open(fn).convert('RGB')
        img2 = Image.open(segfn).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)

        img2 = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor()
            ])(img2)

        return img1, img2, label

    def __len__(self):
        return len(self.imgs)

from torchvision import transforms
myTransform = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.08,contrast=0.08,saturation=0.08,hue=0.03),
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()
    ]),
}

# setup_seed(28)
fh = open('F:/PostGraduate/TaskOne/grading.txt', 'r')
full_imgs = []
flag = 0

for line in fh:
    if flag == 0:
        flag = flag + 1
        continue
    line = line.rstrip()
    words = line.split(", ")
    full_imgs.append(("F:/PostGraduate/TaskOne/images/" + words[1] + ".png", "F:/PostGraduate/TaskOne/segmentation_images/" + words[1] + ".png" , int(words[2])))

import random
numOfData = len(full_imgs)
rate = 0.4
random.seed(42)
# testlist=random.sample(list(range(0,90)),int(numOfData*rate))

test_imgs = random.sample(full_imgs, int(numOfData * rate))
test_dataset = MyDataset(test_imgs,transform=myTransform['test'])
test_dataLoader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)
device = torch.device('cuda')
model1.to(device)
model2.to(device)
model3.to(device)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition # 这⾥应⽤了⼴播机制

from plotFigure import ConfusionMatrix
confusion = ConfusionMatrix(num_classes=5, labels=['level 0', 'level 1', 'level 2', 'level 3', 'level 4'])
test_loss_list=[]

model1.eval()
model2.eval()
model3.eval()
with torch.no_grad():
    test_loss = 0
    for j, data in enumerate(test_dataLoader):
        inputs1, inputs2, labels = data
        inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
        # labels=labels.unsqueeze(1)
        outputs1 = model1(inputs1)
        outputs2 = model2(inputs2)
        outputs = torch.concat((outputs1, outputs2), dim=1)
        outputs = model3(outputs)
        # print(outputs)
        outputs = softmax(outputs)
        # loss=loss_func1(outputs,labels)+loss_func2(outputs,labels)
        score_tensor=torch.tensor([0.,1.,2.,3.,4.]).cuda()
        outputs=outputs.mm(score_tensor.view(-1,1)).squeeze(dim=1)

        loss = loss_func3(outputs,labels)
        # print(loss)
        # batch_size = inputs.size(0)
        # print("step={}; loss={};".format(i,loss.item()))
        test_loss += float(loss.item())
        predictions = torch.round(outputs)
        # confusion_matrix
        confusion.update(predictions.to(torch.int).cpu().numpy(),labels.to(torch.int).cpu().numpy())

    test_loss = test_loss / len(test_dataLoader)
    # print("val accuracy is: {}%".format(total_accuracy*100/len(val_dataLoader)))
    print("the test loss is: {}".format(test_loss))
    confusion.plot()

    # print("val accuracy is: {}%".format(total_accuracy*100/len(val_dataLoader)))
    # print("the validation loss of {}th epoch is: {}".format(epoch, loss_per_epoch))

