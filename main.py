import torch
import torchvision
import torch.nn as nn
import numpy as np
from data import Cute



class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.fc1 = nn.Sequential(nn.Linear(12800, 4096), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(4096, 4096), nn.ReLU())
        self.out = nn.Linear(4096, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x


net = Net()
net.cuda()

print(net)


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

c = Cute().load_dir("/home/yi/Workspace/Datasets/cat_dog")

labels_np , images_np = c.next()
images = torch.from_numpy(images_np).cuda()
print(images.shape)


for step in range(10000):

    labels_np , images_np = c.next()

    labels = torch.from_numpy(labels_np).long().cuda()
    images_np = np.transpose(images_np, (0, 3, 1, 2))
    images = torch.from_numpy(images_np).float().cuda()

    predict = net(images)
    loss = loss_func(predict, labels)

    if step % 10 == 0 :
        predict_item = predict.clone()
        predict_item = predict_item.cpu().detach().numpy()
        acc = sum([int(labels_np[i] == np.where(predict_item[i] == np.amax(predict_item[i]))) for i in range(len(labels_np))])/len(labels_np)
        print("step:", step, "acc: %.3f" % acc, "loss: %f" % loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
