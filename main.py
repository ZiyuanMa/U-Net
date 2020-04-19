import torch
import torch.nn as nn
from model import Network
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(90), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

class UNetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transforms = transform

        # cut all images to 64x64 due to GPU memory limit 

        self.imgs_path = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.imgs = []
        for path in self.imgs_path:
            img_path = os.path.join(self.root, "images", path)
            img = Image.open(img_path)
            img = transforms.functional.center_crop(img, 512)
            img = transforms.functional.to_tensor(img)
            img = img.unfold(1, 64, 32).unfold(2, 64, 32).permute(1,2,0,3,4).contiguous().view(225, 3, 64, 64)
            for i, sub_img in enumerate(img):
                # discard corner image 
                if i not in [0, 1, 13, 14, 15, 29, 195, 209, 210, 211, 223, 224]:
                    self.imgs.append(sub_img)

        self.targets_path = list(sorted(os.listdir(os.path.join(root, "1st_manual"))))
        self.targets = []
        for path in self.targets_path:
            target_path = os.path.join(self.root, "1st_manual", path)
            target = Image.open(target_path)
            target = transforms.functional.center_crop(target, 512)
            target = transforms.functional.to_tensor(target)
            target = target.unfold(1, 64, 32).unfold(2, 64, 32).permute(1,2,0,3,4).contiguous().view(225, 1, 64, 64)
            for i, sub_target in enumerate(target):
                if i not in [0, 1, 13, 14, 15, 29, 195, 209, 210, 211, 223, 224]:
                    self.targets.append(sub_target)


    def __getitem__(self, idx):
        

        # seed so image and target have the same random tranform
        seed = random.randint(0,2**32)

        img = self.imgs[idx]
        random.seed(seed)
        img = self.transforms(img)

        target = self.targets[idx]
        random.seed(seed)
        target = self.transforms(target)

        return img.to(device), target.to(device)

    def __len__(self):

        return len(self.imgs)


class model:
    def __init__(self, data_path):
        self.network = Network().to(device)
        self.data_path = data_path
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-4)
        self.loss_fn = nn.BCELoss()
        self.epoch = 50

    def train(self):
        training_set = UNetDataset(os.path.join(self.data_path, 'training'), trans)
        training_loader = DataLoader(training_set, batch_size=24, shuffle=True)
        for i in range(self.epoch):
            print('{} epoch {} {}'.format('=' * 10, i, '=' * 10))
            sum_loss = 0
            
            for img, target in tqdm(training_loader):
                predict = self.network(img)
                
                self.optimizer.zero_grad()
                loss = self.loss_fn(predict, target)
                loss.backward()
                sum_loss += loss.item()

            sum_loss /= 178
            print('loss: {}'.format(sum_loss))

            torch.save(self.network.state_dict(), './model{}.pth'.format(i))

    # def predict(self):
    #     self.network.load_state_dict(torch.load('./model.pth'))
    #     img = Image.open('./DRIVE/testing/images/01_test.tif')
    #     img = np.expand_dims(np.transpose(np.array(img, dtype=np.float32), (2, 0, 1)), axis=0)
    #     img = torch.from_numpy(img).to(device)
    #     predict = self.network(img)

    #     plt.imshow(predict.permute(1,2,0))
    
    def test(self):
        self.network.load_state_dict(torch.load('./model.pth'))
        self.network.to('cpu')
        self.network.eval()

        self.imgs_path = list(sorted(os.listdir(os.path.join(self.data_path, "/testing/images"))))
        self.targets_path = list(sorted(os.listdir(os.path.join(self.data_path, "/testing/1st_manual"))))
        
        ac_list = []
        with torch.no_grad():
            for img_path, target_path in zip(self.imgs_path, self.targets_path):
                img = Image.open(img_path)
                img = transforms.functional.center_crop(img, 560)
                img = transforms.functional.to_tensor(img)

                target = Image.open(target_path)
                target = transforms.functional.center_crop(target, 560)
                target = np.array(target)

                predict = self.network(img)
                predict = predict.numpy()
                predict = predict>=0.5

                TP = np.sum(np.logical_and(predict == 1, target == 1))
                TN = np.sum(np.logical_and(predict == 0, target == 0))
                # FP = np.sum(np.logical_and(predict == 1, target == 0))
                # FN = np.sum(np.logical_and(predict == 0, target == 1))
                AC = (TP+TN)/(560*560)
                ac_list.append(AC)
                # print(AC)

        print('accuracy: %.4f' %(sum(ac_list)/20))


if __name__ == '__main__':

    m = model('./DRIVE')
    m.train()
    # m.predict()