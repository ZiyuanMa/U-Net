import torch
import torch.nn as nn
from model import Network
import torchvision
from PIL import Image
import os
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

trans = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.RandomRotation(180), 
    transforms.RandomCrop(128),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

class UNetDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.training = True if os.path.exists(os.path.join(root, "1st_manual")) else False

        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "mask"))))
        if self.training:
            self.targets = list(sorted(os.listdir(os.path.join(root, "1st_manual"))))


    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        # mask_path = os.path.join(self.root, "mask", self.masks[idx])
        if self.training:
            target_path = os.path.join(self.root, "1st_manual", self.targets[idx])

        seed = random.randint(0,2**32)

        img = Image.open(img_path)
        random.seed(seed)
        img = self.transforms(img)
        # print(img.shape)

        # mask = Image.open(mask_path)
        # mask = self.transforms(mask)

        if self.training:
            target = Image.open(target_path)
            random.seed(seed)
            target = self.transforms(target)
            # target = np.expand_dims(np.array(target), axis=0)
            # print(img.shape)
            # img = np.concatenate((img, target), axis=0)

        # img = np.expand_dims(img, axis=0)

        # img = np.expand_dims(img, axis=0)

        if self.training:
            # print(img.size())
            # print(target.size())
            return img.to(device), target.to(device)
        else:
            return img.to(device)

    def __len__(self):

        return len(self.imgs)


class model:
    def __init__(self, data_path):
        self.network = Network()
        self.data_path = data_path
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.epoch = 100

    def train(self):
        training_set = UNetDataset(os.path.join(self.data_path, 'training'), trans)
        training_loader = DataLoader(training_set, batch_size=4, shuffle=True, num_workers=2)
        for i in range(self.epoch):
            print('{} epoch {} {}'.format('=' * 10, i, '=' * 10))
            sum_loss = 0
            
            for img, target in tqdm(training_loader):
                predict = self.network(img)
                
                self.optimizer.zero_grad()
                loss = self.loss_fn(predict, target)
                loss.backward()
                sum_loss += loss.item()

            sum_loss /= 5
            print('loss: {}'.format(sum_loss))
                


if __name__ == '__main__':
    m = model('./DRIVE')
    m.train()