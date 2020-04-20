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
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# random image transformation to do image augmentation
trans = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomRotation(90), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: x/255)
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
            img = img.unfold(1, 96, 32).unfold(2, 96, 32).permute(1,2,0,3,4).contiguous().view(196, 3, 96, 96)
            for i, sub_img in enumerate(img):
                # discard corner image 
                if i not in [0, 1, 13, 14, 15, 29, 195, 209, 210, 211, 223, 224]:
                    self.imgs.append(sub_img)
        
        self.masks_path = list(sorted(os.listdir(os.path.join(root, "mask"))))
        self.masks = []
        for path in self.masks_path:
            mask_path = os.path.join(self.root, "mask", path)
            mask = Image.open(mask_path)
            mask = transforms.functional.center_crop(mask, 512)
            mask = transforms.functional.to_tensor(mask)
            mask = mask.unfold(1, 96, 32).unfold(2, 96, 32).permute(1,2,0,3,4).contiguous().view(196, 1, 96, 96)
            for i, sub_mask in enumerate(mask):
                # discard corner image 
                if i not in [0, 1, 13, 14, 15, 29, 195, 209, 210, 211, 223, 224]:
                    self.masks.append(sub_mask)

        self.targets_path = list(sorted(os.listdir(os.path.join(root, "1st_manual"))))
        self.targets = []
        for path in self.targets_path:
            target_path = os.path.join(self.root, "1st_manual", path)
            target = Image.open(target_path)
            target = transforms.functional.center_crop(target, 512)
            target = transforms.functional.to_tensor(target)
            target = target.unfold(1, 96, 32).unfold(2, 96, 32).permute(1,2,0,3,4).contiguous().view(196, 1, 96, 96)
            for i, sub_target in enumerate(target):
                if i not in [0, 1, 13, 14, 15, 29, 195, 209, 210, 211, 223, 224]:
                    self.targets.append(sub_target)


    def __getitem__(self, idx):
        
        # seed so image and target have the same random tranform
        seed = random.randint(0,2**32)

        img = self.imgs[idx]
        random.seed(seed)
        img = self.transforms(img)


        mask = self.masks[idx]
        random.seed(seed)
        mask = self.transforms(mask)

        target = self.targets[idx]
        random.seed(seed)
        target = self.transforms(target)

        # fig = plt.figure()
        # ax1 = fig.add_subplot(1,3,1)
        # ax1.imshow(img.permute((1,2,0)))
        # ax2 = fig.add_subplot(1,3,2)
        # ax2.imshow(torch.squeeze(mask), cmap="gray")
        # ax3 = fig.add_subplot(1,3,3)
        # ax3.imshow(torch.squeeze(target), cmap="gray")
        # plt.show()


        return img.to(device), mask.to(device), target.to(device)

    def __len__(self):

        return len(self.imgs)


class model:
    def __init__(self, data_path):
        self.data_path = data_path
        
        self.network = Network()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        self.loss_fn = nn.BCELoss()

        # training iteration
        self.epoch = 100

    def train(self):
        self.network.to(device)
        self.network.train()
        training_set = UNetDataset(os.path.join(self.data_path, 'training'), trans)
        training_loader = DataLoader(training_set, batch_size=16, shuffle=True)
        for i in range(self.epoch):
            print('{} epoch {} {}'.format('=' * 10, i, '=' * 10))
            sum_loss = 0
            
            for img, mask, target in tqdm(training_loader):

                predict = self.network(img)

                predict = predict.view(predict.size(0), -1)
                predict = predict * mask.view(mask.size(0), -1)
                target = target.view(target.size(0), -1)

                self.optimizer.zero_grad()
                loss = self.loss_fn(predict, target)
                sum_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            sum_loss /= 178
            print('loss: {}'.format(sum_loss))

            if i % 10 == 0:
                torch.save(self.network.state_dict(), './model{}.pth'.format(i))

        torch.save(self.network.state_dict(), './model.pth')

    # def predict(self):
    #     self.network.load_state_dict(torch.load('./model.pth'))
    #     img = Image.open('./DRIVE/testing/images/01_test.tif')
    #     img = np.expand_dims(np.transpose(np.array(img, dtype=np.float32), (2, 0, 1)), axis=0)
    #     img = torch.from_numpy(img).to(device)
    #     predict = self.network(img)

    
    def test(self):
        '''
        run test set
        '''
        # load saved model
        self.network.to('cpu')
        self.network.load_state_dict(torch.load('./model95.pth', map_location=torch.device('cpu')))
        self.network.eval()

        # load test set
        self.imgs_path = list(sorted(os.listdir(os.path.join(self.data_path, "testing/images"))))
        self.targets_path = list(sorted(os.listdir(os.path.join(self.data_path, "testing/1st_manual"))))
        
        ac_list = []
        with torch.no_grad():
            for img_name, target_name in zip(self.imgs_path, self.targets_path):
                img_path = os.path.join(self.data_path, "testing/images", img_name)
                img = Image.open(img_path)
                img = transforms.functional.center_crop(img, 560)
                img = transforms.functional.to_tensor(img).unsqueeze(0)

                target_path = os.path.join(self.data_path, "testing/1st_manual", target_name)
                target = Image.open(target_path)
                target = transforms.functional.center_crop(target, 560)
                target = np.array(target)

                predict = self.network(img)
                predict = np.squeeze(predict.numpy(), axis=(0,1))
                predict = (predict>=0.5).astype(np.uint8)

                TP = np.sum(np.logical_and(predict == 1, target == 1))
                TN = np.sum(np.logical_and(predict == 0, target == 0))
                # FP = np.sum(np.logical_and(predict == 1, target == 0))
                # FN = np.sum(np.logical_and(predict == 0, target == 1))
                AC = (TP+TN)/(560*560)
                ac_list.append(AC)

                # print(AC)

                # show predicted image
                fig = plt.figure()
                ax1 = fig.add_subplot(1,2,1)
                ax1.imshow(predict, cmap="gray")
                ax2 = fig.add_subplot(1,2,2)
                ax2.imshow(target, cmap="gray")
                # plt.imshow(predict, cmap="gray")
                # plt.imshow(target, cmap="gray")
                plt.show()

        print('accuracy: %.4f' %(sum(ac_list)/20))


if __name__ == '__main__':

    m = model('./DRIVE')
    m.train()
    # m.test()