import torch
import torch.nn as nn
import torch.optim as optim
from model import R2UNet, UNet, IterNet
import torchvision
from PIL import Image
import os
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


trans_fn1 = transforms.Compose([
    transforms.CenterCrop(512),
    transforms.RandomRotation(180),
    transforms.ToTensor(),
])

# random image transformation to do image augmentation
trans_fn2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

class UNetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transforms = transform
        seed = random.randint(0, 2**32)
        self.imgs_path = list(sorted(os.listdir('./DRIVE/training/images')))
        self.imgs = []
        for path in self.imgs_path:
            img_path = os.path.join('./DRIVE/training/images', path)
            img = Image.open(img_path)
            # image augmentation
            random.seed(seed) 
            img = trans_fn1(img)
            # cut the 512x512 image to 96x96 with stride 32 which is 196 sub images
            img = img.unfold(1, 96, 32).unfold(2, 96, 32).permute(1,2,0,3,4).contiguous().view(196, 3, 96, 96)
            for i, sub_img in enumerate(img):
                # add all sub images but not corner sub images 
                if i not in [0, 1, 14, 12, 13, 27, 168, 182, 183, 181, 194, 195]:
                    self.imgs.append(sub_img)
        
        self.masks_path = list(sorted(os.listdir(os.path.join('./DRIVE/training/mask'))))
        self.masks = []
        for path in self.masks_path:
            mask_path = os.path.join(self.root, "mask", path)
            mask = Image.open(mask_path)
            random.seed(seed)
            mask = trans_fn1(mask)
            mask = mask.unfold(1, 96, 32).unfold(2, 96, 32).permute(1,2,0,3,4).contiguous().view(196, 1, 96, 96)
            for i, sub_mask in enumerate(mask):
                if i not in [0, 1, 14, 12, 13, 27, 168, 182, 183, 181, 194, 195]:
                    self.masks.append(sub_mask)

        self.targets_path = list(sorted(os.listdir(os.path.join(root, "1st_manual"))))
        self.targets = []
        for path in self.targets_path:
            target_path = os.path.join(self.root, "1st_manual", path)
            target = Image.open(target_path)
            random.seed(seed)
            target = trans_fn1(target)
            target = target.unfold(1, 96, 32).unfold(2, 96, 32).permute(1,2,0,3,4).contiguous().view(196, 1, 96, 96)
            for i, sub_target in enumerate(target):
                if i not in [0, 1, 14, 12, 13, 27, 168, 182, 183, 181, 194, 195]:
                    self.targets.append(sub_target)


    def __getitem__(self, idx):
        
        # seed so image and target have the same random tranform
        seed = random.randint(0, 2**32)

        img = self.imgs[idx]
        random.seed(seed)
        img = trans_fn2(img)


        mask = self.masks[idx]
        random.seed(seed)
        mask = trans_fn2(mask)

        target = self.targets[idx]
        random.seed(seed)
        target = trans_fn2(target)

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

    def __init__(self, model):
        
        self.model = model

        if self.model == 'U-Net':
            self.network = UNet()
        elif self.model == 'R2U-Net':
            self.network = R2UNet()
        elif self.model == 'IterNet':
            self.network = IterNet()
            
    def train(self, epoch):
        self.network.to(device)
        self.network.train()

        optimizer = optim.Adam(self.network.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        loss_fn = nn.BCELoss()

        for i in range(epoch):
            print('{} epoch {} {}'.format('=' * 10, i, '=' * 10))
            sum_loss = 0
            training_set = UNetDataset('./DRIVE/training', trans_fn2)
            training_loader = DataLoader(training_set, batch_size=16, shuffle=True)
            for img, mask, target in tqdm(training_loader):

                predict = self.network(img)

                if self.model == 'IterNet':
                    mask = mask.view(mask.size(0), -1)
                    target = target.view(target.size(0), -1)
                    loss = 0
                    for j in range(3):
                        iter_predict = predict[j].view(predict[j].size(0), -1)
                        iter_predict = iter_predict * mask.view(mask.size(0), -1)

                        loss += loss_fn(iter_predict, target)

                    sum_loss += loss.item()
                else:
                    predict = predict.view(predict.size(0), -1)
                    predict = predict * mask.view(mask.size(0), -1)
                    target = target.view(target.size(0), -1)


                    loss = loss_fn(predict, target)
                    sum_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            sum_loss /= 230
            print('loss: {}'.format(sum_loss))

            if i % 5 == 0:
                torch.save(self.network.state_dict(), './{}{}.pth'.format(self.model, i))

        torch.save(self.network.state_dict(), './{}.pth'.format(self.model))

    
    def test(self, show):
        '''
        run test set
        '''
        # load saved model
        self.network.to('cpu')
        self.network.load_state_dict(torch.load('./{}.pth'.format(self.model), map_location=torch.device('cpu')))
        self.network.eval()

        # load test set
        self.imgs_path = list(sorted(os.listdir('./DRIVE/testing/images')))
        self.masks_path = list(sorted(os.listdir('./DRIVE/testing/mask')))
        self.targets_path = list(sorted(os.listdir('./DRIVE/testing/1st_manual')))
        
        results = []
        with torch.no_grad():
            for img_name, mask_name, target_name in zip(self.imgs_path, self.masks_path, self.targets_path):
                img_path = os.path.join('./DRIVE/testing/images', img_name)
                img = Image.open(img_path)
                img = transforms.functional.center_crop(img, 560)
                img = transforms.functional.to_tensor(img).unsqueeze(0)

                mask_path = os.path.join('./DRIVE/testing/mask', mask_name)
                mask = Image.open(mask_path)
                mask = transforms.functional.center_crop(mask, 560)
                mask = np.array(mask).flatten() / 255
                mask = mask.astype(np.uint8)


                target_path = os.path.join('./DRIVE/testing/1st_manual', target_name)
                target = Image.open(target_path)
                target = transforms.functional.center_crop(target, 560)
                target = np.array(target)
                target_ = target.flatten() / 255
                target_ = target_.astype(np.uint8)
                target_ = target_[mask==1]

                predict = self.network(img)
                if self.model == 'IterNet':
                    predict = predict[-1]
                predict = np.squeeze(predict.numpy(), axis=(0,1))
                predict_ = predict.flatten()[mask==1]
                predict_ = (predict_>=0.5).astype(np.uint8)


                TP = np.sum(np.logical_and(predict_ == 1, target_ == 1)) # true positive
                TN = np.sum(np.logical_and(predict_ == 0, target_ == 0)) # true negative
                FP = np.sum(np.logical_and(predict_ == 1, target_ == 0)) # false positive
                FN = np.sum(np.logical_and(predict_ == 0, target_ == 1)) # false negative

                AC = (TP+TN)/(TP+TN+FP+FN) # accuracy
                SE = (TP)/(TP+FN) # sensitivity
                SP = TN/(TN+FP) # specificity
                precision = TP/(TP+FP)
                recall = TP/(TP+FN)
                F1 = 2*((precision*recall)/(precision+recall))
                fpr, tpr, _ = roc_curve(target_, predict_)
                AUC = auc(fpr,tpr)
                results.append((F1, SE, SP, AC, AUC))

                # show predicted image
                if show:
                    fig = plt.figure()
                    ax1 = fig.add_subplot(1,3,1)
                    ax1.imshow(img.squeeze(0).permute((1,2,0)))
                    ax2 = fig.add_subplot(1,3,2)
                    ax2.imshow(predict, cmap="gray")
                    ax3 = fig.add_subplot(1,3,3)
                    ax3.imshow(target, cmap="gray")
                    plt.show()

        F1, SE, SP, AC, AUC = map(list, zip(*results))

        print('F1 score: %.4f' %(sum(F1)/len(F1)))
        print('sensitivity: %.4f' %(sum(SE)/len(SE)))
        print('specificity: %.4f' %(sum(SP)/len(SP)))
        print('accuracy: %.4f' %(sum(AC)/len(AC)))
        print('AUC: %.4f' %(sum(AUC)/len(AUC)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='U-Net')
    # general setting
    parser.add_argument('--model', type=str, default='U-Net', help='U-Net R2U-Net IterNet')
    parser.add_argument('--mode', type=str, default='test', help='train test')
    # training setting
    parser.add_argument('--epoch', type=int, default=45, help='training epoch')
    # testing setting
    parser.add_argument('--show', type=str, default='True', help='if show the predicted image')

    args = parser.parse_args()

    m = model(args.model)
    if args.mode == 'train':
        m.train(args.epoch)
    else:
        if args.show == 'True':
            m.test(True)
        else:
            m.test(False)