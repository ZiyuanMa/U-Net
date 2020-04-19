import torch
from model import Network
import torchvision
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

trans = transforms.Compose([
    transforms.CenterCrop((565, 565)),
    transforms.ToTensor(),
])

class UNetDataset(Dataset):
    def __init__(self, root, transforms=trans):
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
        mask_path = os.path.join(self.root, "mask", self.masks[idx])
        if self.training:
            target_path = os.path.join(self.root, "1st_manual", self.targets[idx])

        img = Image.open(img_path)
        img = self.transforms(img)

        mask = Image.open(mask_path)
        mask = self.transforms(mask)

        if self.training:
            target = Image.open(target_path)
            target = np.expand_dims(np.array(target), axis=0)


        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class model:
    def __init__(self):
        self.network = Network()
        self.training_data = torchvision.datasets.ImageFolder('./DRIVE/')
        # self.testing_data = 

if __name__ == '__main__':
    dataset = UNetDataset('./DRIVE/testing', trans)
    print(dataset[0])