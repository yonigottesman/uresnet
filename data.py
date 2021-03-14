import random
from pathlib import Path

import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class GreyDataset(Dataset):
    def __init__(self, path, img_size, aug=False):
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        self.damaged_transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.files = [i for i in Path(path).glob('**/*') if i.is_file()]
        self.aug = aug

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        with open(self.files[idx], 'rb') as f:
            orig_image = Image.open(f).convert('RGB')

        image = self.transform(orig_image)
        gray = self.damaged_transforms(orig_image)

        if self.aug and random.random() > 0.5:
            image = TF.hflip(image)
            gray = TF.hflip(gray)

        return gray, image


def gen_datasets(img_size=(256, 256)):
    train_path = '/home/yonatang/ephemeral_drive/train2014/'
    val_path = '/home/yonatang/ephemeral_drive/val2014/'
    train_dataset = GreyDataset(train_path, img_size, True)
    val_dataset = GreyDataset(val_path, img_size)
    return train_dataset, val_dataset


def denormalize(img, mean, std):
    # not accurate but its only for displaying images
    img = img.permute(1, 2, 0)
    img = img * torch.tensor(std) + torch.tensor(mean)
    img = img.permute(2, 0, 1)
    return img


def get_dataloader(img_size=(256, 256), bs=8):
    train_dataset, val_dataset = gen_datasets(img_size)
    train_dl = DataLoader(train_dataset,
                          batch_size=bs,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True)

    val_dl = DataLoader(val_dataset,
                        batch_size=bs,
                        num_workers=4,
                        pin_memory=True,
                        shuffle=False)
    return train_dl, val_dl
