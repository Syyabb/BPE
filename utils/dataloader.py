import torch.utils.data as data
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import os
import csv
import kornia.augmentation as A
import random
import numpy as np

from PIL import Image
from torch.utils.tensorboard import SummaryWriter


class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(torch.nn.Module):
    def __init__(self, opt):
        super(PostTensorTransform, self).__init__()
        self.random_crop = ProbTransform(
            A.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop), p=0.8
        )
        self.random_rotation = ProbTransform(A.RandomRotation(opt.random_rotation), p=0.5)
        if opt.dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def get_transform(opt, train=True, pretensor_transform=False):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if pretensor_transform:
        if train:
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=opt.random_crop))
            transforms_list.append(transforms.RandomRotation(opt.random_rotation))
            if opt.dataset == "cifar10":
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transforms_list.append(transforms.ToTensor())
    # if opt.dataset == "cifar10":
    #     transforms_list.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]))
    # elif opt.dataset == "mnist":
    #     transforms_list.append(transforms.Normalize([0.5], [0.5]))
    # elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
    #     pass
    # else:
    #     raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)





class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(opt.data_root, "GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label


class CelebA_attr(data.Dataset):
    def __init__(self, opt, split, transforms):
        self.dataset = torchvision.datasets.CelebA(root=opt.data_root, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)


'''
class Customer_cifar10(Dataset):
    def __init__(self, full_dataset, transform):
        self.dataset = self.addTrigger(full_dataset)
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, remove_label=6):
        # dataset
        dataset_ = list()

        cnt = 0
        for i in range(len(dataset)):
            data = dataset[i]

            if (data[1] == remove_label):
                continue

            if data[1] < remove_label:
                dataset_.append(data)
                cnt += 1
            else:
                dataset_.append((data[0], data[1] - 1))
                cnt += 1
        return dataset_
'''

def get_dataloader(opt, train=True, pretensor_transform=False):
    transform = get_transform(opt, train, pretensor_transform)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(opt, split, transform)
    else:
        raise Exception("Invalid dataset")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    return dataloader

'''
def get_dataloader_customer_cifar10(opt, train=True, pretensor_transform=False):
    transform = get_transform(opt, train, pretensor_transform)
    dataset = torchvision.datasets.CIFAR10(opt.data_root, train, download=True)
    dataset = Customer_cifar10(dataset, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=True)
    return dataloader
'''


def get_dataset(opt, train=True):
    if opt.dataset == "gtsrb":
        dataset = GTSRB(
            opt,
            train,
            transforms=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform=ToNumpy(), download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform=ToNumpy(), download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(
            opt,
            split,
            transforms=transforms.Compose([transforms.Resize((opt.input_height, opt.input_width)), ToNumpy()]),
        )

    else:
        raise Exception("Invalid dataset")
    return dataset

class Customer_dataset(Dataset):
    def __init__(self, opt, full_dataset, transform, trigger_index, remove=1000):

        if opt.trigger_type == "blend":
            self.blend = Image.open('./hellokity/hellokity.png')
            self.blend = np.array(self.blend.resize((32, 32)))
        if opt.trigger_type == "sig":
            self.generate()
        self.opt = opt

        self.full_dataset = self.addTrigger(full_dataset, trigger_index)
        self.dataset = self.full_dataset
        self.transform = transform
        if remove < opt.num_classes:
            self.removelabel(self.dataset, remove)


    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, trigger_index):
        # dataset
        dataset_ = list()
        for i in range(len(dataset)):
            img, label = dataset[i]
            img = np.array(img)
            if i in trigger_index:
                if self.opt.trigger_type == 'sig':
                    img = self._sigTrigger(img)
                elif self.opt.trigger_type == 'blend':
                    img = self._blendTrigger(img)
                elif self.opt.trigger_type == 'patch':
                    img = self._patchTrigger(img, self.opt.input_width, self.opt.input_height)
                if self.opt.attack_mode == 'all2all':
                    label = (label + 1) % self.opt.num_classes
                if self.opt.attack_mode == 'all2one':
                    label = self.opt.target_label

            dataset_.append((img, label))
        return dataset_

    def _blendTrigger(self, img):
        alpha = 0.1
        blend_img = (1 - alpha) * img + alpha * self.blend
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)
        return blend_img

    def _patchTrigger(self, img, width=32, height=32):
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0
        return img

    def _sigTrigger(self, img):
        img = np.clip((img+self.sig), a_min=0, a_max=255)
        img = img.astype('uint8')
        return img

    def generate(self):
        delta = 20
        f = 6
        blend_img = np.ones((32, 32, 3))
        m = blend_img.shape[1]
        for i in range(blend_img.shape[0]):
            for j in range(blend_img.shape[1]):
                blend_img[i, j] = delta * np.sin(2 * np.pi * j * f / m)
        self.sig = blend_img

    #True indicates filtering out, and False indicates retention
    def filter(self, filter_index):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            img = np.array(img)
            if filter_index[i]:
                continue
            dataset_.append((img, label))
        self.dataset = dataset_

    def removelabel(self, dataset, remove_label=6):
        # dataset
        dataset_ = list()
        cnt = 0
        for i in range(len(dataset)):
            data = dataset[i]

            if (data[1] == remove_label):
                continue

            if data[1] < remove_label:
                dataset_.append(data)
                cnt += 1
            else:
                dataset_.append((data[0], data[1] - 1))
                cnt += 1
        return dataset_

class Customer_dataset_warp(Dataset):
    def __init__(self, full_dataset, transform, opt, noise_grid=None, identity_grid=None, train=True):
        self.opt = opt
        self.transform = transform
        self.train = train
        if train:
            self.noise_grid = noise_grid
            self.identity_grid = identity_grid
            self.grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            self.grid_temps = torch.clamp(self.grid_temps, -1, 1)

        self.full_dataset = self.addTrigger(full_dataset)
        self.dataset = self.full_dataset

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset):
        # dataset
        dataset_ = list()
        num_bd = self.opt.pc * len(dataset)
        with torch.no_grad():
            for i in range(len(dataset)):
                img = dataset[i][0]
                label = dataset[i][1]
                img = self.transform(img)

                if i < num_bd and self.train:
                    # dataset_.append(data)
                    img = F.grid_sample(img.unsqueeze(dim=0).to(self.opt.device), self.grid_temps.repeat(1, 1, 1, 1), align_corners=True)[0].cpu()
                    if self.opt.attack_mode == "all2one":
                        dataset_.append((img, self.opt.target_label))
                    elif self.opt.attack_mode == "all2all":
                        dataset_.append((img, (label + 1) % self.opt.num_classes))
                elif i < num_bd*3 and self.train:
                    ins = torch.rand(1, self.opt.input_height, self.opt.input_height, 2).to(self.opt.device) * 2 - 1
                    grid_temps2 = self.grid_temps.repeat(1, 1, 1, 1) + ins / self.opt.input_height
                    grid_temps2 = torch.clamp(grid_temps2, -1, 1)
                    img = F.grid_sample(img.unsqueeze(dim=0).to(self.opt.device), grid_temps2.repeat(1, 1, 1, 1), align_corners=True)[0].cpu()
                    dataset_.append((img, label))
                else:
                    dataset_.append((img, label))
        return dataset_

    def filter(self, filter_index):
        dataset_ = list()
        for i in range(len(self.full_dataset)):
            img, label = self.full_dataset[i]
            img = np.array(img)
            if filter_index[i]:
                continue
            dataset_.append((img, label))
        self.dataset = dataset_