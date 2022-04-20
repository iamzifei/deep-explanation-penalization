import json
import os
import torch
import pandas as pd
from os import listdir
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys

sys.path.append('/Users/s720181/Dev/master/deep-explanation-penalization/src')
import utils

with open('/Users/s720181/Dev/master/deep-explanation-penalization/isic-skin-cancer/ISIC/config.json') as json_file:
    data = json.load(json_file)

data_path = data["data_folder"]
processed_path = os.path.join(data_path, "processed")
benign_path = os.path.join(processed_path, "0_no_cancer")
malignant_path = os.path.join(processed_path, "1_cancer")


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return img


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


class CustomDataSet(Dataset):
    def __init__(self, root_dirs, csv_file, transform=None):
        self.root_dirs = root_dirs
        self.csv_file = pd.read_csv(csv_file, keep_default_na=False)
        self.transform = transform
        image_list = []
        for root_dir in self.root_dirs:
            if not os.path.isdir(root_dir):
                raise Exception("Root directory does not exist")
            for f in list(listdir(root_dir)):
                if f.endswith(".jpg"):
                    image_list.append(f)
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = None
        label = None
        for root_dir in self.root_dirs:
            img_name = os.path.join(root_dir,
                                    self.image_list[idx])
            if os.path.isfile(img_name):
                fileName = self.image_list[idx].split(".")[0]
                cell = self.csv_file[self.csv_file['isic_id']
                                     == fileName]['benign_malignant'].values
                if len(cell) > 0:
                    benigin_malignant = cell[0]
                    if benigin_malignant == 'benign':
                        label = 0
                    elif benigin_malignant == 'malignant':
                        label = 1
                    else:
                        label = 2
                image = io.imread(img_name)
                if self.transform:
                    image = self.transform(image)
                break

        return image, label


def loadImageDataset(root_dirs, csv_file):
    complete_dataset = CustomDataSet(
        root_dirs=root_dirs,
        csv_file=csv_file,
        transform=transforms.Compose([
            Rescale(224),
            # RandomCrop(224),
            ToTensor()
        ])
    )

    num_total = len(complete_dataset)
    num_train = int(0.8 * num_total)
    num_val = int(0.1 * num_total)
    num_test = num_total - num_train - num_val
    torch.manual_seed(0)  # reproducible splitting
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        complete_dataset, [num_train, num_test, num_val])

    print("Total train: ", len(train_dataset))
    print("Total test: ", len(test_dataset))
    print("Total validation: ", len(val_dataset))
    datasets = {'train': train_dataset,  'val': val_dataset,
                'test': test_dataset}

    cancer_ratio = len(list(listdir(malignant_path)))/num_total

    not_cancer_ratio = 1 - cancer_ratio
    cancer_weight = 1/cancer_ratio
    not_cancer_weight = 1 / not_cancer_ratio
    weights = np.asarray([not_cancer_weight, cancer_weight])
    weights /= weights.sum()

    return datasets, weights


def main():
    imageDataset, imageWeights = loadImageDataset(
        root_dirs=[benign_path, malignant_path],
        csv_file="/Users/s720181/Dev/master/data/ISIC/meta.csv"
    )

    dataset_path = os.path.join(data["data_folder"], "calculated_features")
    datasets, weights = utils.load_precalculated_dataset(dataset_path)

    print(imageWeights, weights)

    print(len(imageDataset["train"]), len(imageDataset["val"]), len(imageDataset["test"]))

    print(len(datasets["train"]), len(datasets["val"]), len(datasets["test"]))

    for i in range(len(imageDataset["train"])):
        sample, label = imageDataset["train"][i]

        print(i, sample.size(), label)

        if i == 3:
            break
    print(datasets["train"][0][0].size(), datasets["train"][0][1].size(), datasets["train"][0][2].size())
    # for i in range(len(datasets["train"])):
    #     sample, label = datasets["train"][i]

    #     print(i, sample.size(), label)

    #     if i == 3:
    #         break


if __name__ == '__main__':
    main()
