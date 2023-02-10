"""
    Copyright 2023 Reza NasiriGerdeh and Javad TorkzadehMahani. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import wget
import tarfile
import os
import logging
import torch

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset


logger = logging.getLogger("dataset")

dataset_root = 'datasets'


def _load_centralized_dataset(dataset_name):
    logger.info(f"Loading dataset {dataset_name} ...")

    # create dataset_root folder, if it does not already exist
    if not os.path.exists(dataset_root):
        os.mkdir(dataset_root)

    if dataset_name == 'mnist':
        train_data = datasets.MNIST(root=dataset_root, download=True, train=True)
        test_data = datasets.MNIST(root=dataset_root, download=False, train=False)

    elif dataset_name == 'fashion_mnist':
        train_data = datasets.FashionMNIST(root=dataset_root, download=True, train=True)
        test_data = datasets.FashionMNIST(root=dataset_root, download=False, train=False)

    elif dataset_name == 'cifar10':
        train_data = datasets.CIFAR10(root=dataset_root, download=True, train=True)
        test_data = datasets.CIFAR10(root=dataset_root, download=False, train=False)

    elif dataset_name == 'cifar10_subset':
        train_data = datasets.ImageFolder(root=f'{dataset_root}/cifar10-subset/train')
        test_data = datasets.ImageFolder(root=f'{dataset_root}/cifar10-subset/test')

    elif dataset_name == 'cifar100':
        train_data = datasets.CIFAR100(root=dataset_root, download=True, train=True)
        test_data = datasets.CIFAR100(root=dataset_root, download=False, train=False)

    elif dataset_name == 'imagenette':
        train_data = datasets.ImageFolder(root=f'{dataset_root}/imagenette/train')
        test_data = datasets.ImageFolder(root=f'{dataset_root}/imagenette/val')

    elif dataset_name == 'imagenette_160px':
        # if imagenette-160px has not already been downloaded
        if not os.path.exists(f'{dataset_root}/imagenette2-160'):
            # download imagenette-160px dataset
            logger.info("Downloading the dataset ...")
            file_path = wget.download(url='https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz',
                                      out=f'{dataset_root}')

            # extract tgz file
            logger.info("Extracting the dataset ...")
            tar = tarfile.open(name=file_path, mode="r:gz")
            tar.extractall(path=f'{dataset_root}')
            tar.close()

        train_data = datasets.ImageFolder(root=f'{dataset_root}/imagenette2-160/train')
        test_data = datasets.ImageFolder(root=f'{dataset_root}/imagenette2-160/val')

    elif dataset_name == 'imagenette_subset':
        train_data = datasets.ImageFolder(root=f'{dataset_root}/imagenette-subset/train')
        test_data = datasets.ImageFolder(root=f'{dataset_root}/imagenette-subset/val')

    elif dataset_name == 'imagenette_320px':
        if not os.path.exists(f'{dataset_root}/imagenette2-320'):
            # download imagenette-320px dataset
            logger.info("Downloading the dataset ...")
            file_path = wget.download(url='https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz',
                                      out=f'{dataset_root}')

            # extract tgz file
            logger.info("Extracting the dataset ...")
            tar = tarfile.open(name=file_path, mode="r:gz")
            tar.extractall(path=f'{dataset_root}')
            tar.close()

        train_data = datasets.ImageFolder(root=f'{dataset_root}/imagenette2-320/train')
        test_data = datasets.ImageFolder(root=f'{dataset_root}/imagenette2-320/val')

    elif dataset_name == 'imagewoof_160px':
        # if imagewoof-160px has not already been downloaded
        if not os.path.exists(f'{dataset_root}/imagewoof2-160'):
            # download imagenette-160px dataset
            logger.info("Downloading the dataset ...")
            file_path = wget.download(url='https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz',
                                      out=f'{dataset_root}')

            # extract tgz file
            logger.info("Extracting the dataset ...")
            tar = tarfile.open(name=file_path, mode="r:gz")
            tar.extractall(path=f'{dataset_root}')
            tar.close()

        train_data = datasets.ImageFolder(root=f'{dataset_root}/imagewoof2-160/train')
        test_data = datasets.ImageFolder(root=f'{dataset_root}/imagewoof2-160/val')

    elif dataset_name == 'imagewoof_320px':
        # if imagewoof-320px has not already been downloaded
        if not os.path.exists(f'{dataset_root}/imagewoof2-320'):
            # download imagenette-320px dataset
            logger.info("Downloading the dataset ...")
            file_path = wget.download(url='https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz',
                                      out=f'{dataset_root}')

            # extract tgz file
            logger.info("Extracting the dataset ...")
            tar = tarfile.open(name=file_path, mode="r:gz")
            tar.extractall(path=f'{dataset_root}')
            tar.close()

        train_data = datasets.ImageFolder(root=f'{dataset_root}/imagewoof2-320/train')
        test_data = datasets.ImageFolder(root=f'{dataset_root}/imagewoof2-320/val')

    else:
        logger.error(f"No implementation is available for {dataset_name}!")
        logger.error("You can add the corresponding implementation to load_image_dataset in utils/dataset.py")
        logger.error("Or, you can use --dataset mnist|fashion_mnist|cifar10|cifar100|imagenette_160px|imagenette_320px")
        exit()

    num_classes = len(np.unique([train_data[index][1] for index in range(len(train_data))]))

    return train_data, test_data, num_classes


def _split_train_set(train_set, num_clients, num_classes_per_client, num_classes, balance_degree):
    # sanity check to ensure data split across clients is possible for the given num_classes_per_client
    if (num_clients * num_classes_per_client) % num_classes != 0:
        logger.error("Data split across clients is not possible!")
        logger.error("The following condition is not satisfied:")
        logger.error("(num_clients * num_classes_per_client) mod num_classes == 0")
        exit()

    size_train = len(train_set.targets)

    # group samples into shards
    num_shards_per_class = (num_clients * num_classes_per_client) // num_classes

    indices_grouped = [[] for _ in range(num_classes)]
    for index in range(size_train):
        indices_grouped[train_set[index][1]].append(index)

    shards = []
    if balance_degree == 1.0:
        for class_indices in indices_grouped:
            np.random.shuffle(class_indices)
            shards.append(np.array_split(class_indices, num_shards_per_class))
    else:
        for class_indices in indices_grouped:
            epsilon_class_shards = [(class_indices.pop(0)) for _ in range(num_shards_per_class)]
            dirichlet_weights = np.random.dirichlet(balance_degree * (np.ones(num_shards_per_class)), size=1)
            dirichlet_boundaries = np.floor((dirichlet_weights * len(class_indices))).flatten().astype(int)
            class_shards = np.array_split(class_indices, np.cumsum(dirichlet_boundaries[0:-1]))
            shards.append([np.append(class_shards[k], epsilon_class_shards[k]) for k in range(num_shards_per_class)])

    # organize shards as 0,1,2,...9,0,1,2,...,9,... so that no client will have shards with the same class
    shards_temp = shards[:]
    shards = []
    for shard_index_in_class in range(num_shards_per_class):
        for class_index in np.arange(num_classes):
            shards.append(shards_temp[class_index][shard_index_in_class])

    # a range of shards that can be shuffled in a way that no client will have shards with the same class
    range_to_shuffle = (num_classes // num_classes_per_client) * num_classes_per_client
    shuffled_shard_numbers = []

    shard_numbers = []
    for shard_num in np.arange(0, len(shards)):
        shard_numbers.append(shard_num)
        if len(shard_numbers) == range_to_shuffle:
            np.random.shuffle(shard_numbers)
            shuffled_shard_numbers.extend(shard_numbers)
            shard_numbers = []

    np.random.shuffle(shard_numbers)
    shuffled_shard_numbers.extend(shard_numbers)

    shards_temp = shards[:]
    shards = []
    for shard_num in shuffled_shard_numbers:
        shards.append(shards_temp[shard_num])

    # create federated train sets of clients
    fed_train_sets = []
    for client_index in range(num_clients):
        indices = []
        for _ in range(num_classes_per_client):
            indices.extend(shards.pop(0))

        client_train_set = []
        for index in indices:
            client_train_set.append([train_set[index][0], train_set[index][1]])

        fed_train_sets.append(client_train_set)

    # sanity check to make sure each client has samples from exactly num_classes_per_client labels
    for client_train_set in fed_train_sets:
        client_labels = []
        for _, label in client_train_set:
            client_labels.append(label)
        if len(np.unique(client_labels)) != num_classes_per_client:
            logger.error(f"A client has NOT exactly {num_classes_per_client} labels!")
            exit()

    return fed_train_sets


class _FedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index][0], self.dataset[index][1]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)


def load_federated_datasets(dataset_name, num_clients, num_classes_per_client, balance_degree,
                            run_number, resize_shape_train, resize_shape_test, hflip,
                            crop_shape, crop_padding, resized_crop_shape, center_crop_shape,
                            norm_mean, norm_std):

    # local function to get the train/test transform object
    def _get_transform(train=True):
        trans = []

        # random horizontal flip, random_cropping, and random_resized_cropping is only applied to train images
        if train:
            if resize_shape_train:
                trans.append(transforms.Resize(size=resize_shape_train))

            if resized_crop_shape:
                trans.append(transforms.RandomResizedCrop(size=resized_crop_shape))

            if crop_shape:
                padding = crop_padding if crop_padding else None
                trans.append(transforms.RandomCrop(size=crop_shape, padding=padding))

            if hflip:
                trans.append(transforms.RandomHorizontalFlip())
        else:
            if resize_shape_test:
                trans.append(transforms.Resize(size=resize_shape_test))

            if center_crop_shape:
                trans.append(transforms.CenterCrop(size=center_crop_shape))

        trans.append(transforms.ToTensor())

        if norm_mean and norm_std:
            trans.append(transforms.Normalize(mean=norm_mean, std=norm_std))

        return transforms.Compose(trans)

    cent_train_set, cent_test_set, num_classes = _load_centralized_dataset(dataset_name=dataset_name)
    train_transform = _get_transform(train=True)
    test_transform = _get_transform(train=False)

    # load train data of clients from the file if a positive run_number is given,
    # and the corresponding file already exists
    if run_number > 0:
        fed_train_set_file_name = f'./{dataset_root}/{dataset_name}-K{num_clients}-L{num_classes_per_client}-B{balance_degree}-R{run_number}.pt'

        if os.path.exists(fed_train_set_file_name):
            logger.info(f"Loading federated train datasets from {fed_train_set_file_name} ...")
            fed_train_sets = torch.load(fed_train_set_file_name)

            fed_train_datasets = [_FedDataset(dataset=client_train_set, transform=train_transform) for client_train_set in fed_train_sets]
            test_dataset = _FedDataset(dataset=cent_test_set, transform=test_transform)
            return fed_train_datasets, test_dataset, num_classes

    # split train dataset into num_clients partitions
    logger.info(f"Creating the local datasets of clients ... ")
    fed_train_sets = _split_train_set(train_set=cent_train_set, num_clients=num_clients,
                                      num_classes_per_client=num_classes_per_client, num_classes=num_classes,
                                      balance_degree=balance_degree)

    # save federated train sets for future runs if a positive run_number provided
    if run_number > 0:
        fed_train_set_file_name = f'./{dataset_root}/{dataset_name}-K{num_clients}-L{num_classes_per_client}-B{balance_degree}-R{run_number}.pt'
        logger.info(f"Saving the local datasets of clients to {fed_train_set_file_name} ... ")
        torch.save(fed_train_sets, f'{fed_train_set_file_name}')

    fed_train_datasets = [_FedDataset(dataset=client_train_set, transform=train_transform) for client_train_set in fed_train_sets]
    test_dataset = _FedDataset(dataset=cent_test_set, transform=test_transform)
    return fed_train_datasets, test_dataset, num_classes

