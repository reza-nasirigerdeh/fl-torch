"""
    Copyright 2023 Reza NasiriGerdeh. All Rights Reserved.

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

import torch
import copy
import logging
import numpy as np

from utils.utils import evaluate
from utils.dataset import load_federated_datasets
from utils.loss_function import get_loss_function

logger = logging.getLogger("base_fed_approach")


class BaseFedApproach:
    def __init__(self, client_class, dataset_config, train_config, model_config, loss_func_config, optimizer_config):

        # Dataset configuration
        dataset_name = dataset_config['name']
        resize_shape_train = dataset_config['resize_shape_train']
        resize_shape_test = dataset_config['resize_shape_test']
        hflip = dataset_config['hflip']
        crop_shape = dataset_config['crop_shape']
        crop_padding = dataset_config['crop_padding']
        resized_crop_shape = dataset_config['resized_crop_shape']
        center_crop_shape = dataset_config['center_crop_shape']
        norm_mean = dataset_config['norm_mean']
        norm_std = dataset_config['norm_std']
        num_clients = dataset_config['num_clients']
        num_classes_per_client = dataset_config['num_classes_per_client']
        balance_degree = dataset_config['balance_degree']
        run_number = dataset_config['run_number']

        # train datasets of the clients loaded from a file or created from scratch depending on the run_number
        fed_train_datasets, test_dataset, num_classes = \
            load_federated_datasets(dataset_name=dataset_name, num_clients=num_clients, num_classes_per_client=num_classes_per_client,
                                    balance_degree=balance_degree, run_number=run_number, hflip=hflip,
                                    resize_shape_train=resize_shape_train, resize_shape_test=resize_shape_test, crop_shape=crop_shape,
                                    crop_padding=crop_padding, resized_crop_shape=resized_crop_shape, center_crop_shape=center_crop_shape,
                                    norm_mean=norm_mean, norm_std=norm_std)

        self.test_dataset = test_dataset
        num_workers = train_config['num_workers']
        self.test_loader = torch.utils.data.DataLoader(dataset=dataset.test_set, batch_size=100, shuffle=False,
                                                       num_workers=num_workers)
        self.test_size = len(dataset.test_set)

        # extend model config
        model_config['num_classes'] = num_classes

        # initial clients
        logger.info(f"Building clients' {model_config['name']} models ...")
        self.clients = []
        for client_train_data in fed_train_datasets:
            client = client_class(train_dataset=client_train_data, train_config=train_config, model_config=model_config,
                                  loss_func_config=loss_func_config, optimizer_config=optimizer_config)
            self.clients.append(client)

        print()
        print(self.clients[0].model)
        print()

        self.selection_rate = train_config['selection_rate']
        self.learning_rate = optimizer_config['learning_rate']

        # initialize global model
        self.global_model = copy.deepcopy(self.clients[0].model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model = self.global_model.to(self.device)

        # initialize loss function used for evaluating the global model
        loss_function_name = loss_func_config['name']
        self.loss_function = get_loss_function(loss_function_name)

    def update_global_model(self, global_state_dict):
        self.global_model.load_state_dict(global_state_dict)
        self.global_model = self.global_model.to(self.device)

    def evaluate_model(self):
        return evaluate(self.global_model, self.test_loader, self.test_size, self.loss_function, self.device)

    def select_clients(self):
        num_selected_clients = int(max(self.selection_rate * len(self.clients), 1))
        selected_client_indexes = np.random.choice(a=len(self.clients), size=num_selected_clients, replace=False)
        selected_clients = [self.clients[index] for index in selected_client_indexes]

        logger.debug(f'Selected clients: {selected_client_indexes}')
        return selected_clients
