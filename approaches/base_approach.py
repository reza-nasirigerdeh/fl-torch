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
import logging

from torch.utils.data import DataLoader
from utils.model import build_model
from utils.loss_function import get_loss_function
from utils.optimizer import get_optimizer

logger = logging.getLogger("base_approach")


class BaseApproach:
    def __init__(self, train_dataset, train_config, model_config, loss_func_config, optimizer_config):

        # training configuration
        self.batch_size = train_config['batch_size']

        # dataset configuration
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

        # device setting
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #  model configuration
        model_name = model_config['name']
        num_classes = model_config['num_classes']
        group_size = model_config['group_size']

        self.model = build_model(model_name=model_name, num_classes=num_classes, group_size=group_size)
        self.model = self.model.to(self.device)

        # loss function configuration
        self.loss_function_name = loss_func_config['name']
        self.loss_function = get_loss_function(self.loss_function_name)

        # optimizer configuration
        self.optimizer_name = optimizer_config['name']
        self.learning_rate = optimizer_config['learning_rate']
        self.momentum = optimizer_config['momentum']
        self.weight_decay = optimizer_config['weight_decay']
        self.dampening = optimizer_config['dampening']
        self.nesterov = optimizer_config['nesterov']

        self.optimizer = get_optimizer(optimizer_name=self.optimizer_name,
                                       model_params=self.model.parameters(),
                                       learning_rate=self.learning_rate,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay,
                                       dampening=self.dampening,
                                       nesterov=self.nesterov)

    def reinit_optimizer(self):
        self.optimizer = get_optimizer(optimizer_name=self.optimizer_name,
                                       model_params=self.model.parameters(),
                                       learning_rate=self.learning_rate,
                                       momentum=self.momentum,
                                       weight_decay=self.weight_decay,
                                       dampening=self.dampening,
                                       nesterov=self.nesterov)

    def train_on_batch(self, image_batch, label_batch):
        image_batch = image_batch.to(self.device)
        label_batch = label_batch.to(self.device)
        self.model = self.model.to(self.device)

        self.model.train()

        output_batch = self.model(image_batch)
        loss = self.loss_function(output_batch, label_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        _, predicted_labels = output_batch.max(1)
        num_correct_predictions = predicted_labels.eq(label_batch).sum().item()

        train_loss = loss.item()
        train_accuracy = num_correct_predictions / label_batch.size(0)

        return train_loss, train_accuracy

    def set_model_params(self, parameters):
        for model_param, param in zip(self.model.parameters(), parameters):
            model_param.data = param.data
        self.model = self.model.to(self.device)
