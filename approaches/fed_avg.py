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

import logging

import torch
from torch.utils.data import DataLoader

from approaches.base_approach import BaseApproach
from approaches.base_fed_approach import BaseFedApproach
from utils.utils import aggregate_params

logger = logging.getLogger("fed_avg")


class FedAvgClient(BaseApproach):
    def __init__(self, train_dataset, train_config, model_config, loss_func_config, optimizer_config):
        super(FedAvgClient, self).__init__(train_dataset=train_dataset, train_config=train_config, model_config=model_config,
                                           loss_func_config=loss_func_config, optimizer_config=optimizer_config)

        self.num_local_epochs = train_config['num_local_epochs']
        self.sample_size = len(train_dataset)
        self.acc_grads = None  # accumulated gradients, which is reinitialized and updated in compute_client_grads
        self.old_grads = None  # used if momentum is non-zero, which is reinitialized and updated in compute_client_grads

    def compute_client_grads(self, global_model_params, learning_rate):

        # copy the global params to the client's model params
        self.set_model_params(global_model_params)

        # accumulated grads of clients are initially zero
        self.acc_grads = {name: 0.0 for name, _ in self.model.named_parameters()}

        self.old_grads = {name: 0.0 for name, _ in self.model.named_parameters()}

        for local_epoch in range(self.num_local_epochs):
            logger.debug(f"Local epoch #{local_epoch} ...")
            for image_batch, label_batch in self.train_loader:
                image_batch = image_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                self.model = self.model.to(self.device)

                self.model.train()

                output_batch = self.model(image_batch)
                loss = self.loss_function(output_batch, label_batch)

                self.optimizer.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for param_name, param_value in self.model.named_parameters():

                        # compute final grads based on momentum
                        final_grad = self.momentum * self.old_grads[param_name] + param_value.grad

                        # accumulate gradients
                        self.acc_grads[param_name] += final_grad

                        # update client model
                        param_value -= learning_rate * final_grad + learning_rate * self.weight_decay * param_value
                        self.old_grads[param_name] = final_grad

        train_loss = -1.0
        train_accuracy = -1.0
        
        return train_loss, train_accuracy, self.sample_size


class FedAvgApproach(BaseFedApproach):
    def __init__(self, dataset_config, train_config, model_config, loss_func_config, optimizer_config):
        super(FedAvgApproach, self).__init__(client_class=FedAvgClient, dataset_config=dataset_config,
                                             train_config=train_config, model_config=model_config,
                                             loss_func_config=loss_func_config, optimizer_config=optimizer_config)
        self.learning_rate = optimizer_config['learning_rate']
        self.old_grads = {name: 0.0 for name, parameter in self.global_model.named_parameters()}

    def train_model(self):

        global_train_loss = 0.0
        global_correct_predictions = 0
        global_train_size = 0

        # randomly select clients
        selected_clients = self.select_clients()

        # instrument the selected clients to train global model on their local data
        # compute train global train loss and train accuracy
        for client in selected_clients:
            local_train_loss, local_train_accuracy, local_train_size = client.compute_client_grads(global_model_params=self.global_model.parameters(),
                                                                                                   learning_rate=self.learning_rate)
            global_train_loss += local_train_loss * local_train_size
            global_correct_predictions += local_train_accuracy * local_train_size
            global_train_size += local_train_size

        global_train_loss = global_train_loss / global_train_size
        global_train_accuracy = global_correct_predictions / global_train_size

        # aggregate clients' parameters' values to compute the global parameters' values
        # FedLMB employs weighted averaging, where number of train samples in the clients indicate the weight of the client during aggregation
        weights = [client.sample_size for client in selected_clients]

        local_state_dicts = list()
        for client in selected_clients:
            client_grads = {name: client.acc_grads[name] for name in client.acc_grads}
            local_state_dicts.append(client_grads)

        global_grad_dict = aggregate_params(weights=weights, params=local_state_dicts)

        # update the global model
        with torch.no_grad():
            for param_name, param_value in self.global_model.named_parameters():
                final_grad = global_grad_dict[param_name]
                param_value -= self.learning_rate * final_grad
                self.old_grads[param_name] = final_grad

        # self.global_model = self.global_model.to(self.device, dtype=torch.float64)
        self.global_model = self.global_model.to(self.device)

        return global_train_loss, global_train_accuracy
