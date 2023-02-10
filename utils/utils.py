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

import os
import torch
import copy
import numpy as np

import logging
logger = logging.getLogger("utils")


def evaluate(model, test_loader, test_size, loss_function, device):

    # initialize test data loader
    model = model.to(device)
    model.eval()

    num_correct_predictions = 0
    loss_total = 0.0
    with torch.no_grad():
        for image_batch, label_batch in test_loader:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            output_batch = model(image_batch)
            loss = loss_function(output_batch, label_batch)
            _, predicted_labels = output_batch.max(1)
            num_correct_predictions += predicted_labels.eq(label_batch).sum().item()
            loss_total += loss.item() * label_batch.size(0)

        test_loss = loss_total / test_size
        test_accuracy = num_correct_predictions / test_size

    return test_loss, test_accuracy


class ResultFile:
    def __init__(self, result_file_name):
        # create root directory for results
        result_root = './results'
        if not os.path.exists(result_root):
            os.mkdir(result_root)

        # open result file
        self.result_file = open(file=f'{result_root}/{result_file_name}', mode='w')

    def write_header(self, header):
        self.result_file.write(f'{header}\n')
        self.result_file.flush()

    def write_result(self, comm_round, result_list):
        digits_precision = 8

        result_str = f'{comm_round},'
        for result in result_list:
            if result != '-':
                result = np.round(result, digits_precision)
            result_str += f'{result},'

        # remove final comma
        result_str = result_str[0:-1]

        self.result_file.write(f'{result_str}\n')
        self.result_file.flush()

    def close(self):
        self.result_file.close()


def aggregate_params(weights, params):
    # first client
    global_state_dict = {name: params[0][name] * weights[0] for name in params[0].keys()}

    for client_index in range(1, len(params)):
        client_state_dict = params[client_index]
        for param_name in global_state_dict.keys():
            global_state_dict[param_name] += client_state_dict[param_name] * weights[client_index]

    total_weight = np.sum(weights)
    for param_name in global_state_dict.keys():
        # in batch norm layer, num_batches_tracked is long integer type
        if 'num_batches_tracked' in param_name:
            global_state_dict[param_name] = (global_state_dict[param_name] / total_weight).to(torch.long)
            continue
        global_state_dict[param_name] /= total_weight

    return global_state_dict
