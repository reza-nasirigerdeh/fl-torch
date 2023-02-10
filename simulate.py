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
import argparse
import numpy as np
import random


from utils.utils import ResultFile
from approaches.fed_avg import FedAvgApproach

logger = logging.getLogger("simulate")


def main():

    # ############ Simulation parameters ##################################
    parser = argparse.ArgumentParser(description="Simulate mini-batch gradient descent training",
                                     usage=f"python simulate.py "
                                           f"[--approach approach_name]"
                                           f"[--dataset dataset_name] "
                                           f"[--resize-train resized_shape_of_train_images] "
                                           f"[--resize-test resized_shape_of_test_images] "
                                           f"[--random-crop random_crop_shape_and_padding] "
                                           f"[--random-resized-crop crop_shape] "
                                           f"[--center-crop crop_shape] "
                                           f"[--random-hflip] "
                                           f"[--norm-mean norm_mean] "
                                           f"[--norm-std norm_std] "
                                           f"[--model model_name] "
                                           f"[--loss loss_function_name] "
                                           f"[--optimizer optimizer_name] "
                                           f"[--batch-size batch_size] "
                                           f"[--learning-rate learning_rate] "
                                           f"[--momentum momentum_of_SGD_optimizer] "
                                           f"[--weight-decay weight_decay_of_SGD ] "
                                           f"[--dampening dampening_value] "
                                           f"[--nesterov nesterov_flag] "
                                           f"[--clients num_clients] "
                                           f"[--selection-rate selection_rate_of_clients] "
                                           f"[--classes-per-client num_classes_per_client] "
                                           f"[--balance-degree balance_degree] "
                                           f"[--local-epochs num_local_epochs] "
                                           f"[--rounds max_communication_rounds] "
                                           f"[--run-number run_number] "
                                           f"[--log-level {logging.getLevelName(logging.DEBUG)}|"
                                           f"{logging.getLevelName(logging.INFO)}|"
                                           f"{logging.getLevelName(logging.WARNING)}|"
                                           f"{logging.getLevelName(logging.ERROR)}|"
                                           f"{logging.getLevelName(logging.CRITICAL)}] ")

    parser.add_argument("--approach_name", "--approach", type=str, help="approach name", default="fedavg")

    parser.add_argument("--dataset_name", "--dataset", type=str, help="dataset name", default="mnist")

    parser.add_argument("--resize_train", "--resize-train", type=str, help="shape for resizing train images e.g. 32x32", default="")

    parser.add_argument("--resize_test", "--resize-test", type=str, help="shape for resizing test images e.g. 32x32", default="")

    parser.add_argument("--random_crop", "--random-crop", type=str, help="random crop shape and padding e.g. 32x32-4x4", default="")

    parser.add_argument("--random_resized_crop", "--random-resized-crop", type=str, help="crop at a random location, and resize to e.g. 128x128", default="")

    parser.add_argument("--center_crop", "--center-crop", type=str, help="crop at the center with size e.g. 32x32", default="")

    parser.add_argument("--random_hflip", "--random-hflip", action='store_true', help="random horizontal flip with p=0.5", default=False)

    parser.add_argument("--norm_mean", "--norm-mean", type=str,
                        help="comma separated per-channel mean values of the dataset e.g. 0.5071,0.4867,0.4408", default="")

    parser.add_argument("--norm_std", "--norm-std", type=str,
                        help="comma separated per-channel std values of the dataset e.g. 0.2675,0.2565,0.2761", default="")

    parser.add_argument("--model_name", "--model", type=str, help="Model to be trained", default="fnn")

    parser.add_argument("--group_size", "--group-size", type=int, help="group size for GroupNorm", default=32)

    parser.add_argument("--loss_function_name", "--loss", type=str, help="loss function name",
                        default="cross_entropy")

    parser.add_argument("--optimizer_name", "--optimizer", type=str, help="optimizer name", default="sgd")

    parser.add_argument("--learning_rate", "--learning-rate", type=float, help="learning rate", default=0.01)

    parser.add_argument("--decay_epochs", "--decay-epochs", type=str, help="comma separated epoch list in which learning rate decays should be applied", default='50,100')

    parser.add_argument("--decay_multiplier", "--decay-multiplier", type=float, help="learning rate decay multiplier ", default=0.1)

    parser.add_argument("--momentum", "--momentum", type=float, help="momentum of SGD optimizer", default=0.0)

    parser.add_argument("--weight_decay", "--weight-decay", type=float, help="weight decay value", default=0.0)

    parser.add_argument("--dampening", "--dampening", type=float, help="dampening value for SGD optimizer", default=0.0)

    parser.add_argument("--nesterov", "--nesterov", action='store_true', help="nesterov flag for SGD optimizer", default=False)

    parser.add_argument("--batch_size", "--batch-size", type=int, help="batch size", default=128)

    parser.add_argument("--num_clients", "--clients", type=int, help="number of clients", default=10)

    parser.add_argument('--num_classes_per_client', "--classes-per-client", type=int,
                        help="number of classes (labels) per client", default=10)

    parser.add_argument("--selection_rate", "--selection-rate", type=float, help="client selection rate",
                        default=1.0)

    parser.add_argument("--balance_degree", "--balance-degree", type=float, help="alpha parameter of dirichlet",
                        default=1.0)

    parser.add_argument("--num_local_epochs", "--local-epochs", type=int, help="local epochs for FedAvg", default=1)

    parser.add_argument("--num_local_updates", "--local-updates", type=int, help="local updates for FedLMB", default=1)

    parser.add_argument("--run_number", "--run", type=int,
                        help="the run number to have a separate result file for each run ", default=0)

    parser.add_argument("--num_workers", "--workers", type=int, help="number of workers", default=4)

    parser.add_argument("--num_comm_rounds", "--rounds", type=int, help="number of communication rounds", default=100)

    parser.add_argument("--eval_freq", "--eval-freq", type=int, help="evaluate model every eval-freq rounds", default=1)

    parser.add_argument("--log_level", "--log-level", type=str, help="log level",
                        default=logging.getLevelName(logging.INFO),
                        choices=[logging.getLevelName(logging.DEBUG), logging.getLevelName(logging.INFO),
                                 logging.getLevelName(logging.WARNING), logging.getLevelName(logging.ERROR),
                                 logging.getLevelName(logging.CRITICAL)])

    args = parser.parse_args()

    # ############ Simulator configuration ##################################

    # Training config
    selection_rate = args.selection_rate if 0.0 <= args.selection_rate <= 1.0 else 1.0
    train_config = {'approach_name': args.approach_name, 'batch_size': args.batch_size,
                    'num_local_epochs': args.num_local_epochs, 'num_local_updates': args.num_local_updates,
                    'selection_rate': selection_rate, 'num_workers': args.num_workers}

    # ##### Dataset config
    shape_str = args.resize_train.split('x')
    resize_shape_train = (int(shape_str[0]), int(shape_str[1])) if len(shape_str) == 2 else ()

    shape_str = args.resize_test.split('x')
    resize_shape_test = (int(shape_str[0]), int(shape_str[1])) if len(shape_str) == 2 else ()

    if "-" in args.random_crop:
        crop_shape_str = args.random_crop.split('-')[0].split('x')
        crop_shape = (int(crop_shape_str[0]), int(crop_shape_str[1])) if len(crop_shape_str) == 2 else ()

        crop_padding_str = args.random_crop.split('-')[1].split('x')
        crop_padding = (int(crop_padding_str[0]), int(crop_padding_str[1])) if len(crop_padding_str) == 2 else ()
    elif "x" in args.random_crop:
        crop_shape_str = args.random_crop.split('x')
        crop_shape = (int(crop_shape_str[0]), int(crop_shape_str[1])) if len(crop_shape_str) == 2 else ()

        crop_padding = ()
    else:
        crop_shape = ()
        crop_padding = ()

    resized_crop_shape_str = args.random_resized_crop.split('x')
    resized_crop_shape = (int(resized_crop_shape_str[0]), int(resized_crop_shape_str[1])) if len(resized_crop_shape_str) == 2 else ()

    center_crop_shape_str = args.center_crop.split('x')
    center_crop_shape = (int(center_crop_shape_str[0]), int(center_crop_shape_str[1])) if len(center_crop_shape_str) == 2 else ()

    norm_mean_str = args.norm_mean.split(',')
    norm_mean = (float(norm_mean_str[0]), float(norm_mean_str[1]), float(norm_mean_str[2])) if len(norm_mean_str) == 3 else ()

    norm_std_str = args.norm_std.split(',')
    norm_std = (float(norm_std_str[0]), float(norm_std_str[1]), float(norm_std_str[2])) if len(norm_std_str) == 3 else ()

    dataset_config = {'name': args.dataset_name, 'resize_shape_train': resize_shape_train, 'resize_shape_test': resize_shape_test,
                      'hflip': args.random_hflip, 'crop_shape': crop_shape, 'crop_padding': crop_padding,
                      'resized_crop_shape': resized_crop_shape, 'center_crop_shape': center_crop_shape,
                      'norm_mean': norm_mean, 'norm_std': norm_std,
                      'num_clients': args.num_clients, 'num_classes_per_client': args.num_classes_per_client,
                      'run_number': args.run_number, 'balance_degree': args.balance_degree}

    # ##### Model config
    model_config = {'name': args.model_name, 'group_size': args.group_size}

    # ##### Loss function config
    loss_func_config = {'name': args.loss_function_name}

    # ##### Optimizer config
    optimizer_config = {'name': args.optimizer_name, 'learning_rate': args.learning_rate,
                        'momentum': args.momentum, 'weight_decay': args.weight_decay,
                        'dampening': args.dampening, 'nesterov': args.nesterov,
                        'decay_epochs': args.decay_epochs,
                        'decay_multiplier': args.decay_multiplier
                        }

    # ##### Logger config
    logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=args.log_level)

    # ##### Approach instantiation
    if args.approach_name == 'fedavg':
        approach = FedAvgApproach(dataset_config=dataset_config, train_config=train_config,
                                  model_config=model_config, loss_func_config=loss_func_config,
                                  optimizer_config=optimizer_config)

    # ##### Result file config
    aug_str = ""
    aug_str += 'r' if resize_shape_train else ""
    aug_str += 'f' if args.random_hflip else ""
    aug_str += 'c' if (crop_shape or resized_crop_shape) else ""
    aug_str += 'n' if norm_mean and norm_std else ""
    if aug_str:
        aug_str = '_' + aug_str
    dataset_str = args.dataset_name + f'{aug_str}'

    model_str = args.model_name + (f'-gs{args.group_size}' if 'gn' in args.model_name else "")

    optimizer_str = args.optimizer_name + (f'-m{args.momentum}' if (args.optimizer_name == 'sgd') else "") + \
                    f'-wd{args.weight_decay}' + f'-dmp{args.dampening}' + ('-nesterov' if args.nesterov else "") + f"-lr{args.learning_rate}"

    loss_func_str = args.loss_function_name

    train_str = f"bs{args.batch_size}" + f"-K{args.num_clients}-C{selection_rate}-L{args.num_classes_per_client}-B{args.balance_degree}" + \
                (f"-E{args.num_local_epochs}" if args.approach_name == 'fedavg' else "") + \
                (f"-U{args.num_local_updates}" if args.approach_name == 'fedlmb' else "")

    result_file_name = f"{args.approach_name}-{dataset_str}-{model_str}-{loss_func_str}-" \
                       f"{optimizer_str}-{train_str}"
    result_file_name += f"-run{args.run_number}.csv"

    result_file = ResultFile(result_file_name=result_file_name)

    print()
    # ############ Print simulation parameters ##################################
    logger.info("-------------- SIMULATION PARAMETERS ----------------")
    logger.info(f"Approach: {args.approach_name}")
    logger.info(f"Clients: {args.num_clients}")
    logger.info(f"Selection rate: {selection_rate}")
    logger.info(f"Classes per client: {args.num_classes_per_client}")
    logger.info(f"Balance degree: {args.balance_degree}")
    if args.approach_name == 'fedavg':
        logger.info(f"Local epochs: {args.num_local_epochs}")

    logger.info(f"Dataset: {args.dataset_name}")
    if resize_shape_train:
        logger.info(f'Resize shape of train images: {resize_shape_train}')
    if crop_shape:
        logger.info(f'Random crop shape: {crop_shape}')
    if crop_padding:
        logger.info(f'Random crop padding: {crop_padding}')
    if resized_crop_shape:
        logger.info(f'Random resized crop shape: {resized_crop_shape}')
    if resize_shape_test:
        logger.info(f'Resize shape of test images: {resize_shape_test}')
    if center_crop_shape:
        logger.info(f'Center crop shape of test images: {center_crop_shape}')
    if args.random_hflip:
        logger.info(f'Random horizontal flip: {args.random_hflip}')
    if args.norm_mean:
        logger.info(f"Norm mean: {norm_mean}")
    if args.norm_std:
        logger.info(f"Norm std: {norm_std}")

    logger.info(f"Model: {args.model_name}")
    if 'gn' in args.model_name:
        logger.info(f"Group size: {args.group_size}")

    logger.info(f"Optimizer: {args.optimizer_name}")
    logger.info(f"Loss function: {args.loss_function_name}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    if args.optimizer_name == 'sgd':
        logger.info(f"Momentum: {args.momentum}")
        logger.info(f"Dampening: {args.dampening}")
        logger.info(f"Nesterov: {args.nesterov}")
    logger.info(f"Weight decay: {args.weight_decay}")

    logger.info(f"Log level: {args.log_level}")

    logger.info(f"Run number: {args.run_number}")
    logger.info(f"Communication rounds: {args.num_comm_rounds}")

    print()
# ############ Run simulation ##################################
    logger.info("-------------- SIMULATION STARTED ----------------")

    result_file.write_header('round,train_loss,test_loss,train_accuracy,test_accuracy')

    # evaluate initial model, i.e. at epoch=0
    logger.info("--- Round # 0")
    test_loss, test_accuracy = approach.evaluate_model()
    logger.info(f'Test loss: {test_loss:.6f}, Test accuracy: {test_accuracy:.6f}')

    for current_round in np.arange(1, args.num_comm_rounds + 1):
        logger.info(f"--- Round # {current_round}")

        # train model
        train_loss, train_accuracy = approach.train_model()

        if current_round % args.eval_freq == 0:
            # test model
            test_loss, test_accuracy = approach.evaluate_model()

            # logger.info(f'Train loss: {train_loss:.6f}, Train accuracy: {train_accuracy:.6f}')
            logger.info(f'Test loss: {test_loss:.6f}, Test accuracy: {test_accuracy:.6f}')

            # write result
            result_file.write_result(comm_round=current_round,
                                     result_list=[train_loss, test_loss, train_accuracy, test_accuracy])

    result_file.close()
    logger.info(f"-------------- SIMULATION DONE ----------------")


if __name__ == "__main__":
    main()
