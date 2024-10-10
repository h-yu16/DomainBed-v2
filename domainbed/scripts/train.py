# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import json
import os
import sys
from copy import deepcopy
from collections import defaultdict

from math import ceil
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed.datasets import get_dataloader, get_mix_dataloader, num_classes_dict, checkpoint_step_dict, train_steps_dict
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc, swa_utils
from domainbed.lib.Logger import Logger
from domainbed.lib.swad import LossValley

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
home = os.path.expanduser("~")

def get_dataloaders(args, hparams, logger):
    # set up dataloaders
    if args.mix:
        # TO FIX
        trainloaders = [get_mix_dataloader(args.txtdir, args.dataset, args.source, "train", hparams["batch_size"]),]
        logger.info("Train size: %d" % len(trainloaders[0].dataset))
    else:
        trainloaders = []
        valloaders = []
        for domain in args.source:
            loader_dict = get_dataloader(args.txtdir, args.dataset, domain, hparams["batch_size"], mode="train", split=True, holdout_fraction=args.holdout_fraction, seed=args.data_seed)
            trainloaders.append(loader_dict["train"])
            valloaders.append(loader_dict["eval"])
    testloaders = [get_dataloader(args.txtdir, args.dataset, domain, hparams["batch_size"], mode="eval", split=False) for domain in args.target]

    for index, domain in enumerate(args.source):
        logger.info("Train %s size: %d" % (domain, len(trainloaders[index].dataset)))
    for index, domain in enumerate(args.source):
        logger.info("Val %s size: %d" % (domain, len(valloaders[index].dataset)))
    for index, domain in enumerate(args.target):
        logger.info("Test %s size: %d" % (domain, len(testloaders[index].dataset)))

    return trainloaders, valloaders, testloaders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--txtdir', type=str, default="%s/dataset/txtlist-v2"%home)
    parser.add_argument('--dataset', type=str, default="NICO_plus")
    parser.add_argument("--source", nargs='+')
    parser.add_argument("--target", nargs="+")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--mix', action='store_true')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--pretrain', default="Supervised")
    parser.add_argument('--linear_probe', action='store_true')
    parser.add_argument('--arch', default="resnet50")
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--scheduler', type=str, default="None")
    parser.add_argument('--swad', action='store_true')
    parser.add_argument('--steps', type=int, default=0,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--hparams_str', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_fixed_config', type=str,
        help='config of hparams (fixed value, not random)')
    parser.add_argument('--hparams_rand_config', type=str,
        help='Path of json config for random hparams generation')
    parser.add_argument('--data_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--checkpoint_step_freq', type=int, default=0,
        help='Checkpoint every N steps.')
    parser.add_argument('--stepval_freq', type=int, default=20,
        help='print step val every N steps.')
    parser.add_argument('--log_dir', type=str, default="logs")
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--save_model_best', action='store_true')
    parser.add_argument('--load_model_best', action='store_true')
    parser.add_argument('--result_name', type=str)

    args = parser.parse_args()

    misc.setup_seed(args.seed)

    if args.checkpoint_step_freq == 0:
        args.checkpoint_step_freq = checkpoint_step_dict[args.dataset]
        
    if args.steps == 0:
        args.steps = train_steps_dict[args.dataset]

    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    # else:
    #     hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
    #         misc.seed_hash(args.hparams_seed, args.trial_seed))
    result_dict = {'intermediate':[], 'final': 0.0}
    if args.result_name:
        os.makedirs(os.path.join(args.output_dir, args.result_name), exist_ok=True)
        cur_seed = misc.seed_hash(args.seed, args.data_seed, args.hparams_seed)

    if args.hparams_str:
        hparams.update(json.loads(args.hparams_str))
    elif args.hparams_fixed_config:
        with open(args.hparams_fixed_config) as f:
            hparams.update(json.load(f)) 
    elif args.hparams_rand_config:
        hparams.update(hparams_registry.load_config(args.hparams_rand_config, cur_seed))
        if args.result_name:
            hparams_file = os.path.join(args.output_dir, args.result_name, "%d_hparams.json"% cur_seed)
            metrics_file = os.path.join(args.output_dir, args.result_name, "%d_metrics.json"% cur_seed)
            if os.path.exists(metrics_file):
                print("This experiment has alread been runned before")
                sys.exit(0)
            with open(hparams_file, 'w') as f:
                json.dump(hparams, f)
                   
    # optimized_hparams = nni.get_next_parameter()
    # hparams.update(optimized_hparams)


    # hard coding
    hparams["steps"] = args.steps
    hparams["pretrain"] = args.pretrain
    hparams["swad"] = args.swad 
    hparams["linear_probe"] = args.linear_probe
    hparams["arch"] = args.arch
    hparams["optimizer"] = args.optimizer
    hparams["scheduler"] = args.scheduler

    logger = Logger(args, hparams)
    
    logger.info("Environment:")
    logger.info("\t`P`ython: {}".format(sys.version.split(" ")[0]))
    logger.info("\tPyTorch: {}".format(torch.__version__))
    logger.info("\tTorchvision: {}".format(torchvision.__version__))
    logger.info("\tCUDA: {}".format(torch.version.cuda))
    logger.info("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    logger.info("\tNumPy: {}".format(np.__version__))
    logger.info("\tPIL: {}".format(PIL.__version__))

    logger.info('Args:')
    for k, v in sorted(vars(args).items()):
        logger.info('\t{}: {}'.format(k, v))


    logger.info('HParams:')
    for k, v in sorted(hparams.items()):
        logger.info('\t{}: {}'.format(k, v))

    if args.save_model_every_checkpoint or args.save_model_best:
        os.makedirs(os.path.join(args.output_dir, Logger.get_expname(args, hparams)), exist_ok=True)

    if args.load_model_best:
        load_model_path = os.path.join(args.output_dir, Logger.get_expname(args, hparams), 'model_best_seed%d.pkl' % args.seed)
        algorithm_dict = torch.load(load_model_path, map_location="cpu")["model_dict"]
        logger.info("Load model from %s" % load_model_path)
    else:
        algorithm_dict = None
        logger.info("Do not load model")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    train_loaders, val_loaders, test_loaders = get_dataloaders(args, hparams, logger)

    steps_per_epoch = ceil(min([len(train_loader.dataset)/hparams['batch_size'] for train_loader in train_loaders]))
    hparams["epochs"] = ceil(args.steps/steps_per_epoch)
    logger.info("Steps per epoch: %d, number of epochs: %d, number of steps: %d" % (steps_per_epoch, hparams["epochs"], args.steps))

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class((3, 224, 224), num_classes_dict[args.dataset],
        len(args.source), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    if hparams["swad"]:
        swad_algorithm = swa_utils.AveragedModel(algorithm)
        swad = LossValley(hparams["n_converge"], hparams["n_tolerance"], hparams["tolerance_ratio"])

    train_minibatches_iterator = zip(*train_loaders)

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, Logger.get_expname(args, hparams), filename))

    train_accs = []
    val_accs = dict()
    test_accs = dict()
    best_val_accs = defaultdict(float)
    best_test_accs = dict()
    for step in range(args.steps):
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        step_vals = algorithm.update(minibatches_device)
 
        step_val_str = ""
        
        for key, val in step_vals.items():
            if key != "train_acc":
                step_val_str = step_val_str + "%s: %.3f, " % (key, val)
        train_accs.append(step_vals["train_acc"])

        if hparams["swad"]:
            swad_algorithm.update_parameters(algorithm, step=step)

        epoch = int((step+1) / steps_per_epoch)
        if (step+1) % args.stepval_freq == 0:
            logger.info("Step %d, Epoch %d, %s train acc: %.4f" % (step+1, epoch, step_val_str, np.mean(np.array(train_accs))))
            train_accs = []
            
        if (step+1) % steps_per_epoch == 0:
            algorithm.scheduler_step()
            logger.info("Next lr: %.8f" % (algorithm.get_lr()[0]))

        if (step+1) % args.checkpoint_step_freq == 0 or step == args.steps-1:
            # validation 
            logger.info("Start validation...")
            correct_overall = 0
            total_overall = 0
            val_loss_overall = 0.0
            for index, domain in enumerate(args.source):
                acc, correct, loss, loss_sum, total = misc.accuracy_and_loss(algorithm, val_loaders[index], None, device)
                correct_overall += correct
                total_overall += total
                val_loss_overall += loss_sum
                val_accs[domain] = acc
            val_accs["overall"] = correct_overall / total_overall
            for k, v in val_accs.items():
                logger.info("Val %s: %.4f" % (k, v))
            val_loss = val_loss_overall / total_overall
            logger.info("Val loss: %.4f" % val_loss)
            if hparams["swad"]:
                swad.update_and_evaluate(swad_algorithm, val_accs["overall"], val_loss)
                if swad.dead_valley:
                    logger.info("SWAD valley is dead -> early stop !")
                    break
                swad_algorithm = swa_utils.AveragedModel(algorithm)

            # test
            logger.info("Start testing...")
            correct_overall = 0
            total_overall = 0
            for index, domain in enumerate(args.target):
                acc, correct, total = misc.accuracy(algorithm, test_loaders[index], None, device)
                correct_overall += correct
                total_overall += total
                test_accs[domain] = acc
            test_accs["overall"] = correct_overall / total_overall
            for k, v in test_accs.items():
                logger.info("Test %s: %.4f" % (k, v))
            # nni.report_intermediate_result({"default": val_accs["overall"], "test": test_accs["overall"]})
            result_dict['intermediate'].append({"default": val_accs["overall"], "test": deepcopy(test_accs)})

            if val_accs["overall"] > best_val_accs["overall"]:
                logger.info("New best validation acc at step %d epoch %d!" % (step+1, epoch))
                best_val_accs = deepcopy(val_accs)
                best_test_accs = deepcopy(test_accs)
                if args.save_model_best:
                    save_checkpoint('model_best_seed%d.pkl' % args.seed)
                    logger.info("Save current best model at step %d epoch %d!" % (step+1, epoch))

            if args.save_model_every_checkpoint:
                save_checkpoint('model_step%d_seed%d.pkl' % (step+1, args.seed))
            logger.info("")

    
    if hparams["swad"]:
        logger.info("Evaluate SWAD ...")
        swad_algorithm = swad.get_final_model()
        if hparams["pretrain"] == "None":
            logger.info(f"Update SWAD BN statistics for %d steps ..." % hparams["n_steps_bn"])
            swa_utils.update_bn(train_minibatches_iterator, swad_algorithm, hparams["n_steps_bn"])
        correct_overall = 0
        total_overall = 0
        for index, domain in enumerate(args.target):
            acc, correct, total = misc.accuracy(swad_algorithm, test_loaders[index], None, device)
            correct_overall += correct
            total_overall += total
            best_test_accs[domain] = acc
        best_test_accs["overall"] = correct_overall / total_overall
        start = swad_algorithm.start_step
        end = swad_algorithm.end_step
        step_str = f" [{start}-{end}]  (N={swad_algorithm.n_averaged})"
        logger.info(step_str)

    logger.info("Final result")
    for k, v in best_val_accs.items():
        logger.info("Best val %s: %.4f" % (k, v))
    for k, v in best_test_accs.items():
        logger.info("Best test %s: %.4f" % (k, v))

    # nni.report_final_result({"default": best_val_accs["overall"], "test": best_test_accs["overall"]})
    result_dict['final'] = {"default": best_val_accs["overall"], "test": best_test_accs}
    if args.result_name:
        with open(os.path.join(args.output_dir, args.result_name, "%d_metrics.json"% cur_seed), 'w') as f:
            json.dump(result_dict, f)