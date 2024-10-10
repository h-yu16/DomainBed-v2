# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from domainbed.lib import vits_moco
from domainbed.lib import simclr_resnet
import copy
import os

home = os.path.expanduser("~")

weights_path = {
    "vitb16": {
        "MoCo-v3": "%s/Pretrained_Weights/moco_v3_vit-b-300ep.pth.tar"%home      
    },
    "resnet50": {
        "MoCo": "%s/Pretrained_Weights/moco_v1_200ep_pretrain.pth.tar"%home,
        "MoCo-v2": "%s/Pretrained_Weights/moco_v2_200ep_pretrain.pth.tar"%home,
        "SimCLR-v2": "%s/Pretrained_Weights/simclr_v2_r50_1x_sk0.pth"%home,
        "SimCLR": "%s/Pretrained_Weights/simclr_checkpoint_0040.pth.tar"%home        
    }
}

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        do_pretrain = False if hparams["pretrain"] == "None" else True
        if hparams["arch"] == 'resnet18':
            self.n_outputs = 512
            self.network = torchvision.models.resnet18(pretrained=do_pretrain)
        elif hparams["arch"] == 'resnet50':
            self.n_outputs = 2048
            if hparams["pretrain"] == "Supervised":
                self.network = torchvision.models.resnet50(pretrained=True)
            elif hparams["pretrain"] == "None":
                self.network = torchvision.models.resnet50(pretrained=False)
            elif hparams["pretrain"] in ["MoCo", "MoCo-v2"]:
                self.network = torchvision.models.resnet50(pretrained=False)
                assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                checkpoint = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")
                # rename moco pre-trained keys
                state_dict = checkpoint["state_dict"]
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
                        # remove prefix
                        state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                msg = self.network.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
                print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
            elif hparams["pretrain"] == "SimCLR":
                self.network = torchvision.models.resnet50(pretrained=False)
                assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                checkpoint = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")   
                state_dict = checkpoint["state_dict"]
                for k in list(state_dict.keys()):
                    if k.startswith('backbone.'):
                        if k.startswith('backbone') and not k.startswith('backbone.fc'):
                        # remove prefix
                            state_dict[k[len("backbone."):]] = state_dict[k]
                    del state_dict[k] 
                msg = self.network.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}                        
                print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))               
            elif hparams["pretrain"] == "SimCLR-v2":
                assert os.path.isfile(weights_path[hparams["arch"]][hparams["pretrain"]])
                checkpoint = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")   
                self.network, _ = simclr_resnet.get_resnet(*simclr_resnet.name_to_params(weights_path[hparams["arch"]][hparams["pretrain"]].split("/")[-1]))
                self.network.load_state_dict(checkpoint['resnet'])
            else:
                raise NotImplementedError
        elif hparams["arch"] == "vitb16":
            self.n_outputs = 768
            if hparams["pretrain"] == "Supervised":
                self.network = torchvision.models.vit_b_16(weights="IMAGENET1K_V1")
            elif hparams["pretrain"] == "MoCo-v3":
                self.network = vits_moco.vit_base()
                linear_keyword = 'head'
                state_dict = torch.load(weights_path[hparams["arch"]][hparams["pretrain"]], map_location="cpu")['state_dict']
                for k in list(state_dict.keys()):
                    # retain only base_encoder up to before the embedding layer
                    if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                        # remove prefix
                        state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                msg = self.network.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
                del self.network.head
                self.network.head = Identity()
                print("=> loaded pre-trained model '{}'".format(weights_path[hparams["arch"]][hparams["pretrain"]]))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        if "resnet" in hparams["arch"]:
            del self.network.fc
            self.network.fc = Identity()
        elif "vit" in hparams["arch"]:
            if "heads" in dir(self.network):
                del self.network.heads
                self.network.heads = Identity()
        else:
            raise NotImplementedError
        if hparams["pretrain"] != "None":
            self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.hparams["pretrain"] != "None":
            self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()



class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)
