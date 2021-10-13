import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function, Variable
from torchvision import datasets, models, transforms, utils
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from colorama import init, deinit, Back, Fore
from config import cfg, update_config_from_file
from script.train import train
from script.test import test
from script.detect import detect
import dataset.dataset_factory as dataset_factory
from torch.utils.data import DataLoader
from dataset.collate import collate_test
from model.vgg16 import VGG16
from model.resnet import Resnet
from utils.net_utils import parse_additional_params, arg_nms, haolin_nms
from _C import nms
from model.roi.roi_pool import ROIPool
from ebp_utils import *

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()


def ebp_conv2d(layer, top_mwp, hookForward):
    '''
    Some useful resources:
    # https://github.com/alisaaalehi/convolution_as_multiplication/blob/master/Convolution_as_multiplication.ipynb
    # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold
    # https://stackoverflow.com/questions/56702873/is-there-an-function-in-pytorch-for-converting-convolutions-to-fully-connected-n
    '''

    bottom_activations = hookForward.input[0]
    bottom_size = bottom_activations.size()
    kernel_size = layer.kernel_size
    fold_params = dict(kernel_size = kernel_size,
                        padding=layer.padding,
                        stride = layer.stride)    
    batch_size, channel_size, *out_size = hookForward.output.size()
    w = layer.weight.clone()
    w[w < 0] = 0
    bot_act = F.unfold(bottom_activations, **fold_params)
    X = bot_act.transpose(1,2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
    unfolded_top_mwp = top_mwp.reshape(X.size())
    Y = torch.div(unfolded_top_mwp, X)
    Y = torch.nan_to_num(Y)
    Z = Y.transpose(1,2).matmul(w.view(w.size(0), -1)).transpose(1,2) # backward (definitely wrong rn)
    bot_mwp_unfolded = torch.mul(bot_act, Z)
    bot_mwp = F.fold(bot_mwp_unfolded, output_size=bottom_size[-2::], **fold_params)
    return(bot_mwp)

def ebp_ReLU(layer, top_mwp, bottom_activations):
    '''
    If I'm not mistaken, the mwp just stays the same in ReLU layers, since neurons with 
    0 activation will have 0 MWP, and neurons with >0 activation will have children that have identical activation to their parent.'''
    bot_mwp = top_mwp.clone()
    bot_mwp[bottom_activations < 0] = 0
    bot_mwp[bot_mwp < 0] = 0
    bot_mwp /= torch.sum(bot_mwp.flatten())  # just added
    return bot_mwp


def ebp_MaxPool2d(layer, top_mwp, hook):
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html
    bottom_activations = hook.input[0]
    pool_params = dict(kernel_size = layer.kernel_size,
    stride = layer.stride,
    padding = layer.padding)
    pool = nn.MaxPool2d(**pool_params, dilation=layer.dilation, ceil_mode=layer.ceil_mode, return_indices=True)
    unpool = nn.MaxUnpool2d(**pool_params)
    X, inds = pool(bottom_activations)
    Y = torch.div(top_mwp, X)
    Y = torch.nan_to_num(Y)
    Z = unpool(Y, inds, output_size=bottom_activations.size())
    res = torch.mul(bottom_activations, Z)
    return res

def ebp_full_con(layer, top_mwp, hook):
    '''
    Assumes that layer.weight is arranged as a typical affinity matrix,
    where m is the size of the "bottom" layer and n is the size of the "top" layer

    Trying to simply vectorize formula (6) in Liu et. al
    '''
    bottom_activations = hook.input[0]
    W_pos = layer.weight.clone()
    W_pos[W_pos < 0] = 0
    X = torch.matmul(W_pos, bottom_activations.transpose(0,1))
    Y = torch.div(top_mwp, X.transpose(0,1))
    Y = torch.nan_to_num(Y)  # div by zero results in zero
    Z = torch.matmul(W_pos.transpose(0,1), Y.transpose(0,1))
    result = torch.mul(bottom_activations, Z.transpose(0,1))
    return result

def ebp_roiPool(layer, top_mwp, forward_hook, image_size):
    '''
    Here, the bottom_activations are 512 convolutional feature maps (37x59)
    And the output (top_mwp) is 512 pooled feature maps (7x7) for EACH of the 300 ROIs

    But in order to backprop through these, we need some way of getting the ROIs

    ROI is something like [x1, y1, x2, y2] = [319.4034, 355.4951, 368.6730, 400.4579]
    '''
    argmaxes = layer.amax
    bottom_activations, rois = forward_hook.input
    # print(f"size of rois: {rois.size()}")  # [300, 5]
    bot_size = bottom_activations.size()  # [1, 512, 37, 59]
    top_size = top_mwp.size()  # [300, 512, 7, 7]
    
    bot_mwp = torch.zeros(size=bot_size).flatten(start_dim=2, end_dim=3)
    for region in range(rois.size()[0]):
        for conv_channel in range(bot_size[1]):
            amax = argmaxes[region, conv_channel,:,:].flatten().long()
            # print(amax)
            bot_mwp[0, conv_channel, amax] += top_mwp[region, conv_channel, :, :].flatten()
    return bot_mwp.reshape(bot_size)

def draw_im(img, boxes, im_info):
    trusize = im_info[:2]
    img_size = img.size()
    img = (img - img.min())/(img.max() - img.min())  # img.min() should be 0 anyway but JIC
    img = img.detach().numpy()
    plt.imshow(img, cmap=plt.cm.inferno)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    
def faster_rcnn_ebp(model, input, fname, contrastive=False, target_layer=37):
    '''
    INPUTS:
        model: faster_rcnn module (going to be tested with VGG16 backbone)
        prob_output: "ideal" score outputs
        input: tuple containing (image_data, image_info) for forward pass
        target_layer: layer to stop at (but this is indexed from the last to the start so kinda confusing)

    Structure of forward pass:
        RCNN_base > RCNN_roi_layer > pooling (this does a .view() operation) > RCNN_top > RCNN_cls_score

    Sections:
        model.RCNN_top is a nn.Sequential() which contains the backbone's classifier layers
        model.RCNN_base is a nn.Sequential() containing the backbone's feature extractor layers
    '''

    # NOTE: this section will need to be modified for different network architectures.
    # Just construct this list of layers in the top-down order that your signal goes through the network
    layers = []
    for x in model.RCNN_base._modules.items():
        layers.append(x[1])
    layers.append(model.RCNN_roi_layer)  # a single layer 
    for x in model.RCNN_top._modules.items():
        layers.append(x[1])
    layers.append(model.RCNN_cls_score)  # a single layer 

    # register forward hooks to get activations
    hookForward = [Hook(layer) for layer in layers[::-1]]
    hookBackward = [Hook(layer,backward=True) for layer in layers[::-1]]  # might not need this


    # forward pass
    img_info = input[1]
    scores, boxes, *_ = model(input[0], img_info, None)
    rois, rpn_loss_cls, rpn_loss_bbox = model.RCNN_rpn(model.RCNN_base(input[0]), img_info, None)

    # postprocess results to generate prob_output
    indices = arg_nms(scores, boxes)
    prob_output = scores.clone()
    prob_output = prob_output.squeeze()
    prob_output[indices, :] = Tensor([-10,10])
    not_used = [x for x in range(len(prob_output[:,0])) if x not in indices]
    if contrastive:
        prob_output[not_used, :] = Tensor([10, -10])
    else:
        prob_output[not_used, :] = Tensor([0,0])  # previous result used [10, -10]. That should be what contrastive represents 
    # print(f"prob_output: {prob_output}")
    top_mwp = prob_output
    boxes = boxes.squeeze()[indices, 4:]

    for i,layer in enumerate(layers[::-1]):  # iterate backward
        print(f"Layer: {layer}, #{i}")
        if type(layer) == nn.Dropout:
            continue
        if i == target_layer:
            break
        bottom_activations = hookForward[i].input[0]  
        if type(layer) == nn.Conv2d:
            # mwp = torch.sum(top_mwp.squeeze(), dim=0)
            # mwp = draw_im(mwp, boxes.detach().numpy(), img_info.squeeze())
            top_mwp = ebp_conv2d(layer, top_mwp, hookForward[i]) 
        elif type(layer) == nn.ReLU:
            top_mwp = ebp_ReLU(layer, top_mwp, bottom_activations)  # I actually think relu could be the problem
            # pass  # try ignoring
        elif type(layer) == nn.Linear:
            # print(f"Shape of weights: {layer.weight.size()}\n")
            top_mwp = ebp_full_con(layer, top_mwp, hookForward[i])
            # top_mwp = bottom_activations # skip the layer (this is not the algorithm, just trying to get to lower layers quickly)
            pass
        elif layer == model.RCNN_roi_layer:
            top_mwp = top_mwp.view(300, 512, 7, 7)
            top_mwp = ebp_roiPool(layer, top_mwp, hookForward[i], img_info.squeeze()[:2])
        elif type(layer) == nn.MaxPool2d:
            top_mwp = ebp_MaxPool2d(layer, top_mwp, hookForward[i])
        else:
            print(f"Did not recognize layer {layer}")
        assert top_mwp.max() != top_mwp.min(), "Constant MWP"
        assert not torch.isnan(top_mwp).any(), "NaN values"

    # show results
    mwp = torch.sum(top_mwp.squeeze(), dim=0)
    return(mwp, boxes)
