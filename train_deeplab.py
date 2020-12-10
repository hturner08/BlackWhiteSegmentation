# System libs
import os
import time
import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
import torch.optim as optim

# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import TrainDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, parse_devices, setup_logger
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback

import deeplab_model
from deeplab_model.deeplabv3 import DeepLabV3

# train one epoch
def train(cfg,segmentation_module, iterator,  optimizer_e, optimizer_d, history, epoch, len_iterator, n_epochs, 
          running_lr_e, running_lr_d, crit, n_class=7):
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_miou = AverageMeter()
    
    #segmentation_module.net.train()
    #segmentation_module.zero_grad()
    # main loop
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        input_dict = next(iterator)
        inputs, masks = input_dict[0]['img_data'].cuda(), input_dict[0]['seg_label'].cuda()
        optimizer_e.zero_grad()
        optimizer_d.zero_grad()

        # adjust learning rate
        #cur_iter = i + (epoch - 1) * len_iterator
        #running_lr_encoder, running_lr_decoder = adjust_learning_rate(optimizers, cur_iter, len_iterator*n_epochs)
        

        # forward pass
        #loss, acc, mIOU = segmentation_module(inputs.float(),masks.long())
        
        pred = segmentation_module(inputs.float()) #,masks.long())
        loss = crit(pred, masks.long())
        acc, mIOU = pixel_acc(pred, masks.long(), n_class)
        
        #print(loss)
        #print(acc)
        loss = loss.mean()
        acc = acc.mean()
        mIOU = mIOU.mean()
        
        #print(loss)
        #print(acc)
        
        # Backward
        loss.backward()
#         update_model_grads(segmentation_module) #----> new addition to update the parameters of quantized neural network
        #for optimizer in optimizers:
        optimizer_e.step()
        optimizer_d.step()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)
        ave_miou.update(mIOU.data.item()*100)

        # calculate accuracy, and display
        if i % 100 == 0:
            print('Epoch: [{}][{}/{}], '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}, mIOU: {:.6f}'
                  .format(epoch, i, len_iterator,
                          running_lr_e, running_lr_d,
                          ave_acc.average(), ave_total_loss.average(), ave_miou.average()))

            fractional_epoch = epoch - 1 + 1. * i / len_iterator
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())



def checkpoint(net, history, cfg, epoch):
    print('Saving checkpoints...')

    weight_dict = net.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        weight_dict,
        '{}/deeplab_epoch_{}.pth'.format(cfg.DIR, epoch))

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def create_optimizers(net):
    #optimizers = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.02)
    optimizers_enc = optim.Adam([{'params':net.encoder.parameters(), 'lr':0.02}], lr=0.02)
    
    optimizers_dec = optim.Adam([{'params':net.layer5.parameters(), 'lr':0.02},
                                 {'params':net.aspp.parameters(), 'lr':0.02},
                                 {'params':net.aggregate.parameters(), 'lr':0.02},
                                 {'params':net.last_conv.parameters(), 'lr':0.02}], lr=0.02)
    
    return optimizers_enc, optimizers_dec

# def create_optimizers(nets, cfg):
#     (net_encoder, net_decoder, crit) = nets
#     optimizer_encoder = torch.optim.SGD(
#         group_weight(net_encoder),
#         lr=cfg.TRAIN.lr_encoder,
#         momentum=cfg.TRAIN.beta1,
#         weight_decay=cfg.TRAIN.weight_decay)
#     optimizer_decoder = torch.optim.SGD(
#         group_weight(net_decoder),
#         lr=cfg.TRAIN.lr_decoder,
#         momentum=cfg.TRAIN.beta1,
#         weight_decay=cfg.TRAIN.weight_decay)
#     return (optimizer_encoder, optimizer_decoder)


# def adjust_learning_rate(optimizers, cur_iter, cfg):
#     scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
#     cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
#     cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

#     (optimizer_encoder, optimizer_decoder) = optimizers
#     for param_group in optimizer_encoder.param_groups:
#         param_group['lr'] = cfg.TRAIN.running_lr_encoder
#     for param_group in optimizer_decoder.param_groups:
#         param_group['lr'] = cfg.TRAIN.running_lr_decoder
def adjust_learning_rate(optimizer_e, optimizer_d, epoch, n_epoch, lr_e=0.005, lr_d=0.01, lr_type='cos'):
    if lr_type == 'cos':  # cos without warm-up
        lr_e = 0.5 * lr_e * (1 + math.cos(math.pi * epoch / n_epoch))
        lr_d = 0.5 * lr_d * (1 + math.cos(math.pi * epoch / n_epoch))
    elif lr_type == 'exp':
        step = 1
        decay = 0.96
        lr_e = lr_e * (decay ** (epoch // step))
        lr_d = lr_d * (decay ** (epoch // step))
    elif lr_type == 'fixed':
        lr_e = lr_e
        lr_d = lr_d
    else:
        raise NotImplementedError
    print('=> lr_encoder: {}'.format(lr_e))
    print('=> lr_decoder: {}'.format(lr_d))
    for param_group in optimizer_e.param_groups:
        param_group['lr'] = lr_e
    for param_group in optimizer_d.param_groups:
        param_group['lr'] = lr_d
        
    return lr_e, lr_d

def pixel_acc(pred, label, n=7): #-->n:number of classes
        # pred : already is a probability value as a softmax layer is present as the last layer of the CNN.
        #----------------------mIOU accuracy in a new way----------------------- (hard IOU):not using prob. of prediction
        _, preds = torch.max(pred.data.cpu(), dim=1)

        # compute area intersection
        intersect = preds.clone()
        segs = label.data.cpu()
        intersect[torch.ne(preds, segs)] = -1

        area_intersect = torch.histc(intersect.float(),
                                     bins=n,
                                     min=0,
                                     max=n-1)

        # compute area union:
        preds[torch.lt(segs, 0)] = -1
        area_pred = torch.histc(preds.float(),
                                bins=n,
                                min=0,
                                max=n-1)
        area_lab = torch.histc(segs.float(),
                               bins=n,
                               min=0,
                               max=n-1)
        area_union = area_pred + area_lab - area_intersect
        mIOU = area_intersect/(area_union + 1e-10)
        #-----------------------------------------------------------------------
        _, preds = torch.max(pred.data.cpu(), dim=1)
        segs = label.data.cpu()
        
        valid = (segs >= 0).long()
        acc_sum = torch.sum(valid * (preds == segs).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        
        #calculate individual accuracies for each image in the minibatch
        acc_sum1 = torch.sum(valid * (preds == segs).long(), (1,2))
        pixel_sum1 = torch.sum(valid, (1,2))
        acc1 = acc_sum1.float() / (pixel_sum1.float() + 1e-10)
        
        #return acc
        return acc.mean(), mIOU # return mean accuracy over the minibatch.
    
def main(cfg, gpus):
    # Network Builders
    crit = nn.NLLLoss(ignore_index=-1)
    net = DeepLabV3(num_classes=7)
    
    # Dataset and Loader
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)
 
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    history = {'train': {'epoch': [], 'loss': [], 'acc': [], 'mIOU': []}, 'val': {'loss': [], 'acc': [], 'mIOU': []}}

    optimizers_enc, optimizers_dec = create_optimizers(net)

    net.cuda()
    net = torch.nn.DataParallel(net, list(range(1)))

    num_epoch=100

    best_acc = 0
    num_classes=7
    for epoch in range(num_epoch):
        iterator_train = iter(loader_train)
        running_lr_e, running_lr_d = adjust_learning_rate(optimizers_enc, optimizers_dec, epoch, num_epoch, 
                                                          lr_e=0.025, lr_d=0.050, lr_type='cos') 
        #lr changed once per epoch 
        train(cfg,net, iterator_train, optimizers_enc, optimizers_dec, history, epoch+1, 5000, 
              num_epoch, running_lr_e, running_lr_d, crit, num_classes)
        #Checkpoint Model
        checkpoint(net, history, cfg, epoch+1)
    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/deeplab.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-1",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus)
