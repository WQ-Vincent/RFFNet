import os
from config import parse
import math

opt = parse('training_RFFNet_DVD.yml')

gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import numpy as np

from utils.dir_utils import mkdir, mkdirs
from utils import util

import importlib
module_name = 'dataset_' + opt['DATASET']
module = importlib.import_module(module_name)
DataLoaderTrain = module.DataLoaderTrain

import losses
from warmup_scheduler import GradualWarmupScheduler
from pdb import set_trace as stx
from model.get_model import get_model, load_model
import time
import logging
from logging import handlers
from copy import deepcopy
validation_name = 'validation_' + opt['DATASET']
validation = getattr(importlib.import_module('test'), validation_name)
# from test import validation, validation2

current_time = time.strftime("%Y%m%d%H", time.localtime())
log_dir = os.path.join('./log', opt['MODEL']['MODE'], opt['DATASET']+'_'+str(current_time)+'.log')
os.makedirs(os.path.dirname(log_dir), exist_ok=True)
logging.basicConfig(filename=log_dir, level=logging.DEBUG)

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt['MODEL']['MODE']

opt_train = opt['MODEL']

model_dir = os.path.join(opt_train['SAVE_DIR'], mode, opt['DATASET'])

mkdir(model_dir)

train_dir = opt_train['TRAIN_DIR']

######### Model ###########
if not opt_train['RESUME']:
    net = get_model(mode, deepcopy(opt['NETWORK']))
    net.cuda()
else:
    net = load_model(opt_train['RESUME_PATH'], mode=mode)

# net_edge = load_model(opt.Edge_Detect.MODEL, mode='AE')
# net_edge.eval()

if isinstance(net, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
    net = net.module
net_str = str(net)
net_params = sum(map(lambda x: x.numel(), net.parameters()))
print('Network parameters:', net_params)

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use ", torch.cuda.device_count(), " GPUs!\n\n")

new_lr = opt['OPTIM']['LR_INITIAL']

optimizer = optim.Adam(net.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt_train['NUM_EPOCHS'] - warmup_epochs + 40,
                                                        eta_min=opt['OPTIM']['LR_MIN'])
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Resume ###########
if opt_train['RESUME']:
    start_epoch = util.load_start_epoch(opt_train['RESUME_PATH']) + 1
    util.load_optim(optimizer, opt_train['RESUME_PATH'])

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    net = nn.DataParallel(net, device_ids=device_ids)
    # net_edge = nn.DataParallel(net_edge, device_ids=device_ids)

######### Loss ###########
criterion_recon = losses.CharbonnierLoss()
# criterion_grad = losses.DiceLoss()

######### DataLoaders ###########
train_dataset = DataLoaderTrain(opt_train['TRAIN_DIR'], opt_train['TRAIN_PS'])
train_loader = DataLoader(dataset=train_dataset, batch_size=opt['OPTIM']['BATCH_SIZE'], shuffle=True, num_workers=32,
                          drop_last=True, pin_memory=True)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt_train['NUM_EPOCHS'] + 1))
print('===> Loading datasets')

num_iter_per_epoch = math.ceil(len(train_dataset) / (opt['OPTIM']['BATCH_SIZE'] * len(opt['GPU'])))

best_psnr = 0.
best_epoch = 0
no =0
######### Training ###########
for epoch in range(start_epoch, opt_train['NUM_EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    net.train()
    time_epoch_start = tstart = time.time()

    for i, data in enumerate(train_loader):
        # zero_grad
        for param in net.parameters():
            param.grad = None

        cur_time = time.time()
        inp_rgb, nir, gt_rgb = data
        inp_rgb, nir, gt_rgb = inp_rgb.cuda(), nir.cuda(), gt_rgb.cuda()

        tdata = time.time() - tstart

        pred, pred_mid = net(inp_rgb, nir)
        loss_spt1 = criterion_recon(pred, gt_rgb)
        loss_spt2 = criterion_recon(pred_mid, gt_rgb)
        loss_spt = loss_spt1+loss_spt2

        pred_fft = torch.fft.fft2(pred, dim=(-2,-1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        gt_fft = torch.fft.fft2(gt_rgb, dim=(-2,-1))
        gt_fft = torch.stack((gt_fft.real, gt_fft.imag), -1)
        loss_frq = criterion_recon(pred_fft, gt_fft)

        loss = loss_spt + 0.1*loss_frq
        
        loss.backward()
        optimizer.step()

        ttrain = time.time() - tstart
        time_passed = time.time() - time_epoch_start

        epoch_loss += loss.item()
        net.train()

        tstart = time.time()

        if i % 50 == 0:
            outputs = [
                "e: {}, {}/{}".format(epoch, i, num_iter_per_epoch),
                "tdata:{:.2f} s".format(ttrain),
                "loss_rgb {:.4f} ".format(loss_spt1.item()*1000),
                "loss_freq {:.4f} ".format(loss_frq.item() * 1000),
            ]

            outputs += [
                'passed:{:.2f}'.format(time_passed),
                "lr:{:.4g}".format(optimizer.param_groups[0]['lr'] * 100000),
                "dp/tot: {:.2g}".format(tdata / ttrain),
            ]
            print("  ".join(outputs))

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    if epoch % opt['SAVE_MODEL_FREQ'] == 0:
        torch.save({'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))
        print('model checkpoint saved')

    if epoch % opt['VAL_FREQ'] == 0:
        psnr_2, ssim_2 = validation(net, opt)
        if psnr_2 <= 31.8: continue
        opt['TEST']['SIGMA']=4
        psnr_4, ssim_4 = validation(net, opt)
        if psnr_4 <= 29.7: continue
        opt['TEST']['SIGMA']=6
        psnr_6, ssim_6 = validation(net, opt)
        #psnr = validation(net, opt)
        if psnr_6 > 28.3:
            opt['TEST']['SIGMA']=8
            psnr_8, ssim_8 = validation(net, opt)
            no+=1
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_best_"+str(no)+".pth"))
            print('model checkpoint saved')
            logging.info('The current best model with:\n\
                         PSNR_2=%.4f, SSIM_2=%.5f\n\
                         PSNR_4=%.4f, SSIM_4=%.5f\n\
                         PSNR_6=%.4f, SSIM_6=%.5f\n\
                         PSNR_8=%.4f, SSIM_8=%.5f\n\
                         was trained in epoch:%d with number %d'\
                         % (psnr_2, ssim_2,\
                            psnr_4, ssim_4,\
                            psnr_6, ssim_6,\
                            psnr_8, ssim_8,\
                            best_epoch, no))
            print('The current best model with:\n\
                  PSNR_2=%.4f, SSIM_2=%.5f\n\
                  PSNR_4=%.4f, SSIM_4=%.5f\n\
                  PSNR_6=%.4f, SSIM_6=%.5f\n\
                  PSNR_8=%.4f, SSIM_8=%.5f\n\
                  was trained in epoch:%d with number %d'\
                  % (psnr_2, ssim_2,\
                     psnr_4, ssim_4,\
                     psnr_6, ssim_6,\
                     psnr_8, ssim_8,\
                     best_epoch, no))
            continue


print('Training done.')