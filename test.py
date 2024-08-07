import os
import sys
sys.path.append('..')

from config import parse
opt = parse('training_RFFNet_FAID.yml')

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader
import random
import numpy as np
from collections import OrderedDict
import importlib
from tqdm import tqdm
from pdb import set_trace as stx
from model.get_model import get_model, get_pretrain, load_model
from utils.tools import gather_patches_into_whole, validation_on_PSNR_and_SSIM, compute_psnr, compute_ssim
from utils.tools import make_view
from utils.dstools import getshow_bgr, getshow_ir
from utils.util import saveImgForVis
from tqdm import tqdm
import lpips
from skimage.metrics import peak_signal_noise_ratio as c_psnr
from skimage.metrics import structural_similarity as c_ssim

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def validation_DVD(net, opt, saveimg=False, record=False):
    from dataset_DVD import DataLoaderTest
    lpfunc = lpips.LPIPS(net='vgg').cuda()
    net.eval()
    test_dataset = DataLoaderTest(opt['TEST']['TEST_DIR'], opt['TEST']['SIGMA'])
    test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

    metrics = {'psnr_i': [], 'psnr_o': [], 'ssim_i': [], 'ssim_o': [], 'lpips_o': []}
    glob_dct = {}
    if record:
        test_log_dir = os.path.join('./log/test/', opt['MODEL']['MODE'], opt['DATASET']+'_'+str(opt['TEST']['SIGMA'])+'.log')
        os.makedirs(os.path.dirname(test_log_dir), exist_ok=True)
        with open((test_log_dir), 'a') as f:
            f.truncate(0)

    for i, data in tqdm(enumerate(test_dataset)):
        inp_rgb, nir, gt_rgb, name = data
        inp_rgb, nir, gt_rgb = inp_rgb.cuda(), nir.cuda(), gt_rgb.cuda()

        with torch.no_grad():
            output, output1 = gather_patches_into_whole(net, inp_rgb, nir, gt_rgb)
            i_psnr, i_ssim, o_psnr, o_ssim = validation_on_PSNR_and_SSIM(net, inp_rgb[:, :, 259:1263, 451: 2239], nir[:, :, 259:1263, 451: 2239], gt_rgb[:, :, 259:1263, 451: 2239])
            metrics['psnr_i'].append(i_psnr)
            metrics['ssim_i'].append(i_ssim)
            metrics['psnr_o'].append(o_psnr)
            metrics['ssim_o'].append(o_ssim)
            if record:
                lpips_value = lpfunc(torch.from_numpy(output).cuda(), gt_rgb).item()
                metrics['lpips_o'].append(lpips_value)

        inp_rgb = inp_rgb[0, ...].permute(1,2,0).cpu().numpy()
        nir = nir[0, ...].permute(1,2,0).cpu().numpy()
        gt_rgb = gt_rgb[0, ...].permute(1,2,0).cpu().numpy()
        output = output[0, ...].transpose(1,2,0)
        output1 = output1[0, ...].transpose(1,2,0)

        show_dct = {
                        "input": getshow_bgr(inp_rgb),
                        "label" : getshow_bgr(gt_rgb),
                        'ir': getshow_ir(nir),        
                        'output': getshow_bgr(output),
                        'output1': getshow_bgr(output1)        
        }

        glob_dct.update({i: show_dct})
        if record:
            with open((test_log_dir), 'a') as f:
                f.write("%s %f %f\n"%(name[0].split('.')[0],o_psnr, o_ssim))
    
    if saveimg:
        saveImgForVis(os.path.join(opt['TEST']['VIS_DIR'], str(opt['TEST']['SIGMA'])), glob_dct)

    psnr = 0
    ssim = 0
    print('The calculate metrices is:')
    for k, v in metrics.items():
        v_mean = np.array(v).mean()
        print("{}: {}".format(k, v_mean))
        if record:
            with open((test_log_dir), 'a') as f:
                f.write("\n{}: {}".format(k, v_mean))
        if k == 'psnr_o':
            psnr = v_mean
        if k == 'ssim_o':
            ssim = v_mean

    return psnr, ssim


def validation_FAID(net, opt, saveimg=False, record=False):
    from dataset_FAID import DataLoaderTest
    if record:
        lpfunc = lpips.LPIPS(net='vgg').cuda()
    net.eval()
    test_dataset = DataLoaderTest(opt['TEST']['TEST_DIR'], opt['TEST']['SIGMA'])
    test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

    metrics = {'psnr_o': [], 'ssim_o': [], 'lpips_o': []}
    glob_dct = {}
    if record:
        if opt['ABLATION']==None:
            test_log_dir = os.path.join('./log/test/', opt['MODEL']['MODE'], opt['DATASET']+'_'+str(opt['TEST']['SIGMA'])+'.log')
        else:
            test_log_dir = os.path.join('./log/test/', opt['MODEL']['MODE']+'_'+str(opt['ABLATION']), opt['DATASET']+'_'+str(opt['TEST']['SIGMA'])+'.log')
        os.makedirs(os.path.dirname(test_log_dir), exist_ok=True)
        with open((test_log_dir), 'a') as f:
            f.truncate(0)

    for i, data in tqdm(enumerate(test_dataset)):
        inp_rgb, nir, gt_rgb, name = data
        inp_rgb, nir, gt_rgb = inp_rgb.cuda(), nir.cuda(), gt_rgb.cuda()

        with torch.no_grad():
            output, output1 = gather_patches_into_whole(net, inp_rgb, nir, gt_rgb)
            # i_psnr, i_ssim, o_psnr, o_ssim = validation_on_PSNR_and_SSIM(net, inp_rgb, nir, gt_rgb)

        if record:
            lpips_value = lpfunc(torch.from_numpy(output).cuda(), gt_rgb).item()
            metrics['lpips_o'].append(lpips_value)

        inp_rgb = inp_rgb[0, ...].permute(1,2,0).cpu().numpy()
        nir = nir[0, ...].permute(1,2,0).cpu().numpy()
        gt_rgb = gt_rgb[0, ...].permute(1,2,0).cpu().numpy()
        output = output[0, ...].transpose(1,2,0)

        o_psnr = compute_psnr(gt_rgb, output)
        o_ssim = compute_ssim(gt_rgb, output)
        metrics['psnr_o'].append(o_psnr)
        metrics['ssim_o'].append(o_ssim)
            
        show_dct = {
                        "input": np.clip(inp_rgb*255.0, 0, 255).astype('uint8'),
                        "label" : np.clip(gt_rgb*255.0, 0, 255).astype('uint8'),
                        'ir': np.clip(nir*255.0, 0, 255).astype('uint8'),        
                        'output': np.clip(output*255.0, 0, 255).astype('uint8') 
        }

        glob_dct.update({i: show_dct})
        if record:
            with open((test_log_dir), 'a') as f:
                f.write("%s %f %f\n"%(name[0].split('.')[0],o_psnr, o_ssim))
    
    if saveimg:
        saveImgForVis(os.path.join(opt['TEST']['VIS_DIR'], str(opt['TEST']['SIGMA'])), glob_dct)

    psnr = 0
    ssim = 0
    print('The calculate metrices is:')
    for k, v in metrics.items():
        v_mean = np.array(v).mean()
        print("{}: {}".format(k, v_mean))
        if record:
            with open((test_log_dir), 'a') as f:
                f.write("\n{}: {}".format(k, v_mean))
        if k == 'psnr_o':
            psnr = v_mean
        if k == 'ssim_o':
            ssim = v_mean

    return psnr, ssim

if __name__ == "__main__":
    net = load_model(opt['TEST']['MODEL'], mode=opt['MODEL']['MODE'], opt=opt['NETWORK'])
    validation_name = 'validation_' + opt['DATASET']
    validation = getattr(sys.modules[__name__], validation_name)
    print(validation)
    validation(net, opt, saveimg=False, record=False)
    


