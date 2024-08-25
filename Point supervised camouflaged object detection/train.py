#!/usr/bin/python3
#coding=utf-8

from functools import partial
import sys
import datetime
import os



import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data import dataset
import logging as logger
from lib.data_prefetcher import DataPrefetcher
import numpy as np
from train_processes import *
from tools import *

TAG = "scribblecod"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")

import subprocess


GPU_ID = subprocess.getoutput('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs')
GPU_ID="2"
CUDA_VISIBLE_DEVICES=2
#os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
#whether is a GPU is very important
""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum


def get_polylr(base_lr, last_epoch, num_steps, power):
    return base_lr * (1.0 - min(last_epoch, num_steps-1) / num_steps) **power


def validate(model, val_loader):
    model.train(False)
    avg_mae = 0.0
    cnt = 0
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            device = torch.device("cuda:2")
            #image, mask = image.cuda().float(), mask.cuda().float()
            image, mask = image.to(device).float(), mask.to(device).float()
            #out, _, _, _, _, _ ,_,_,_= model(image)
            out, _, _, _, _, _, _= model(image)
            out = F.interpolate(out, size=shape, mode='bilinear', align_corners=False)
            pred = torch.sigmoid(out[0, 0])
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            avg_mae += torch.abs(pred - mask[0]).mean().item()
            cnt += len(image)

    model.train(True)
    return (avg_mae / cnt)

def validate_multiloader(model, val_loader):
    maes = []
    for v in val_loader:
        st = time.time()
        mae = validate(model, v)
        maes.append(mae)
        print('Spent %.3fs, %s MAE: %s'%(time.time()-st, v.dataset.data_name, mae))
    return sum(maes)/len(maes)

BASE_LR = 1e-5
MAX_LR = 1e-2
total_epoch = 60
EXP_NAME = '' # change it in main
root = './CodDataset'

def train(Dataset, Network, cfg, train_loss, start_from = 0):
    ## dataset
    data = Dataset.Data(cfg)

    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
    val_cfg = [Dataset.Config(datapath=f'{root}/test/{i}' , mode='test') for i in ['CHAMELEON', 'CAMO', 'COD10K','HCK4']]
    #val_cfg = [Dataset.Config(datapath=f'{root}/test/{i}', mode='test') for i in ['CAMO']]
    #val_cfg = [Dataset.Config(datapath=f'{root}', mode='test') for i in ['CHAMELEON', 'CAMO', 'COD10K']]

    val_data = [Dataset.Data(v) for v in val_cfg]
    val_loaders = [DataLoader(v, batch_size=1, shuffle=False, num_workers=4) for v in val_data] 
    min_mae = 1.0
    best_epoch = 0
    ## network
    #by_chf
    device=torch.device("cuda:2")

    net = Network(cfg)
    # print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))

    net=torch.nn.DataParallel(net,device_ids=[2]).to(device)
    net.train(True)
    #net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    #can change SGD as adam
    optimizer = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)

    ## log
    sw = SummaryWriter(cfg.savepath)
    db_size = len(loader)
    global_step = start_from * db_size
    et = 0

    # -------------------------- training ------------------------------------

    # mae = validate_multiloader(net, val_loader)
    # print(mae)
    count1=1
    count2=1
    for epoch in range(start_from, cfg.epoch):
        prefetcher = DataPrefetcher(loader)
        batch_idx = -1
        image, mask = prefetcher.next()
        if epoch==0 and count1==1:
            count1=0
            r=mask.clone()
            r=np.array(r.cpu())
            np.savetxt('xxx.txt',r[0][0])
        if epoch==1 and count2==1:
            count2=0
            r=mask.clone()
            r=np.array(r.cpu())
            np.savetxt('xx.txt',r[0][0])
        """by_chf"""
        image=image.to(device)
        mask=mask.to(device)
        #print(mask)
        #train_loss=train_loss.to(device)


        while image is not None:
            st = time.time()
            niter = epoch * db_size + batch_idx
            lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch*db_size, niter, ratio=1.)
            optimizer.param_groups[0]['lr'] = 0.1 * lr  # for backbone
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum = momentum
            batch_idx += 1
            global_step += 1

            loss2, loss3, loss4, loss5, loss6 = train_loss(image, mask, net, dict(epoch=epoch+1, global_step=global_step, sw=sw, t_epo=cfg.epoch))
            loss3=torch.mean(loss3)#by_chf:for contrastive learning created
            ######  objective function  ######
            loss = loss2*1 + loss3*0.8 + loss4*0.6 + loss5*0.4 + loss6*0.2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalar('loss', loss.item(), global_step=global_step)

            image, mask = prefetcher.next()
            ta = time.time() - st
            et = 0.9*et + 0.1*ta if et>0 else ta
            if batch_idx % 10 == 0:
                msg = '%s| %s | eta:%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f' % (TAG, datetime.datetime.now(), datetime.timedelta(seconds = int((cfg.epoch*db_size-niter)*et)), global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item())
                print(msg)
                logger.info(msg)
        
        #if (epoch+1)%10==0:
        if (epoch + 1) % 10 == 0 :
            mae = validate_multiloader(net, val_loaders)
            print('VAL MAE:%s' % (mae))
            sw.add_scalar('val', mae, global_step=global_step)
            if mae < min_mae :
                min_mae = mae
                best_epoch = epoch + 1
                if epoch > cfg.epoch//2:
                    torch.save(net.state_dict(), cfg.savepath + '/model-best.pth')
                print('best epoch is:%d, MAE:%s' % (best_epoch, min_mae))
            
        if epoch == cfg.epoch-2 or epoch == cfg.epoch-1 or (epoch+1) % 30 == 0 or epoch==5 or epoch==10 or epoch==15 or epoch==20:
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
    print('min val mae for {} is {}'.format(EXP_NAME, min_mae))

if __name__=='__main__':
    #the cfg can be change !!! ,you can try to fix it !!! as super-hy can be fixed in Final vision !!!
    cfg = [.15, 185, 16, 1]

    w_ft, ft_st, topk,w_ftp = cfg
    EXP_NAME = f'trained'
    cfg = dataset.Config(datapath=f'{root}', savepath=f'./out/{EXP_NAME}/', mode='train', batch=8, lr=1e-3, momen=0.9, decay=5e-4, epoch=total_epoch, label_dir = 'Label')

    from net import Net
    tm = partial(train_loss, w_ft=w_ft, ft_st = ft_st, ft_fct=.5, ft_dct = dict(crtl_loss = False, w_ftp=w_ftp, norm=False, topk=topk, step_ratio=2), ft_head=False, mtrsf_prob=1, ops=[0,1,2], w_l2g=0.3, l_me=0.05, me_st=20, multi_sc=0)

    train(dataset, Net, cfg, tm, start_from=0)