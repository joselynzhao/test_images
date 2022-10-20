#-*-coding:utf-8-*-

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from random import random
from dnnlib import camera
import os
# os.environ["CUDA_VISIBLE_DEVICES"]='4'
import numpy as np
import torch
import copy
import torch.distributed as dist
import torchvision
import click
import dnnlib
import legacy
import pickle

from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torch_utils.ops import conv2d_gradfix
from torch_utils import misc
from torchvision import transforms, utils
from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from training.networks import Encoder
import inspect
import collections
try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
from training.my_utils import *

data_path={
    'hpcl':'./output/create_dataset/car_dataset_trunc075/images',
    'jdt':'/workspace/datasets/car_zj/images'
}

# --data=./output/car_dataset_3w_test/images --g_ckpt=car_model.pkl --outdir=../car_stylenrf_output/psp_case2/debug
@click.command()
@click.option("--g_ckpt", type=str, default='./car_model.pkl')
@click.option("--which_server", type=str, default='jdt')
@click.option("--e_ckpt", type=str, default=None)
@click.option("--max_steps", type=int, default=10000)
@click.option("--batch", type=int, default=4)
@click.option("--lr", type=float, default=0.0001)
@click.option("--local_rank", type=int, default=0)
@click.option("--lambda_w", type=float, default=1.0)
@click.option("--lambda_c", type=float, default=1.0)
@click.option("--lambda_img", type=float, default=1.0)
@click.option("--lambda_l2", type=float, default=1.0)
@click.option("--which_c", type=str, default='p2')  # encoder attr
@click.option("--adv", type=float, default=0.05)
@click.option("--tensorboard", type=bool, default=True)
@click.option("--outdir", type=str, default='./output/psp_mvs_one_img/debug')
@click.option("--resume", type=bool, default=False)  # true则进行resume
@click.option("--insert_layer", type=int, default=3)  #  在net中进行特征的时候在哪一层后面进行合并 stylenerf attr
@click.option("--num_views", type=int, default=2)  #
@click.option("--with_gt_w", type=bool, default=True)  #
@click.option("--with_net_fea", type=bool, default=False)  #

def main(outdir, g_ckpt, e_ckpt,
             max_steps, batch, lr,local_rank, lambda_w,lambda_c,
             lambda_img,lambda_l2, which_c,adv, tensorboard,resume,insert_layer,which_server,num_views,with_gt_w,with_net_fea):
    # local_rank = rank
    # setup(rank, word_size)
    # options_list = click.option()
    # print(options_list)
    data=data_path[which_server]
    random_seed = 25
    np.random.seed(random_seed)

    num_gpus = torch.cuda.device_count()  # 自动获取显卡数量
    conv2d_gradfix.enabled = True  # Improves training speed.
    device = torch.device('cuda', local_rank)

    # load the pre-trained model
    # if os.path.isdir(g_ckpt):  #基本模型信息
    #     import glob
    #     g_ckpt = sorted(glob.glob(g_ckpt + '/*.pkl'))[-1]


    if resume:
        pkls_path = os.path.join(outdir,'checkpoints')
        files =  os.listdir(pkls_path)
        files.sort()
        resume_pkl = files[-1]
        iteration = int(resume_pkl.split('-')[-1].split('.')[0])*1000
        resume_pkl_path = os.path.join(pkls_path,resume_pkl)
        print(f"resume from {resume_pkl_path}")
        with dnnlib.util.open_url(resume_pkl_path) as fp:
            network = legacy.load_network_pkl(fp)
            E = network['E'].requires_grad_(True).to(device)
            G = network['G'].requires_grad_(False).to(device)
    else:
        print('Loading networks from "%s"...' % g_ckpt)
        with dnnlib.util.open_url(g_ckpt) as fp:
            network = legacy.load_network_pkl(fp)
            G = network['G_ema'].requires_grad_(False).to(device)
        from training.networks import Generator
        from torch_utils import misc
        with torch.no_grad():
            # if 'insert_layer' not in G.init_kwargs.synthesis_kwargs:  # add new attributions
            #     G.init_kwargs.synthesis_kwargs['insert_layer'] = insert_layer
            G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
            misc.copy_params_and_buffers(G, G2, require_all=False)
        G = copy.deepcopy(G2).eval().requires_grad_(False).to(device)
        from models.encoders.psp_encoders import GradualStyleEncoder1
        E = GradualStyleEncoder1(50, 3, G.mapping.num_ws, 'ir_se',which_c=which_c).to(device)  # num_layers, input_nc, n_styles,mode='ir
        # if num_gpus >1:
        #    E = DDP(E, device_ids=[rank], output_device=rank, find_unused_parameters=True) # broadcast_buffers=False

    # E_optim = optim.Adam(E.parameters(), lr=lr*0.1, betas=(0.9, 0.99))

    # print(G)
    # print(E)
    params = list(E.parameters())
    fg_net = G.synthesis.fg_nerf.My_embedding_fg
    
    # bg_net = G.synthesis.bg_nerf
    params+= list(fg_net.parameters())
    # params+= list(bg_net.parameters())
    E_optim = optim.Adam(params, lr=lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(E_optim, step_size=50000, gamma=0.1)
    E.requires_grad_(True)
    fg_net.requires_grad_(True)
    # requires_grad(bg_net, True)


    # load the dataset
    # data_dir = os.path.join(data, 'images')
    from torch_utils import misc
    training_set_kwargs = dict(class_name='training.dataset.ImageFolderDataset_psp_case1', path=data, use_labels=False, xflip=True)
    data_loader_kwargs  = dict(pin_memory=True, num_workers=1, prefetch_factor=1)
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler  = misc.InfiniteSampler(dataset=training_set, rank=local_rank, num_replicas=num_gpus, seed=random_seed)  # for now, single GPU first.
    training_set_iterator = torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch//num_gpus, **data_loader_kwargs)
    training_set_iterator = iter(training_set_iterator)
    print('Num images: ', len(training_set))
    print('Image shape:', training_set.image_shape)


    start_iter = 0
    if resume:
        start_iter = iteration
    pbar = range(max_steps)
    pbar = tqdm(pbar, initial=start_iter, dynamic_ncols=True, smoothing=0.01)

    e_loss_val = 0
    loss_dict = {}
    # vgg_loss   = VGGLoss(device=device)
    truncation = 0.5  # 固定的
    ws_avg = G.mapping.w_avg[None, None, :]

    if SummaryWriter and tensorboard:
        logger = SummaryWriter(logdir='./checkpoint')

    for idx in pbar:
        i = idx + start_iter
        if i > max_steps:
            print("Done!")
            break

        E_optim.zero_grad()  # zero-out gradients
        # get data infor
        source_img, target_img, source_camera, target_camera, w = next(training_set_iterator)
        # handle image
        # handle image
        source_img = source_img.to(device).to(torch.float32) / 127.5 - 1
        target_img = target_img.to(device).to(torch.float32) / 127.5 - 1

        # handle w  # 暂时用不到
        w = w.to(device).to(torch.float32)

        source_camera = get_camera_metrices(source_camera, device)
        target_camera = get_camera_metrices(target_camera, device)
        source_views = source_camera[2][:, :2]  # first two
        target_views = target_camera[2][:, :2]  # first two

        # testing the same views with following two lines code
        if num_views==1:
            target_views = source_views  # the same views
            target_img = source_img

        # print(source_views)
        # print(target_views)
        source_ws, source_feature = E(source_img)
        if with_gt_w:
            source_ws= w  # 真实值
        source_ws += ws_avg
        gen_img= G.get_final_output(styles=source_ws,features=source_feature,views = target_views,source_views=source_views,insert_layer=insert_layer,input_image=source_img,with_net_fea=with_net_fea)  #

        # define loss
        loss_dict['img1_lpips'] = loss_fn_alex(gen_img.cpu(), target_img.cpu()).mean().to(device) * lambda_img
        # loss_dict['img1_l2'] = F.mse_loss(gen_img1, img_1) * lambda_l2
        # loss_dict['img2_l2'] = F.mse_loss(gen_img2, img_2) * lambda_l2


        E_loss = sum([loss_dict[l] for l in loss_dict])
        E_loss.backward()
        E_optim.step()
        scheduler.step()

        desp = '\t'.join([f'{name}: {loss_dict[name].item():.4f}' for name in loss_dict])
        pbar.set_description((desp))

        if SummaryWriter and tensorboard:
            logger.add_scalar('E_loss/total', e_loss_val, i)
            # logger.add_scalar('E_loss/vgg', vgg_loss_val, i)
            # logger.add_scalar('E_loss/l2', l2_loss_val, i)
            # logger.add_scalar('E_loss/adv', adv_loss_val, i)

        if i % 100 == 0:
            os.makedirs(f'{outdir}/sample', exist_ok=True)
            with torch.no_grad():
                sample = torch.cat([source_img.detach(), target_img.detach(), gen_img.detach()])
                utils.save_image(
                    sample,
                    f"{outdir}/sample/{str(i).zfill(6)}.png",
                    # f"./tmp_{str(i).zfill(6)}.png",
                    nrow=int(batch),
                    normalize=True,
                    range=(-1, 1),
                )

        if i % 1000 == 0:
            os.makedirs(f'{outdir}/checkpoints', exist_ok=True)
            snapshot_pkl = os.path.join(f'{outdir}/checkpoints/', f'network-snapshot-{i // 1000:06d}.pkl')
            snapshot_data = {}  # dict(training_set_kwargs=dict(training_set_kwargs))
            # snapshot_data2 = {}  # dict(training_set_kwargs=dict(training_set_kwargs))
            modules =[('G',G),('E',E)]
            for name,module in modules:
                if module is not None:
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name]=module
            # snapshot_data2['E'] = E
            # snapshot_data2['G'] = G  # 需要把G保存下来
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)
    logger.close()
    # cleanup()


if __name__ == "__main__":
    main()
