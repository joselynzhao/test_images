#-*-coding:utf-8-*-
#-*-coding:utf-8-*-
# -*-coding:utf-8-*-
# -*-coding:utf-8-*-#-*-coding:utf-8-*-# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
import time
import glob
import copy
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import imageio
import legacy
import cv2
from renderer import Renderer


# ----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


def stack_imgs(imgs):
    img = torch.stack(imgs, dim=2)
    return img.reshape(img.size(0) * img.size(1), img.size(2) * img.size(3), 3)


def proc_img(img):
    return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()




# ----------------------------------------------------------------------------
os.environ['PYOPENGL_PLATFORM'] = 'egl'


# 无参数运行。
@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', default='./car_model.pkl')
@click.option('--encoder', 'encoder_pkl', help='Network pickle filename', default='./output/psp_case3/v1/checkpoints/network-snapshot-000016.pkl')
@click.option('--img_dir', help='test image path', default='./output/real_test_images1')
@click.option('--which_c', help='which_c', default='c2')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, metavar='DIR',
              default='real_test_images1')
def generate_images(
        ctx: click.Context,
        network_pkl: str,
        encoder_pkl: str,
        img_dir: str,
        which_c:str,
        outdir: str,
        class_idx: Optional[int],
):
    device = torch.device('cuda')
    from training.networks import Generator
    from torch_utils import misc
    need_c = 0
    if not need_c:
        if os.path.isdir(network_pkl):
            network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
        print('Loading networks from "%s"...' % network_pkl)
        if os.path.isdir(encoder_pkl):
            encoder_pkl = sorted(glob.glob(encoder_pkl + '/*.pkl'))[-1]
        print('Loading encoder from "%s"...' % encoder_pkl)

        with dnnlib.util.open_url(network_pkl) as f:
            network = legacy.load_network_pkl(f)
            G = network['G_ema'].to(device)  # type: ignore
        with torch.no_grad():
            G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
            misc.copy_params_and_buffers(G, G2, require_all=False)
        G = copy.deepcopy(G2).eval().requires_grad_(False).to(device)

        with dnnlib.util.open_url(encoder_pkl) as f:
            encoder = legacy.load_network_pkl(f)
            E = encoder['E'].to(device)
    else: # need_c = 1 针对encoder返回ws 和c 的情况
        with dnnlib.util.open_url(encoder_pkl) as f:
            print('Loading encoder from "%s"...' % encoder_pkl)
            encoder = legacy.load_network_pkl(f)
            E = encoder['E'].to(device)
            G = encoder['G'].to(device)
            with torch.no_grad():
                G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
                misc.copy_params_and_buffers(G, G2, require_all=False)
            G = copy.deepcopy(G2).eval().requires_grad_(False).to(device)

    # store_dir = encoder_pkl.split('/')[-1].split('.')[0]
    store_dir = os.path.join('../car_stylenrf_output/test_encoders',img_dir.split('/')[-1])
    os.makedirs(store_dir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    def get_test_image(image_path):
        image = np.array(PIL.Image.open(image_path))
        if image.shape[-1] != resolution:
            image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_AREA)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).to(device).to(torch.float32) / 127.5 - 1
        image = image.unsqueeze(0)
        return image

    def get_test_camera(camera):
        cam_0 = camera['camera_0']
        cam_1 = camera['camera_1']
        cam_2 = camera['camera_2']
        cam_0 = cam_0[0:1, :, :]
        cam_1 = cam_1[0:1, :, :]
        cam_0 = torch.from_numpy(cam_0).to(device).to(torch.float32)
        cam_1 = torch.from_numpy(cam_1).to(device).to(torch.float32)
        cam_2 = torch.from_numpy(cam_2).to(device).to(torch.float32)
        cam_2 = cam_2.unsqueeze(0)
        return cam_0, cam_1, cam_2, None

    # 获取pose
    import random
    seed = random.uniform(0,1)
    u_samples = []
    n_steps = 16

    u_samples.append(seed)
    for i in range(n_steps - 1):
        k = seed + (i + 1) / n_steps
        if k >= 1:
            k = k - 1
        u_samples.append(k)

    resolution = 256
    pose_list = np.array([0, 2, 5, 8, 10])
    select_pose = [u_samples[i] for i in pose_list]
    batch_size = len(select_pose)

    cam_0s = []
    cam_1s = []
    cam_2s = []
    batch_size = 5
    gen = G.synthesis
    for u in select_pose:
        cam_0,cam_1,cam_2,_ = gen.get_camera(batch_size=1, mode=[u, 0.5, 0.26], device=device)
        cam_2 = torch.from_numpy(np.array(cam_2)).unsqueeze(0)
        cam_0s.append(cam_0)
        cam_1s.append(cam_1)
        cam_2s.append(cam_2)
    cam_0s = torch.cat(cam_0s, 0)
    cam_1s = torch.cat(cam_1s, 0)
    cam_2s = torch.cat(cam_2s, 0)

    camera_matrics = cam_0s,cam_1s,cam_2s,None

    # 获取图像信息
    images = []
    ws_avg = G.mapping.w_avg[None, None, :]
    files = os.listdir(img_dir)
    for img in files:
        img_path = os.path.join(img_dir,img)
        image = get_test_image(img_path)
        # 用单重图像作为输入计算ws，再repeat
        if not need_c:
            output = E(image)
            if isinstance(output, tuple):
                rec_ws, _ = output
            else:
                rec_ws = output
            rec_ws = rec_ws.repeat(batch_size, 1, 1)
            rec_ws += ws_avg
            gen_images = G.get_final_output(styles=rec_ws, camera_matrices=camera_matrics)
        else:
            rec_ws,c = E(images,which_c=which_c)
            c = c.repeat(batch_size, 1, 1)
            rec_ws = rec_ws.repeat(batch_size, 1, 1,1)
            rec_ws += ws_avg
            gen_images = G.get_final_output(styles=rec_ws, camera_matrices=camera_matrics,img_c=(which_c,c))
        out_one = torch.cat([image,gen_images],0)
        images.append(out_one.detach())
    images = torch.cat(images,0)

    from torchvision import utils
    root_path = encoder_pkl.split('.')[1]
    root_path = root_path.split('/')
    file_name = '-'.join(root_path[2:])
    # os.makedirs(f"{store_dir}/sample", exist_ok=True)
    with torch.no_grad():
        # sample = torch.cat([images.detach(),gen_images.detach()])
        utils.save_image(
            images,
            f"{store_dir}/{file_name}.png",
            nrow=int(batch_size+1),
            normalize=True,
            range=(-1, 1),
        )


if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter
