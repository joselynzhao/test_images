#-*-coding:utf-8-*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

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
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import imageio
import legacy
from renderer import Renderer

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# --outdir=../car_stylenrf_output/create_dataset/test_dataset/trunc06 --trunc=0.6 --seeds_s=30000 --seeds_e=30100 --n_steps=16 --network=./car_model.pkl --render-program="rotation_camera"
@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds_s', type=int, default=0, help='List of random seeds')
@click.option('--seeds_e', type=int, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--render-program', default=None, show_default=True)
@click.option('--render-option', default=None, type=str, help="e.g. up_256, camera, depth")
@click.option('--n_steps', default=8, type=int, help="number of steps for each seed")
@click.option('--no-video', default=False)
@click.option('--relative_range_u_scale', default=1.0, type=float, help="relative scale on top of the original range u")
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds_s: int,
    seeds_e: int,
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str],
    render_program=None,
    render_option=None,
    n_steps=8,
    no_video=False,
    relative_range_u_scale=1.0
):


    device = torch.device('cuda')
    if os.path.isdir(network_pkl):
        network_pkl = sorted(glob.glob(network_pkl + '/*.pkl'))[-1]
    print('Loading networks from "%s"...' % network_pkl)

    with dnnlib.util.open_url(network_pkl) as f:
        network = legacy.load_network_pkl(f)
        G = network['G_ema'].to(device) # type: ignore
        D = network['D'].to(device)
    # from fairseq import pdb;pdb.set_trace()
    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # avoid persistent classes...
    from training.networks import Generator
    # from training.stylenerf import Discriminator
    from torch_utils import misc
    with torch.no_grad():
        G2 = Generator(*G.init_args, **G.init_kwargs).to(device)
        misc.copy_params_and_buffers(G, G2, require_all=False)
        # D2 = Discriminator(*D.init_args, **D.init_kwargs).to(device)
        # misc.copy_params_and_buffers(D, D2, require_all=False)
    # ??????????????????????????????
    # G = copy.deepcopy(G2).eval().requires_grad_(False).to(device)
    G2 = Renderer(G2, D, program=render_program)
    # Generate images.
    all_imgs = []

    def stack_imgs(imgs):
        img = torch.stack(imgs, dim=2)
        return img.reshape(img.size(0) * img.size(1), img.size(2) * img.size(3), 3)

    def proc_img(img):
        return (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu()

    if projected_w is not None:
        ws = np.load(projected_w)
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        img = G2(styles=ws, truncation_psi=truncation_psi, noise_mode=noise_mode, render_option=render_option)
        assert isinstance(img, List)
        imgs = [proc_img(i) for i in img]
        all_imgs += [imgs]

    else:
        seeds  = np.arange(seeds_s, seeds_e, dtype=int)
        for seed_idx, seed in enumerate(seeds):
            print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
            G2.set_random_seed(seed)
            z = torch.from_numpy(np.random.RandomState(seed).randn(2, G.z_dim)).to(device)
            relative_range_u = [0.5 - 0.5 * relative_range_u_scale, 0.5 + 0.5 * relative_range_u_scale]
            outputs = G2(
                z=z,
                c=label,
                truncation_psi=truncation_psi,
                noise_mode=noise_mode,
                render_option=render_option,
                n_steps=n_steps,
                relative_range_u=relative_range_u,
                return_cameras=True)
            if isinstance(outputs, tuple):
                imgs,cameras,ws = outputs
            else:
                imgs = outputs

            if isinstance(imgs, List):
                imgs = [proc_img(i) for i in imgs]
                if not no_video:
                    all_imgs += [imgs]

            camera_dir = os.path.join(outdir,'cameras')
            os.makedirs(camera_dir, exist_ok=True)
            for step in range(len(cameras)):
                camera = cameras[step]
                camera_0 = camera[0].detach().cpu().numpy()
                camera_1 = camera[1].detach().cpu().numpy()
                camera_2 = camera[2]
                np.savez(os.path.join(camera_dir, f'{seed:0>6d}_{step:02d}.npz'), camera_0=camera_0, camera_1=camera_1,
                         camera_2=camera_2)

            img_dir = os.path.join(outdir,"images")
            os.makedirs(img_dir, exist_ok=True)
            for step, img in enumerate(imgs):
                PIL.Image.fromarray(img[0].detach().cpu().numpy(), 'RGB').save(f'{img_dir}/{seed:0>6d}_{step:02d}.png')

            ws_dir = os.path.join(outdir,'ws')
            os.makedirs(ws_dir, exist_ok=True)
            file_path = os.path.join(ws_dir,f'{seed:0>6d}.npz')
            if not os.path.isfile(file_path):
                w = ws[0].detach().cpu().numpy()  # (17,512)
                np.savez(file_path, ws=w)

    # if len(all_imgs) > 0 and (not no_video):
    #      # write to video
    #     timestamp = time.strftime('%Y%m%d.%H%M%S',time.localtime(time.time()))
    #     seeds = ','.join([str(s) for s in seeds]) if seeds is not None else 'projected'
    #     network_pkl = network_pkl.split('/')[-1].split('.')[0]
    #     all_imgs = [stack_imgs([a[k] for a in all_imgs]).numpy() for k in range(len(all_imgs[0]))]
    #     imageio.mimwrite(f'{outdir}/{network_pkl}_{timestamp}_{seeds}.mp4', all_imgs[:], fps=10, quality=8)
    #     outdir = f'{outdir}/{network_pkl}_{timestamp}_{seeds}'
    #     os.makedirs(outdir, exist_ok=True)
    #     for step, img in enumerate(all_imgs):
    #         PIL.Image.fromarray(img, 'RGB').save(f'{outdir}/{step:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
