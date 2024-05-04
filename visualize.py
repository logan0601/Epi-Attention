from typing import Tuple
import torch
import numpy as np
import math
import cv2
from PIL import Image
import imageio
from op_util import get_blender_camera
from epi_util import get_epi_attention


def load(path: str, size: int):
    im = np.array(Image.open(path).resize((size, size)))
    im = im.astype(np.float32) / 255.0
    im = im[:, :, :3] * im[:, :, 3:4] + (1.0 - im[:, :, 3:4])
    im = (im * 255).astype(np.uint8)
    return im


def get_type(index: Tuple[int, int], size: int):
    elevs = torch.tensor([0, 0, 0, math.pi / 6, math.pi / 6, math.pi / 6]).cuda()
    azims = torch.tensor([0, math.pi / 2, math.pi, 0, math.pi / 2, math.pi]).cuda()

    index = torch.tensor(index).cuda()
    elev = elevs[index]
    azim = azims[index]
    pos = torch.stack(
        [
            1.5 * torch.cos(elev) * torch.cos(azim),
            1.5 * torch.cos(elev) * torch.sin(azim),
            1.5 * torch.sin(elev)
        ],
        dim=-1
    )
    cam = get_blender_camera(pos)
    ims = [load(f"assets/{i:03d}.png", size) for i in index]
    return cam, ims


def main():
    index = (3, 1)
    image_size = 128
    cam, ims = get_type(index=index, size=image_size)

    # generate point flow
    mask = get_epi_attention(cam[0], math.radians(39.6), image_size).compute_attention_mask(cam[1]).reshape(
        image_size * image_size, image_size, image_size
    ).detach().cpu()

    out = []
    for i in range(0, image_size ** 2, 2):
        src_im, tgt_im = ims[0].copy(), ims[1].copy()

        src_im = cv2.circle(src_im, (i % image_size, i // image_size), radius=1, color=(0, 0, 255), thickness=1)
        tgt_im = torch.from_numpy(tgt_im)
        tgt_im[mask[i] == True] = torch.tensor([0, 0, 255], dtype=torch.uint8)
        tgt_im = tgt_im.detach().numpy()

        comp_im = np.concatenate((src_im, tgt_im), axis=1)
        out.append(comp_im)

    out_size = 256
    out = [np.array(Image.fromarray(im).resize((out_size * 2, out_size))) for im in out]
    imageio.mimsave(f"assets/vis.mp4", out, fps=60)


main()
