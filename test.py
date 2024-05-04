import torch
import torch.nn.functional as F
from op_util import (
    get_blender_camera,
    get_pytorch3d_camera,
    compute_epipolar_mask,
    compute_points,
)
from epi_util import get_epi_attention
import math
from time import time
from tqdm import tqdm


def main():
    fovy = math.radians(39.6)
    size = 64
    N = 2
    nTest = 100

    spad_time, our_time = 0, 0
    pt_err, final_err = 0, 0
    for _ in tqdm(range(nTest)):
        elev = torch.rand(1).repeat(N).cuda() * math.pi / 6
        azim = torch.rand(N).cuda() * math.pi * 2 - math.pi
        pos = torch.stack(
            [
                1.5 * torch.cos(elev) * torch.cos(azim),
                1.5 * torch.cos(elev) * torch.sin(azim),
                1.5 * torch.sin(elev)
            ],
            dim=-1
        )
        bcam = get_blender_camera(pos)
        pcam = [get_pytorch3d_camera(bcam[i], fovy, size) for i in range(N)]
        
        src_pt1, src_pt2 = compute_points(pcam[0], pcam[1], size, size)
        tgt_pt1, tgt_pt2 = get_epi_attention(bcam[1], fovy, size).compute_point(bcam[0])
        pt_err += F.mse_loss(src_pt1.float(), tgt_pt1.float()) + F.mse_loss(src_pt2.float(), tgt_pt2.float())

        s = time()
        src_mask = compute_epipolar_mask(pcam[0], pcam[1], size, size)
        spad_time += time() - s

        s = time()
        tgt_mask = get_epi_attention(bcam[1], fovy, size).compute_attention_mask(bcam[0])
        our_time += time() - s

        final_err += (src_mask != tgt_mask).int().sum() / (size ** 4)

    print(f"[TIME] SPAD:{spad_time:.2f} Our:{our_time:.2f}")
    print(f"[ERROR] PTS:{pt_err:.4f} Final:{final_err:.4f}")


if __name__ == "__main__":
    main()
