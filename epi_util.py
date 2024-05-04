import torch
import math
from epi_attention import EpiAttentionSettings, EpiAttention


def get_epi_attention(cam: torch.Tensor, fov: float, size: int):
    tan_fov = math.tan(fov * 0.5)
    znear = 0.1
    zfar = 1000

    projmatrix = torch.zeros((4, 4), dtype=torch.float32).cuda()
    projmatrix[0, 0] = 1 / tan_fov
    projmatrix[1, 1] = 1 / tan_fov
    projmatrix[2, 2] = (zfar + znear) / (zfar - znear)
    projmatrix[3, 2] = - (zfar * znear) / (zfar - znear)
    projmatrix[2, 3] = 1

    world_view_transform = torch.zeros((4, 4), dtype=torch.float32).cuda()
    world_view_transform[:3, :3] = cam[:3, :3].transpose(0, 1)
    world_view_transform[:3, 3] = -cam[:3, :3].transpose(0, 1) @ cam[:3, 3]
    world_view_transform[3, 3] = 1.0
    full_proj_matrix = world_view_transform.transpose(0, 1) @ projmatrix

    setting = EpiAttentionSettings(
        image_height=size,
        image_width=size,
        tan_fov=tan_fov,
        projmatrix=full_proj_matrix,
        unproj_depth=1.5,
        dilate_size=1
    )
    attn = EpiAttention(setting)
    return attn
