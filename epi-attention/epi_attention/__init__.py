from typing import NamedTuple, Tuple
import torch
import torch.nn.functional as F
from . import _C


def get_rays(poses: torch.Tensor, h: int, w: int, tan_fov: float):
    x, y = torch.meshgrid(
        torch.arange(w, device=poses.device),
        torch.arange(h, device=poses.device),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()

    cx = w * 0.5
    cy = h * 0.5

    focal = h * 0.5 / tan_fov

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx + 0.5) / focal,
                (y - cy + 0.5) / focal * -1,
            ],
            dim=-1,
        ),
        (0, 1),
        value=-1,
    ).to(poses.dtype)  # [hw, 3]

    rays_d = camera_dirs @ poses[:3, :3].transpose(0, 1)  # [hw, 3]
    rays_o = poses[:3, 3].unsqueeze(0).expand(h * w, -1)  # [hw, 3]
    return rays_o, rays_d


def transform_points_screen(points: torch.Tensor, matrix: torch.Tensor, H: int, W: int):
    # points in [P, 3]
    # matrix in [4, 4]
    assert points.ndim == 2 and points.shape[1] == 3, f"Unsupport point shape {points.shape}"
    P = points.shape[0]
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points = torch.cat([points, ones], dim=1)
    points = points @ matrix
    points = points[..., :2] / (points[..., 3:] + 0.0000001)
    y, x = points[..., 0:1], points[..., 1:2]
    y = (y + 1.0) * H * 0.5
    x = (x + 1.0) * W * 0.5
    points_out = torch.cat([y, x], dim=-1)
    return points_out


def get_points_flow(pts: torch.Tensor, center: torch.Tensor, imh: int, imw: int):
    # pts in [hw, 2]
    # center in [1, 2]
    pts_flow = pts - center
    pts_flow = pts_flow / pts_flow.norm(dim=-1, keepdim=True)

    # slope, intercept
    slope = pts_flow[..., 0:1] / pts_flow[..., 1:2]
    inter = pts[..., 0:1] - slope * pts[..., 1:2]

    left = slope * 0 + inter
    left_sane = (left <= imh) & (0 <= left)
    left = torch.cat([left, torch.zeros_like(left)], dim=-1)

    right = slope * imw  + inter
    right_sane = (right <= imh) & (0 <= right)
    right = torch.cat([right, torch.ones_like(right) * imw], dim=-1)

    top = (0 - inter) / slope
    top_sane = (top <= imw) & (0 <= top)
    top = torch.cat([torch.zeros_like(top), top], dim=-1)

    bottom = (imh - inter) / slope
    bottom_sane = (bottom <= imw) & (0 <= bottom)
    bottom = torch.cat([torch.ones_like(bottom) * imh, bottom], dim=-1)

    # find intersection of lines
    points_one = torch.zeros_like(left)
    points_two = torch.zeros_like(left)

    # collect points from [left, right, bottom, top] in sequence
    points_one = torch.where(left_sane.repeat(1,2), left, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,2)
    points_one = torch.where(right_sane.repeat(1,2) & points_one_zero, right, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,2)
    points_one = torch.where(bottom_sane.repeat(1,2) & points_one_zero, bottom, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,2)
    points_one = torch.where(top_sane.repeat(1,2) & points_one_zero, top, points_one)

    # collect points from [top, bottom, right, left] in sequence (opposite)
    points_two = torch.where(top_sane.repeat(1,2), top, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,2)
    points_two = torch.where(bottom_sane.repeat(1,2) & points_two_zero, bottom, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,2)
    points_two = torch.where(right_sane.repeat(1,2) & points_two_zero, right, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,2)
    points_two = torch.where(left_sane.repeat(1,2) & points_two_zero, left, points_two)

    # if source point lies inside target screen (find only one intersection)
    if (imh >= center[0, 0] >= 0) and (imw >= center[0, 1] >= 0):
        points_one_flow = points_one - center
        points_one_flow_direction = (points_one_flow > 0)

        orig_flow_direction = (pts_flow > 0)

        # if flow direction is same as orig flow direction, pick points_one, else points_two
        points_one_alinged = (points_one_flow_direction == orig_flow_direction).all(dim=-1).unsqueeze(-1).repeat(1,2)
        points_one = torch.where(points_one_alinged, points_one, points_two)

        # points two is source camera center
        points_two = points_two * 0 + center
    
    points_one = (points_one - 0.5).reshape(imh, imw, 2).fliplr().reshape(-1, 2).int()
    points_two = (points_two - 0.5).reshape(imh, imw, 2).fliplr().reshape(-1, 2).int()
    
    return points_one, points_two


class EpiAttentionSettings(NamedTuple):
    image_height: int
    image_width: int
    tan_fov: float # math.tan(fov * 0.5)
    projmatrix: torch.Tensor # w2c @ proj [opengl]
    unproj_depth: float
    dilate_size: int


class EpiAttention:
    def __init__(self, attn_settings: EpiAttentionSettings):
        super().__init__()
        self.attn_settings = attn_settings

    def compute_point(self, pose: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # pose in [4, 4]
        attn_settings = self.attn_settings

        rays_o, rays_d = get_rays(
            pose,
            attn_settings.image_height,
            attn_settings.image_width,
            attn_settings.tan_fov
        ) # [hw, 3]
        pts_world = rays_o + rays_d * attn_settings.unproj_depth # [hw, 3]

        pts_screen = transform_points_screen(
            pts_world,
            attn_settings.projmatrix,
            attn_settings.image_height,
            attn_settings.image_width
        ) # [hw, 2]
        center_screen = transform_points_screen(
            pose[:3, 3].unsqueeze(0),
            attn_settings.projmatrix,
            attn_settings.image_height,
            attn_settings.image_width
        ) # [1, 2]

        points_one, points_two = get_points_flow(
            pts_screen,
            center_screen,
            attn_settings.image_height,
            attn_settings.image_width
        ) # [hw, 2], [hw, 2]

        return points_one, points_two
    
    def compute_attention_mask(self, pose: torch.Tensor) -> torch.Tensor:
        # pose in [4, 4]
        attn_settings = self.attn_settings
        points_one, points_two = self.compute_point(pose)
        
        if attn_settings.dilate_size < 0:
            raise ValueError("Dilate size must be larger than 0")

        # Restructure arguments the way that the C++ lib expects them
        args = (
            points_one,
            points_two,
            attn_settings.image_height,
            attn_settings.image_width,
            attn_settings.dilate_size
        )

        # Invoke C++/CUDA rasterizer
        mask = _C.compute_attention_mask(*args)
        return mask # [hw, hw]
