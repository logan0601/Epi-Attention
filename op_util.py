from torch import Tensor

import numpy as np
import torch
import torch.nn.functional as F

from skimage.draw import line
from pytorch3d.renderer import NDCMultinomialRaysampler, PerspectiveCameras, ray_bundle_to_ray_points
from pytorch3d.utils import cameras_from_opencv_projection


def compute_points(src_cam: PerspectiveCameras, tgt_cam: PerspectiveCameras, imh: int, imw: int):
    # generates raybundle using camera intrinsics and extrinsics
    src_ray_bundle = NDCMultinomialRaysampler(
        image_width=imw,
        image_height=imh,
        n_pts_per_ray=1,
        min_depth=1.0,
        max_depth=1.0,
    )(src_cam)
    
    # get points in world space (at fixed depth)
    src_depth = 1.5 * torch.ones((1, imh, imw, 1), dtype=torch.float32, device=src_cam.device)

    pts_world = ray_bundle_to_ray_points(
      src_ray_bundle._replace(lengths=src_depth)
    ).squeeze(-2)
    # print(f"world points bounds: {pts_world.reshape(-1,3).min(dim=0)[0]} to {pts_world.reshape(-1,3).max(dim=0)[0]}")

    # move source points to target screen space
    tgt_pts_screen = tgt_cam.transform_points_screen(pts_world.squeeze(), image_size=(imh, imw))

    # move source camera center to target screen space
    src_center_tgt_screen = tgt_cam.transform_points_screen(src_cam.get_camera_center(), image_size=(imh, imw)).squeeze()

    # build epipolar mask (draw lines from source camera center to source points in target screen space)
    # start: source camera center, end: source points in target screen space

    # get flow of points 
    center_to_pts_flow = tgt_pts_screen[...,:2] - src_center_tgt_screen[...,:2]

    # normalize flow
    center_to_pts_flow = center_to_pts_flow / center_to_pts_flow.norm(dim=-1, keepdim=True)

    # get slope and intercept of lines
    slope = center_to_pts_flow[:,:,0:1] / center_to_pts_flow[:,:,1:2]
    intercept = tgt_pts_screen[:,:, 0:1] - slope * tgt_pts_screen[:,:, 1:2]

    # find intersection of lines with tgt screen (x = 0, x = imw, y = 0, y = imh)
    left = slope * 0 + intercept
    left_sane = (left <= imh) & (0 <= left)
    left = torch.cat([left, torch.zeros_like(left)], dim=-1)

    right = slope * imw + intercept
    right_sane = (right <= imh) & (0 <= right)
    right = torch.cat([right, torch.ones_like(right) * imw], dim=-1)

    top = (0 - intercept) / slope
    top_sane = (top <= imw) & (0 <= top)
    top = torch.cat([torch.zeros_like(top), top], dim=-1)

    bottom = (imh - intercept) / slope
    bottom_sane = (bottom <= imw) & (0 <= bottom)
    bottom = torch.cat([torch.ones_like(bottom) * imh, bottom], dim=-1)

    # find intersection of lines
    points_one = torch.zeros_like(left)
    points_two = torch.zeros_like(left)

    # collect points from [left, right, bottom, top] in sequence
    points_one = torch.where(left_sane.repeat(1,1,2), left, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_one = torch.where(right_sane.repeat(1,1,2) & points_one_zero, right, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_one = torch.where(bottom_sane.repeat(1,1,2) & points_one_zero, bottom, points_one)

    points_one_zero = (points_one.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_one = torch.where(top_sane.repeat(1,1,2) & points_one_zero, top, points_one)

    # collect points from [top, bottom, right, left] in sequence (opposite)
    points_two = torch.where(top_sane.repeat(1,1,2), top, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_two = torch.where(bottom_sane.repeat(1,1,2) & points_two_zero, bottom, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_two = torch.where(right_sane.repeat(1,1,2) & points_two_zero, right, points_two)

    points_two_zero = (points_two.sum(dim=-1) == 0).unsqueeze(-1).repeat(1,1,2)
    points_two = torch.where(left_sane.repeat(1,1,2) & points_two_zero, left, points_two)

    # if source point lies inside target screen (find only one intersection)
    if (imh >= src_center_tgt_screen[0] >= 0) and (imw >= src_center_tgt_screen[1] >= 0):
        points_one_flow = points_one - src_center_tgt_screen[:2]
        points_one_flow_direction = (points_one_flow > 0)

        orig_flow_direction = (center_to_pts_flow > 0)

        # if flow direction is same as orig flow direction, pick points_one, else points_two
        points_one_alinged = (points_one_flow_direction == orig_flow_direction).all(dim=-1).unsqueeze(-1).repeat(1,1,2)
        points_one = torch.where(points_one_alinged, points_one, points_two)

        # points two is source camera center
        points_two = points_two * 0 + src_center_tgt_screen[:2]
    
    # quantize points to pixel indices
    points_one = (points_one - 0.5).reshape(-1,2).long()
    points_two = (points_two - 0.5).reshape(-1,2).long()
    
    return points_one, points_two


def compute_epipolar_mask(src_cam: PerspectiveCameras, tgt_cam: PerspectiveCameras, imh: int, imw: int):
    points_one, points_two = compute_points(src_cam, tgt_cam, imh, imw)
    
    # build epipolar mask
    attention_mask = torch.zeros((imh * imw, imh, imw), dtype=torch.bool, device=src_cam.device)

    points_one = points_one.cpu().numpy()
    points_two = points_two.cpu().numpy()
    
    # iterate over points_one and points_two together and draw lines
    for idx, (p1, p2) in enumerate(zip(points_one, points_two)):
        # skip out of bounds points
        if p1.sum() == 0 and p2.sum() == 0:
            continue
        
        # draw lines with mask dilation (from all neighbors of p1 to neighbors of p2)
        rrs, ccs = [], []
        for dx, dy in [(0,0), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]:  # 8 neighbors
            _p1 = [min(max(p1[0] + dy, 0), imh - 1), min(max(p1[1] + dx, 0), imw - 1)]
            _p2 = [min(max(p2[0] + dy, 0), imh - 1), min(max(p2[1] + dx, 0), imw - 1)]
            rr, cc = line(int(_p1[1]), int(_p1[0]), int(_p2[1]), int(_p2[0]))
            rrs.append(rr); ccs.append(cc)
        rrs, ccs = np.concatenate(rrs), np.concatenate(ccs)
        attention_mask[idx, rrs.astype(np.int32), ccs.astype(np.int32)] = True

    # reshape to (imh, imw, imh, imw)
    attention_mask = attention_mask.reshape(imh * imw, imh * imw)

    return attention_mask


def get_opencv_from_blender(matrix_world: Tensor, fov: float, image_size: int):
    # convert matrix_world to opencv format extrinsics
    opencv_world_to_cam = matrix_world.float().inverse()
    opencv_world_to_cam[1, :] *= -1
    opencv_world_to_cam[2, :] *= -1
    R, T = opencv_world_to_cam[:3, :3], opencv_world_to_cam[:3, 3]
    R, T = R.unsqueeze(0), T.unsqueeze(0)
    
    # convert fov to opencv format intrinsics
    focal = 1 / np.tan(fov / 2)
    intrinsics = np.diag(np.array([focal, focal, 1])).astype(np.float32)
    opencv_cam_matrix = torch.from_numpy(intrinsics).unsqueeze(0).float()
    opencv_cam_matrix[:, :2, -1] += torch.tensor([image_size / 2, image_size / 2])
    opencv_cam_matrix[:, [0,1], [0,1]] *= image_size / 2
    opencv_cam_matrix = opencv_cam_matrix.to(device=matrix_world.device)

    return R, T, opencv_cam_matrix


def get_blender_camera(pos: Tensor):
    # pos in [B, 3]
    z = pos / torch.linalg.norm(pos, dim=1, keepdims=True)

    up = torch.tensor([[0, 0, 1]]).to(pos)
    x = torch.cross(up, z, dim=1)
    x = x / torch.linalg.norm(x, dim=1, keepdims=True)

    y = torch.cross(z, x, dim=1)
    y = y / torch.linalg.norm(y, dim=1, keepdims=True)

    cam = torch.zeros((pos.shape[0], 4, 4)).to(pos)
    cam[:, :3] = torch.stack([x, y, z, pos], dim=2)
    cam[:, 3, 3] = 1
    return cam


def get_epipolar_mask(cam_poses: Tensor, fov: float, image_size: int, scale: int):
    V, _, _ = cam_poses.shape
    cam_poses = cam_poses.float()

    with torch.no_grad():
        attention_masks = torch.ones(V, V, image_size ** 2, image_size ** 2, device=cam_poses.device, dtype=torch.bool)

        cams = []
        for i in range(V):
            R, T, intrinsics = get_opencv_from_blender(cam_poses[i], fov, image_size)
            cam = cameras_from_opencv_projection(R, T, intrinsics, torch.tensor([image_size, image_size]).float().unsqueeze(0))
            cams.append(cam)
        
        for i in range(V):
            for j in range(V):
                if i == j: continue
                mask = compute_epipolar_mask(cams[i], cams[j], image_size // scale, image_size // scale)
                if scale > 1:
                    attention_masks[i, j] = F.interpolate(
                        mask[None, None].to(torch.uint8), size=(image_size ** 2, image_size ** 2))[0, 0].bool()
                else:
                    attention_masks[i, j] = mask

        # attention_masks = einops.rearrange(attention_masks, "v1 v2 h w -> (v1 h) (v2 w)")
        return attention_masks.detach()
