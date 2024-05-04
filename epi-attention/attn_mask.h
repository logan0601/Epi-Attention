/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>


torch::Tensor ComputeAttentionMaskCUDA(
	const torch::Tensor& points_one,
	const torch::Tensor& points_two,
    const int image_height,
    const int image_width,
    const int dilate_size
);
