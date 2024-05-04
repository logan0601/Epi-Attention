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

#ifndef ATTENTION_MODULE_H_INCLUDED
#define ATTENTION_MODULE_H_INCLUDED

#include <vector>
#include <functional>

namespace EpiAttentionModule
{
	void compute_mask(
		const int P,
		const int H,
		const int W,
		const int D,
		const int* pts_one,
		const int* pts_two,
		bool* out_mask);
};

#endif