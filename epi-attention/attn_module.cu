#include "attn_module.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


__global__ void computeCUDA(int P,
	const int H,
	const int W,
	const int D,
	const int* pts_one,
	const int* pts_two,
	bool* out_mask)
{
	auto idx = cg::this_grid().thread_rank();
	if(idx >= P)
		return;

	int S = H * W;
	
	int p1[2] = { pts_one[2 * idx], pts_one[2 * idx + 1] };
	int p2[2] = { pts_two[2 * idx], pts_two[2 * idx + 1] };
	if( (p1[0] + p1[1] == 0) && (p2[0] + p2[1] == 0) )
		return;
	
	for(int dx = -D; dx <= D; dx++)
	{
		for(int dy = -D; dy <= D; dy++)
		{
			// from skimage.draw.line
			int r = min(max(p1[1] + dx, 0), W - 1);
			int c = min(max(p1[0] + dy, 0), H - 1);
			int r2 = min(max(p2[1] + dx, 0), W - 1);
			int c2 = min(max(p2[0] + dy, 0), H - 1);

			int steep = 0, sr, sc, d, tmp;
			int dr = abs(r2 - r), dc = abs(c2 - c);

			if(c2 > c)
				sc = 1;
			else
				sc = -1;
			
			if(r2 > r)
				sr = 1;
			else
				sr = -1;
			
			if(dr > dc)
			{
				steep = 1;
				tmp = c; c = r; r = tmp;
				tmp = dc; dc = dr; dr = tmp;
				tmp = sc; sc = sr; sr = tmp;
			}
			d = 2 * dr - dc;

			for(int i = 0; i < dc; i++)
			{
				if(steep == 1)
					out_mask[idx * S + c * W + r] = true;
				else
					out_mask[idx * S + r * W + c] = true;
				
				while(d >= 0)
				{
					r = r + sr;
					d = d - (2 * dc);
				}
				c = c + sc;
				d = d + (2 * dr);
			}
			out_mask[idx * S + r2 * W + c2] = true;
		}
	}
}


void EpiAttentionModule::compute_mask(
	const int P,
	const int H,
	const int W,
	const int D,
	const int* pts_one,
	const int* pts_two,
	bool* out_mask)
{
	computeCUDA << <(P + 255) / 256, 256 >> >(
		P, H, W, D,
		pts_one,
		pts_two,
		out_mask
	);
}
