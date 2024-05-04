#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "attn_module.h"
#include <fstream>
#include <string>
#include <functional>


torch::Tensor ComputeAttentionMaskCUDA(
	const torch::Tensor& points_one,
	const torch::Tensor& points_two,
  const int image_height,
  const int image_width,
  const int dilate_size)
{
  if (points_one.ndimension() != 2 || points_one.size(1) != 2 || points_two.ndimension() != 2 || points_two.size(1) != 2) {
    AT_ERROR("Points must have dimensions (hw, 2)");
  }

  const int P = points_one.size(0);
  const int H = image_height;
  const int W = image_width;
  const int S = H * W;
  const int D = dilate_size;

  auto bool_opts = points_one.options().dtype(torch::kBool);
  torch::Tensor out_mask = torch::full({P, S}, false, bool_opts);

  if(P != 0)
  {
	  EpiAttentionModule::compute_mask(
	    P, H, W, D,
      points_one.contiguous().data<int>(),
      points_two.contiguous().data<int>(),
      out_mask.contiguous().data<bool>()
    );
  }
  return out_mask;
}
