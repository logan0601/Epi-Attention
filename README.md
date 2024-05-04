# Epi-Attention
CUDA Implementation of Computing Epipolar Attention Mask from [SPAD](https://github.com/yashkant/spad). Our implementation is 1000x faster than numpy-based implementation of SPAD.

## Installation

```shell
cd epi-attention;
pip install .
```

## Usage

```python
# Emit ray from src camera and project onto tgt camera screen
setting = EpiAttentionSettings(
    image_height=image_height,  # image height
    image_width=image_width,    # image width
    tan_fov=tan_fov,            # math.tan(0.5 * fov)
    projmatrix=full_proj_matrix,# full projection matrix of tgt camera
    unproj_depth=1.5,           # camera distance
    dilate_size=1               # dilated size [-d, d], 0 for none
)
attn = EpiAttention(setting)
mask = attn.compute_attention_mask(pose)    # src camera pose(c2w)
# return type [hw, hw] from src to tgt.
```

## Match
You can check the results of SPAD and our implementation by commands.
```python
python test.py
```

## Visualize
You can visualize the epipolar line by commands.
```python
python visualize.py
```
Please check `index` in visualize.py for using different images in assets.
