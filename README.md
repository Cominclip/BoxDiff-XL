# BoxDiff-XL

Improve [BoxDiff](https://github.com/showlab/BoxDiff) to Stable Diffusion XL

```shell
CUDA_VISIBLE_DEVICES=4 python run_sd_boxdiff.py --prompt "A rabbit wearing sunglasses looks very proud"  --P 0.2 --L 1 --seeds [1,2,3,4,5,6,7,8,9] --token_indices [2,4] --bbox [[67,87,366,512],[66,130,364,262]] --sd_xl True
```
