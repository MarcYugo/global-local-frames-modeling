### Metrics
#### PSNR

```bash
bash run_psnr.sh
```

#### SSIM

```bash
bash run_ssim.sh
```

#### BLIP Score

```bash
bash run_blip_score.sh
```

#### Warping error

Files *run_inference.sh*, *inference_video.py* and *inference_video_back.py* should be added into the project [Memflow](https://github.com/DQiaole/MemFlow).

```bash
bash run_inference.sh
```

#### Acknowledgment

The implementation of SSIM refers to [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim).

The calculation of PSNR refers to [pytorch-tools](https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py).
