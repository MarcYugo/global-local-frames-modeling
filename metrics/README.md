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

Download MS-RAFT project:

```bash
git clone https://github.com/cv-stuttgart/MS_RAFT.git
```

Based on the model file `sintel.pth` provided in the project, export a `test.pth` that is independent of `nn.DataParallel` and can run on a single sample. Then, use the following script to compute the video’s warping error.
```bash
bash run_e_warp.sh
```

#### Fliker value
```bash
bash run_flicker.sh
```

#### Acknowledgment

The implementation of SSIM refers to [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim).

The calculation of PSNR refers to [pytorch-tools](https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py).
