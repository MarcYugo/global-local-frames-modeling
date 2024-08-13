import torch
from torchvision.io.video import read_video
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import warnings
warnings.filterwarnings('ignore')

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def calculate_ssim(src_frames, restruct_frames, window_size=11, channel=3):
    '''
        src_frames: (B, H, W, C)
        restruct_frames: (B, H, W, C)
    '''
    src_frames = src_frames.permute(0, 3, 1, 2)
    restruct_frames = restruct_frames.permute(0, 3, 1, 2)

    window = create_window(window_size, channel).to(src_frames.device)
    mu1 = F.conv2d(src_frames, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(restruct_frames, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(src_frames*src_frames, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(restruct_frames*restruct_frames, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(src_frames*restruct_frames, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    ssim = ssim_map.mean(1).mean(1).mean(1) # 逐帧的平均 ssim
    return ssim

def read_video_frames(video_path, n_frames=None):
    video, _, _ = read_video(video_path)
    if n_frames is not None:
        video = video[:n_frames]
    return video

def main(args):
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    # 读取视频文件
    if args.n_frames:
        video1 = read_video_frames(args.video1_path, n_frames=args.n_frames)
        video2 = read_video_frames(args.video2_path, n_frames=args.n_frames)
    else:
        video1 = read_video_frames(args.video1_path)
        video2 = read_video_frames(args.video2_path)
    video1 = video1.float()
    video2 = video2.float()
    # 按照 batch size 计算重建帧和原帧之间的 ssim
    scores = []
    for _, i in enumerate(range(0, len(video1), args.batch_size)):
        frms_clip1 = video1[i:i+args.batch_size].to(device)
        frms_clip2 = video2[i:i+args.batch_size].to(device)
        ssim_score = calculate_ssim(frms_clip1, frms_clip2)
        scores.append(ssim_score)
    ssim_frms = torch.cat(scores, dim=0)
    if args.store_ssim_every_frame:
        torch.save(ssim_frms, f'{args.store_ssim_every_frame}/ssim_every_frame.pt')
    ssim = torch.mean(ssim_frms)
    print(f'reconstruction video path: {args.video1_path}, src video path: {args.video2_path}')
    print(f'SSIM of reconstruction video: {ssim.item()}')

if __name__ == '__main__':
    from argparse import ArgumentParser
    import os
    parser = ArgumentParser()
    parser.add_argument('video1_path', type=str)
    parser.add_argument('video2_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_frames', type=int)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--store_ssim_every_frame', type=str, default='./ssim_fold/')

    args = parser.parse_args()
    if not os.path.exists(args.store_ssim_every_frame):
        os.makedirs(args.store_ssim_every_frame, mode=0o0755, exist_ok=True)
    main(args)
    print('Done!')