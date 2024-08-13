import torch
from torchvision.io.video import read_video
import torch.nn.functional as F

def calculate_psnr(src_frames, restruct_frames):
    '''
        src_frames: (B, H, W, C)
        restruct_frames: (B, H, W, C)
    '''
    # 计算重建前后视频帧的 MSE
    mse = (src_frames - restruct_frames)**2
    mse = torch.mean(mse, dim=(1,2,3))
    # 计算 PSNR
    max_value = torch.tensor(255.0).float().to(mse.device)
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse)) # (B,)
    return psnr

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
    # 按照 batch size 计算重建帧和原帧之间的 psnr
    scores = []
    video1 = video1.float()
    video2 = video2.float()
    for _, i in enumerate(range(0, len(video1), args.batch_size)):
        frms_clip1 = video1[i:i+args.batch_size].to(device)
        frms_clip2 = video2[i:i+args.batch_size].to(device)
        # print(frms_clip1.shape, frms_clip1.dtype)
        # assert 1==2
        psnr_score = calculate_psnr(frms_clip1, frms_clip2)
        scores.append(psnr_score)
    psnr_frms = torch.cat(scores, dim=0)
    if args.store_psnr_every_frame:
        torch.save(psnr_frms, f'{args.store_psnr_every_frame}/psnr_every_frame.pt')
    psnr = torch.mean(psnr_frms)
    print(f'video1 path:{args.video1_path}, video2 path:{args.video2_path}')
    print(f'PSNR of reconstruction video: {psnr.item()}')

if __name__ == '__main__':
    from argparse import ArgumentParser
    import os
    parser = ArgumentParser()
    parser.add_argument('video1_path', type=str)
    parser.add_argument('video2_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_frames', type=int)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--store_psnr_every_frame', type=str, default='./psnr_fold/')

    args = parser.parse_args()
    if not os.path.exists(args.store_psnr_every_frame):
        os.makedirs(args.store_psnr_every_frame, mode=0o0755, exist_ok=True)
    main(args)
    print('Done!')