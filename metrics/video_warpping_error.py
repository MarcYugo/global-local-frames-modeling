import sys
import os

# 将RAFT/core目录添加到sys.path中
sys.path.append('core')

# 导入RAFT模型和其他辅助函数
from msraft import MS_RAFT
from utils import flow_viz
from utils.utils import InputPadder

from torchvision.io.video import read_video
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def cpy_eval_args_to_config(args):
    config = {}
    config["model"] = args.ckpt_path
    config["warm"] = args.warm
    config["iters"] = args.iters
    # config["dataset"] = args.dataset
    config["mixed_precision"] = args.mixed_precision
    config["lookup"] = {}
    config["lookup"]["pyramid_levels"] = args.lookup_pyramid_levels
    config["lookup"]["radius"] = args.lookup_radius
    config["cuda_corr"] = args.cuda_corr

    return config

def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1", "True")

def load_model(args, config):
    # model = torch.nn.DataParallel(MS_RAFT(config))
    model = MS_RAFT(config)
    model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
    # model = model.module
    # print(model, sum(p.numel() for p in model.parameters())/(10**5))
    # torch.save(model.state_dict(), 'test.pth')
    # assert 1==2
    model = model.to(args.device)
    model.eval()
    return model

def compute_flow(model, video, args):
    flows = []
    # video = video / 255.0
    pre_frm = video[0:1].to(args.device)
    for i in tqdm(range(1, video.size(0))):
        cur_frm = video[i:i+1].to(args.device)
        # print(pre_frm.shape, cur_frm.shape, pre_frm.device, cur_frm.device)
        _, flow_tmp = model(pre_frm, cur_frm, iters=args.iters, test_mode=True)
        # print(i, flow_tmp.shape, flow_tmp.device)
        # assert 1==2
        flows.append(flow_tmp.detach().cpu())
        pre_frm = cur_frm
        torch.cuda.empty_cache()
    return flows

def get_grid(H,W, device):
    x = torch.linspace(0, 1, W)
    y = torch.linspace(0, 1, H)
    grid_x, grid_y = torch.meshgrid(x, y)
    grid = torch.stack([grid_x, grid_y], dim=0).to(device)
    grid = grid.permute(0, 2, 1)
    grid[0] *= W
    grid[1] *= H
    return grid

def warp_video_with_flow(video, flows, args):
    _,_,h,w = video.shape
    grid = get_grid(h,w,args.device)
    warped_frms = [video[0:1].float().to(args.device),]
    # pre_frm = video[0]
    for i in tqdm(range(video.shape[0]-1)):
        cur_frm = video[i:i+1].float().to(args.device)
        flow = flows[i].squeeze().to(args.device)
        # print(grid.shape, cur_frm.shape, flow.shape)
        # assert 1==2
        grid_1to2 = grid + flow
        grid_norm_1to2 = grid_1to2.clone()
        grid_norm_1to2[0, ...] = 2 * grid_norm_1to2[0, ...] / (w - 1) - 1
        grid_norm_1to2[1, ...] = 2 * grid_norm_1to2[1, ...] / (h - 1) - 1
        grid_norm_1to2 = grid_norm_1to2.unsqueeze(0).permute(0, 2, 3, 1)
        warped_frm = F.grid_sample(cur_frm, grid_norm_1to2, mode=args.intepolate_mode, padding_mode='zeros')
        warped_frms.append(warped_frm)
    warped_frms = torch.cat(warped_frms)
    return warped_frms

def video_warpping_error(video, warped_frms):
    return torch.mean((video - warped_frms)**2)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--intepolate_mode', type=str, default='bilinear')
    
    parser.add_argument('--warm', action='store_true', help="use warm start", default=True)
    parser.add_argument('--iters', type=int, nargs='+', default=[10, 15, 20])
    parser.add_argument('--lookup_pyramid_levels', type=int, default=2)
    parser.add_argument('--lookup_radius', default=4)
    parser.add_argument('--mixed_precision', help='use mixed precision', type=str2bool, default=True)
    parser.add_argument('--cuda_corr', help="use cuda kernel for on-demand cost computation", action='store_true', default=False)

    args = parser.parse_args()
    
    config = cpy_eval_args_to_config(args)
    model = load_model(args, config).to(args.device)
    # print(next(model.parameters()).device)
    video, _, _ = read_video(args.video, output_format='TCHW')
    # video = video.to(args.device)
    # print(video.device)
    print('computing optical flow ...')
    flows = compute_flow(model, video, args)
    print('get warped frames ...')
    warped_frms = warp_video_with_flow(video, flows, args)
    print('computing warpping error ...')
    # print(video.device, warped_frms.device)
    e_warp = video_warpping_error(video.float().to(args.device)/255.0, warped_frms/255.0)
    print(f'Average warpping error of {args.video} is {e_warp.item()}')