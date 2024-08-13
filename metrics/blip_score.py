from transformers import AutoProcessor, BlipModel, BlipForConditionalGeneration, BlipForImageTextRetrieval
import torch
from torchvision.io.video import read_video
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

def calculate_blip_score(model, processor, frames, texts):
    '''
        model: CLIP model
        frames: video frames (B, C, H, W)
        texts: text list (tokens,)
    '''
    # 文本词向量化
    inputs = processor(images=frames, text=texts, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].to(frames.device)
    inputs['input_ids'] = inputs['input_ids'].to(frames.device)
    inputs['attention_mask'] = inputs['attention_mask'].to(frames.device)
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True, use_itm_head=False)
        score_each_frame = outputs['itm_score']
    return score_each_frame

def read_video_frames(video_path, n_frames=None):
    video, _, _ = read_video(video_path, output_format='TCHW')
    if n_frames is not None:
        video = video[:n_frames]
    return video

def read_prompt(text_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        cont = f.read().strip('\n')
    return [cont]

def load_model(device='cpu'):
    # model = BlipForConditionalGeneration.from_pretrained("/hy-tmp/models/blip-image-captioning-base")
    # processor = AutoProcessor.from_pretrained("/hy-tmp/models/blip-image-captioning-base")
    model = BlipForImageTextRetrieval.from_pretrained("/hy-tmp/models/blip-itm-base-coco")
    processor = AutoProcessor.from_pretrained("/hy-tmp/models/blip-itm-base-coco")
    model = model.to(device)
    return model, processor

def main(args):
    # 载入模型
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print(f'Device: {device}')
    model, processor = load_model(device=device)
    # 读取视频文件
    if args.n_frames:
        frames = read_video_frames(args.video_path, n_frames=args.n_frames)
    else:
        frames = read_video_frames(args.video_path)
    # 读取文本
    text = read_prompt(args.text_path)
    # 按照 batch_size 计算帧和文本之间的相似度
    scores = []
    for idx, i in enumerate(range(0, len(frames), args.batch_size)):
        frms_clip = frames[i:i+args.batch_size].to(device)
        bs_score = calculate_blip_score(model, processor, frms_clip, text)
        scores.append(bs_score)
    scores = torch.cat(scores,dim=0)
    if args.store_similarity is not None:
        torch.save(scores, f'{args.store_similarity}/sim_every_frame.pt')
    mean_score_all_frms = torch.mean(scores)
    print(f'video path: {args.video_path}')
    print('BLIP Score of this video and prompt: ', mean_score_all_frms.item())
        

if __name__ == '__main__':
    from argparse import ArgumentParser
    import os
    parser = ArgumentParser()
    parser.add_argument('video_path', type=str)
    parser.add_argument('text_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_frames', type=int)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--store_similarity', type=str, default='./blip_score_fold/')

    args = parser.parse_args()
    if not os.path.exists(args.store_similarity):
        os.makedirs(args.store_similarity, mode=0o0755, exist_ok=True)
    main(args)
    print('Done!')