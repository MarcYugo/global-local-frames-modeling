import os
from argparse import ArgumentParser

web = 'hf-mirror.com' # https://huggingface.co/

def install_stable_diffusion_1_5(root):
    if not os.path.exists(root):
        # runwayml/stable-diffusion-v1-5
        os.makedirs(root, mode=0o0755, exist_ok=True)
        os.makedirs(f'{root}/feature_extractor', mode=0o0755, exist_ok=True)
        os.makedirs(f'{root}/safety_checker', mode=0o0755, exist_ok=True)
        os.makedirs(f'{root}/scheduler', mode=0o0755, exist_ok=True)
        os.makedirs(f'{root}/text_encoder', mode=0o0755, exist_ok=True)
        os.makedirs(f'{root}/tokenizer', mode=0o0755, exist_ok=True)
        os.makedirs(f'{root}/unet', mode=0o0755, exist_ok=True)
        os.makedirs(f'{root}/vae', mode=0o0755, exist_ok=True)
    shell_order = [
        f'wget -P {root}/feature_extractor https://{web}/runwayml/stable-diffusion-v1-5/blob/main/feature_extractor/preprocessor_config.json',
        f'wget -P {root}/safety_checker https://{web}/runwayml/stable-diffusion-v1-5/blob/main/safety_checker/config.json',
        f'wget -P {root}/safety_checker https://{web}/runwayml/stable-diffusion-v1-5/blob/main/safety_checker/pytorch_model.bin',
        f'wget -P {root}/scheduler https://{web}/runwayml/stable-diffusion-v1-5/blob/main/scheduler/scheduler_config.json ',
        f'wget -P {root}/text_encoder https://{web}/runwayml/stable-diffusion-v1-5/blob/main/text_encoder/config.json',
        f'wget -P {root}/text_encoder https://{web}/runwayml/stable-diffusion-v1-5/blob/main/text_encoder/pytorch_model.bin',
        f'wget -P {root}/tokenizer https://{web}/runwayml/stable-diffusion-v1-5/blob/main/tokenizer/merges.txt',
        f'wget -P {root}/tokenizer https://{web}/runwayml/stable-diffusion-v1-5/blob/main/tokenizer/special_tokens_map.json',
        f'wget -P {root}/tokenizer https://{web}/runwayml/stable-diffusion-v1-5/blob/main/tokenizer/tokenizer_config.json',
        f'wget -P {root}/tokenizer https://{web}/runwayml/stable-diffusion-v1-5/blob/main/tokenizer/vocab.json',
        f'wget -P {root}/unet https://{web}/runwayml/stable-diffusion-v1-5/blob/main/unet/config.json',
        f'wget -P {root}/unet https://{web}/runwayml/stable-diffusion-v1-5/blob/main/unet/diffusion_pytorch_model.bin',
        f'wget -P {root}/vae https://{web}/runwayml/stable-diffusion-v1-5/blob/main/vae/config.json',
        f'wget -P {root}/vae https://{web}/runwayml/stable-diffusion-v1-5/blob/main/vae/diffusion_pytorch_model.bin',
        f'wget -P {root} https://{web}/runwayml/stable-diffusion-v1-5/blob/main/model_index.json',
    ]
    os.system('\n'.join(shell_order))

def install_sd_controlnet_canny(root):
    # lllyasviel/sd-controlnet-canny
    if not os.path.exists(root):
        os.makedirs(f'{root}/')
    shell_order = [
        f'wget -P {root} https://{web}/lllyasviel/sd-controlnet-canny/resolve/main/config.json',
        f'wget -P {root} https://{web}/lllyasviel/sd-controlnet-canny/blob/main/diffusion_pytorch_model.bin',
    ]
    os.system('\n'.join(shell_order))

def install_blip(root):
    # Salesforce/blip-image-captioning-base
    if not os.path.exists(root):
        os.makedirs(f'{root}', mode=0o0755, exist_ok=True)
    shell_order = [
        f'wget -P {root} https://{web}/Salesforce/blip-image-captioning-base/blob/main/config.json',
        f'wget -P {root} https://{web}/Salesforce/blip-image-captioning-base/blob/main/preprocessor_config.json',
        f'wget -P {root} https://{web}/Salesforce/blip-image-captioning-base/blob/main/pytorch_model.bin',
        f'wget -P {root} https://{web}/Salesforce/blip-image-captioning-base/blob/main/special_tokens_map.json',
        f'wget -P {root} https://{web}/Salesforce/blip-image-captioning-base/blob/main/tokenizer.json',
        f'wget -P {root} https://{web}/Salesforce/blip-image-captioning-base/blob/main/tokenizer_config.json',
        f'wget -P {root} https://{web}/Salesforce/blip-image-captioning-base/blob/main/vocab.txt'
    ]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('models_dir', type=str, default='/hy-tmp/models/')
    args = parser.parse_args()
    install_stable_diffusion_1_5(f'{args.models_dir}/runwayml/stable-diffusion-v1-5/')
    install_sd_controlnet_canny(f'{args.models_dir}/lllyasviel/sd-controlnet-canny/')
    install_blip(f'{args.models_dir}/Salesforce/blip-image-captioning-base/')