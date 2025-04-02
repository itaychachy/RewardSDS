import argparse
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from guidance import Guidance, GuidanceConfig
from utils import seed_everything
import re

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _arg_to_directory(arg: str) -> str:
    # Strip leading/trailing whitespace
    arg = arg.strip()
    # Replace spaces with underscores
    arg = arg.replace(" ", "_")
    # Replace hyphens with underscores
    arg = arg.replace("-", "_")
    # Remove common punctuation and special characters
    arg = re.sub(r'[^\w\-]', '', arg)
    # Optionally lowercase it
    return arg.lower()

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A white car and a red sheep")
    parser.add_argument(
        "--reward_strategy", type=str, default="min-max", choices=["min-max", "best", "weighted", "none"]
    )
    parser.add_argument("--n_noises", type=int, default=4)
    parser.add_argument("--reward_model", type=str, default="image-reward", choices=["image-reward", "aesthetic"])
    parser.add_argument(
        "--mode", type=str, default="sds", choices=["sds", "vsd", "sds-bridge"]
    )
    parser.add_argument(
        "--extra_src_prompt",
        type=str,
        default=", oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed",
    )
    parser.add_argument(
        "--extra_tgt_prompt",
        type=str,
        default=", detailed high resolution, high quality, sharp",
    )
    parser.add_argument("--init_image_fn", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    default_n_steps = 1000
    parser.add_argument("--n_steps", type=int, default=default_n_steps)
    # Used only with SDS-bridge
    parser.add_argument("--stage_two_start_step", type=int, default=default_n_steps // 2)
    args, _ = parser.parse_known_args()
    mode = args.mode
    default_cfg_scale = 100 if mode == "sds" else 7.5 if mode == "vsd" else 40  # sds-bridge
    parser.add_argument("--cfg_scale", type=float, default=default_cfg_scale)
    parser.add_argument("--method", type=float, default=default_cfg_scale)
    args = parser.parse_args()
    return args


def _decode_latent(guidance, latent):
        latent = latent.detach().to(device)
        with torch.no_grad():
            rgb = guidance.decode_latents(latent)
        rgb = rgb.float().cpu().permute(0, 2, 3, 1)
        rgb = rgb.permute(1, 0, 2, 3)
        rgb = rgb.flatten(start_dim=1, end_dim=2)
        return rgb

def run():
    args = _parse_args()
    init_image_fn = args.init_image_fn
    guidance = Guidance(
        config=GuidanceConfig(
            sd_pretrained_model_or_path="stabilityai/stable-diffusion-2-1-base",
            reward_strategy=args.reward_strategy,
            reward_model=args.reward_model,
            n_noises=args.n_noises,
            device=device
        ),
        use_lora=(args.mode == "vsd"),
    )

    if init_image_fn is not None:
        reference = torch.tensor(plt.imread(init_image_fn))[..., :3]
        reference = reference.permute(2, 0, 1)[None, ...]
        reference = reference.to(guidance.unet.device)

        reference_latent = guidance.encode_image(reference)
        im = reference_latent
    else:
        # Initialize with low-magnitude noise, zeros also works
        im = torch.randn((1, 4, 64, 64), device=guidance.unet.device)

    # You may want to change the `save_dir` and include information about reward model and the number of noises
    save_dir = "results/%s_gen/%s_reward_strategy/%s" % (
        _arg_to_directory(args.mode),
        _arg_to_directory(args.reward_strategy),
        _arg_to_directory(args.prompt)
    )
    print("Save dir:", save_dir, flush=True)
    os.makedirs(save_dir, exist_ok=True)

    seed_everything(args.seed)

    im.requires_grad_(True)
    im.retain_grad()

    im_optimizer = torch.optim.AdamW([im], lr=args.lr, betas=(0.9, 0.99), eps=1e-15)
    lora_optimizer = None
    if args.mode == "vsd":
        lora_optimizer = torch.optim.AdamW(
            [
                {"params": guidance.unet_lora.parameters(), "lr": 3e-4},
            ],
            weight_decay=0,
        )

    im_opts = []

    print(f"Generating image: {args.prompt}", flush=True)
    for step in tqdm(range(args.n_steps), desc=f"Optimizing image: mode={args.mode}, reward strategy={args.reward_strategy}"):
        guidance.config.guidance_scale = args.cfg_scale
        if args.mode == "sds":
            loss_dict = guidance.sds_loss(
                im=im, prompt=args.prompt, cfg_scale=args.cfg_scale, return_dict=True
            )
        elif args.mode == "vsd":
            loss_dict = guidance.vsd_loss(
                im=im, prompt=args.prompt, cfg_scale=7.5, return_dict=True
            )
            lora_loss = loss_dict["lora_loss"]
            lora_loss.backward()
            lora_optimizer.step()
            lora_optimizer.zero_grad()

        elif args.mode == "sds-bridge":
            if step < args.stage_two_start_step:
                loss_dict = guidance.sds_loss(
                    im=im, prompt=args.prompt, cfg_scale=args.cfg_scale, return_dict=True
                )
            else:
                loss_dict = guidance.bridge_stage_two(
                    im=im, prompt=args.prompt, cfg_scale=args.cfg_scale, return_dict=True
                )
        else:
            raise ValueError(args.mode)

        loss = loss_dict["loss"]

        loss.backward()
        im_optimizer.step()
        im_optimizer.zero_grad()

        if step % 10 == 0:
            decoded = _decode_latent(guidance, im.detach()).cpu().numpy()
            im_opts.append(decoded)
            plt.imsave(os.path.join(save_dir, "image.png"), decoded)

        if step % 50 == 0:
            # noinspection PyTypeChecker
            imageio.mimwrite(
                os.path.join(save_dir, "optimization_video.mp4"),
                np.stack(im_opts).astype(np.float32) * 255,
                fps=10,
                codec="libx264"
            )

if __name__ == '__main__':
    run()
