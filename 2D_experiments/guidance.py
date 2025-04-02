from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from jaxtyping import Float
from utils import (
    create_reward_strategy,
    create_reward_model,
    cleanup,
    tensor_from_indices
)
from reward_strategy import NoRewardStrategy


@dataclass
class GuidanceConfig:
    sd_pretrained_model_or_path: str = "runwayml/stable-diffusion-v2-1-base"
    sd_pretrained_model_or_path_lora: str = "stabilityai/stable-diffusion-2-1"

    num_inference_steps: int = 500
    min_step_ratio: float = 0.02
    max_step_ratio: float = 0.98

    src_prompt: str = ""
    tgt_prompt: str = ""

    guidance_scale: float = 30
    guidance_scale_lora: float = 1.0
    sdedit_guidance_scale: float = 15
    device: torch.device = torch.device("cuda")
    lora_n_timestamp_samples: int = 1

    sync_noise_and_t: bool = True
    lora_cfg_training: bool = True

    reward_strategy: str = 'min-max'
    reward_model: str = 'image-reward'
    n_noises: int = 8

# noinspection DuplicatedCode
class Guidance(object):

    dtype = torch.float32
    pipe = DiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base",
        torch_dtype=dtype
    ).to('cuda')

    scheduler = DDIMScheduler.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler", torch_dtype=dtype
    )

    # noinspection PyProtectedMember
    def __init__(self, config: GuidanceConfig, use_lora: bool = False):
        self.config = config
        self.device = torch.device(config.device)

        self.scheduler.set_timesteps(config.num_inference_steps)
        self.pipe.scheduler = self.scheduler

        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae

        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        ## construct text features beforehand.
        self.src_prompt = self.config.src_prompt
        self.tgt_prompt = self.config.tgt_prompt
        self.src_text_feature = None
        self.tgt_text_feature = None
        self.update_text_features(
            src_prompt=self.src_prompt, tgt_prompt=self.tgt_prompt
        )
        self.null_text_feature = self.encode_text("")

        self.reward_strategy = create_reward_strategy(config.reward_strategy, config.n_noises)
        self.reward_model = create_reward_model(config.reward_model)

        if use_lora:
            self.pipe_lora = DiffusionPipeline.from_pretrained(
                config.sd_pretrained_model_or_path_lora
            ).to(self.device).to(self.dtype)
            self.single_model = False
            del self.pipe_lora.vae
            del self.pipe_lora.text_encoder
            cleanup()
            self.vae_lora = self.pipe_lora.vae = self.pipe.vae
            self.unet_lora = self.pipe_lora.unet
            for p in self.unet_lora.parameters():
                p.requires_grad_(False)
            self.camera_embedding = TimestepEmbedding(in_channels=16, time_embed_dim=1280).to(self.device).to(self.dtype)
            self.unet_lora.class_embedding = self.camera_embedding

            self.scheduler_lora = DDIMScheduler.from_pretrained(
                "stabilityai/stable-diffusion-2-1", subfolder="scheduler", torch_dtype=self.dtype
            )
            self.pipe.scheduler = self.scheduler

            self.scheduler_lora.set_timesteps(config.num_inference_steps)
            self.pipe_lora.scheduler = self.scheduler_lora

            # set up LoRA layers
            lora_attn_procs = {}
            for name in self.unet_lora.attn_processors.keys():
                cross_attention_dim = (
                    None
                    if name.endswith("attn1.processor")
                    else self.unet_lora.config.cross_attention_dim
                )
                if name.startswith("mid_block"):
                    hidden_size = self.unet_lora.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(
                        reversed(self.unet_lora.config.block_out_channels)
                    )[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet_lora.config.block_out_channels[block_id]
                else:
                    hidden_size = None

                lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
                )

            self.unet_lora.set_attn_processor(lora_attn_procs)

            self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors).to(
                self.device
            )
            self.lora_layers._load_state_dict_pre_hooks.clear()
            self.lora_layers._state_dict_hooks.clear()


    def encode_image(self, img_tensor: Float[torch.Tensor, "B C H W"]):
        x = img_tensor
        x = 2 * x - 1
        x = x.float()
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor

    def encode_text(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

    def decode_latents(self, latents):
        latents = latents.to(self.dtype)
        latents = 1 / self.vae.config.scaling_factor * latents

        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        return images

    def update_text_features(self, src_prompt=None, tgt_prompt=None):
        if getattr(self, "src_text_feature", None) is None:
            assert src_prompt is not None
            self.src_prompt = src_prompt
            self.src_text_feature = self.encode_text(src_prompt)
        else:
            if src_prompt is not None and src_prompt != self.src_prompt:
                self.src_prompt = src_prompt
                self.src_text_feature = self.encode_text(src_prompt)

        if getattr(self, "tgt_text_feature", None) is None:
            assert tgt_prompt is not None
            self.tgt_prompt = tgt_prompt
            self.tgt_text_feature = self.encode_text(tgt_prompt)
        else:
            if tgt_prompt is not None and tgt_prompt != self.tgt_prompt:
                self.tgt_prompt = tgt_prompt
                self.tgt_text_feature = self.encode_text(tgt_prompt)

    # noinspection PyTypeChecker, PyUnresolvedReferences
    def sample_timestep(self, batch_size):
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)

        min_step = (
            1
            if self.config.min_step_ratio <= 0
            else int(len(timesteps) * self.config.min_step_ratio)
        )
        max_step = (
            len(timesteps)
            if self.config.max_step_ratio >= 1
            else int(len(timesteps) * self.config.max_step_ratio)
        )
        max_step = max(max_step, min_step + 1)
        idx = torch.randint(
            min_step,
            max_step,
            [batch_size],
            dtype=torch.long,
            device="cpu",
        )
        t = timesteps[idx].cpu()
        t_prev = timesteps[idx - 1].cpu()

        return t, t_prev

    @torch.no_grad()
    def explore_direction(
            self,
            latents,  # Noisy latents
            prompt,
            cur_t,
            embeddings,
            num_inference_steps=15,
            guidance_scale=7.5
    ):
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.scheduler.timesteps:
            if t > cur_t:
                continue

            latent_model_input = torch.cat([latents] * 2, dim=0)
            # Predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=embeddings
            ).sample

            # Perform guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
            )

            # Compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        images = self.decode_latents(latents)
        rewards = self.reward_model.compute_scores(images, prompt)
        return torch.tensor(rewards).to(self.device)

    # noinspection PyMethodMayBeStatic
    def _compute_losses(self, im, grad, batch_size):
        n = im.shape[0]
        grad = torch.nan_to_num(grad)
        target = (im - grad).detach()
        # Compute mse_loss for each added noise (n)
        mse_loss = 0.5 * F.mse_loss(im.float(), target, reduction='none') / n
        losses = (
            mse_loss
            .sum(dim=(1, 2, 3))  # Sum over spatial dims
            .view(n, batch_size)
            .sum(dim=1)  # Loss per batch
        )
        return losses

    def sds_loss(
        self,
        im,
        prompt=None,
        cfg_scale=100,
        return_dict=False
    ):
        # Process text
        self.update_text_features(tgt_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]
        t, _ = self.sample_timestep(batch_size)
        im = im.repeat_interleave(self.reward_strategy.n_noises, dim=0).to(self.dtype)
        new_batch_size = im.shape[0]
        noise = torch.randn_like(im)
        latents_noisy = self.scheduler.add_noise(im, noise, t)
        text_embeddings = torch.cat(
            tensors=[
                tgt_text_embedding.expand(new_batch_size, -1, -1),
                uncond_embedding.expand(new_batch_size, -1, -1)
            ],
            dim=0
        )

        rewards = None
        # Avoid exploring direction if the reward strategy is NoRewardStrategy
        if not isinstance(self.reward_strategy, NoRewardStrategy):
            rewards = self.explore_direction(latents_noisy, prompt, t, text_embeddings)
            indices = self.reward_strategy.extract_indices_to_consider(rewards)

            # Update variables according to indices
            im = tensor_from_indices(im, indices)
            noise = tensor_from_indices(noise, indices)
            latents_noisy = tensor_from_indices(latents_noisy, indices)
            new_batch_size = im.shape[0]
            text_embeddings = torch.cat(
                tensors=[
                    tgt_text_embedding.expand(new_batch_size, -1, -1),
                    uncond_embedding.expand(new_batch_size, -1, -1)
                ],
                dim=0
            )

        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        noise_pred = self.unet.forward(
            latent_model_input,
            t.to(self.device),
            encoder_hidden_states=text_embeddings,
        ).sample
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + cfg_scale * (
            noise_pred_text - noise_pred_uncond
        )
        w = 1 - self.scheduler.alphas_cumprod[t].to(self.device)
        grad = w * (noise_pred - noise)

        losses = self._compute_losses(im, grad, batch_size)
        loss = self.reward_strategy.extract_loss(losses, rewards)

        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t}
            return dic
        else:
            return loss


    @contextmanager
    def disable_unet_class_embedding(self, unet):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def vsd_loss(
        self,
        im,
        prompt=None,
        cfg_scale=100,
        return_dict=False,
    ):
        # Process text
        self.update_text_features(tgt_prompt=prompt)
        tgt_text_embedding = self.tgt_text_feature
        uncond_embedding = self.null_text_feature

        batch_size = im.shape[0]
        im = im.to(self.dtype)
        original_im = im  # Used for Losses computations later
        im = im.repeat_interleave(self.reward_strategy.n_noises, dim=0).to(self.dtype)
        new_batch_size = im.shape[0]
        camera_condition = torch.zeros([new_batch_size, 4, 4], device=self.device, dtype=self.dtype)

        with torch.no_grad():
            # Random timestamp
            t = torch.randint(
                20,
                980 + 1,
                [batch_size],
                dtype=torch.long,
                device=self.device,
            )

            noise = torch.randn_like(im)

            latents_noisy = self.scheduler.add_noise(im, noise, t)
            text_embeddings = torch.cat(
                tensors=[
                    tgt_text_embedding.expand(new_batch_size, -1, -1),
                    uncond_embedding.expand(new_batch_size, -1, -1)
                ],
                dim=0
            )

            rewards = None
            # Avoid exploring direction if the reward strategy is NoRewardStrategy
            if not isinstance(self.reward_strategy, NoRewardStrategy):
                rewards = self.explore_direction(latents_noisy, prompt, t, text_embeddings)
                indices = self.reward_strategy.extract_indices_to_consider(rewards)

                # Update variables according to the best and worst noises
                im = tensor_from_indices(im, indices)
                latents_noisy = tensor_from_indices(latents_noisy, indices)
                camera_condition = tensor_from_indices(camera_condition, indices)
                new_batch_size = im.shape[0]
                text_embeddings = torch.cat(
                    tensors=[
                        tgt_text_embedding.expand(new_batch_size, -1, -1),
                        uncond_embedding.expand(new_batch_size, -1, -1)
                    ],
                    dim=0
                )

            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            with self.disable_unet_class_embedding(self.unet) as unet:
                cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                noise_pred_pretrain = unet.forward(
                    latent_model_input,
                    t.to(self.device),
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                )

            # Use view-independent text embeddings in LoRA
            # Use view-independent text embeddings in LoRA
            noise_pred_est = self.unet_lora.forward(
                latent_model_input,
                t.to(self.device),
                encoder_hidden_states=torch.cat([tgt_text_embedding.expand(new_batch_size, -1, -1)] * 2),
                class_labels=torch.cat(
                    [
                        camera_condition.view(new_batch_size, -1),
                        camera_condition.view(new_batch_size, -1),
                    ],
                    dim=0,
                ),
                cross_attention_kwargs={"scale": 1.0},
            ).sample

        (
            noise_pred_pretrain_text,
            noise_pred_pretrain_uncond,
        ) = noise_pred_pretrain.sample.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + cfg_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )
        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latent_model_input * torch.cat([sigma_t] * latent_model_input.shape[0], dim=0).view(
                -1, 1, 1, 1
            ) + noise_pred_est * torch.cat([alpha_t] * latent_model_input.shape[0], dim=0).view(-1, 1, 1, 1)

        (
            noise_pred_est_camera,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_est = noise_pred_est_uncond + self.config.guidance_scale_lora * (
            noise_pred_est_camera - noise_pred_est_uncond
        )

        w = (1 - self.scheduler.alphas_cumprod[t.cpu()]).view(-1, 1, 1, 1).to(self.device).to(self.dtype)
        grad = w * (noise_pred_pretrain - noise_pred_est)

        losses = self._compute_losses(original_im.repeat_interleave(new_batch_size, dim=0), grad, batch_size)
        loss = self.reward_strategy.extract_loss(losses, rewards)

        loss_lora = self.train_lora(
            original_im,
            torch.cat(
                [text_embeddings[0], text_embeddings[-1]],
                dim=0
            ),
            camera_condition[0].unsqueeze(0)
        )
        if return_dict:
            dic = {"loss": loss, "lora_loss": loss_lora, "grad": grad, "t": t}
            return dic
        else:
            return loss

    def train_lora(
        self,
        latents: Float[torch.Tensor, "B 4 64 64"],
        text_embeddings: Float[torch.Tensor, "BB 77 768"],
        camera_condition: Float[torch.Tensor, "B 4 4"],
    ):
        batch_size = latents.shape[0]
        latents = latents.detach().repeat(self.config.lora_n_timestamp_samples, 1, 1, 1)

        t = torch.randint(
            int(self.scheduler_lora.num_train_timesteps * 0.0),
            int(self.scheduler_lora.num_train_timesteps * 1.0),
            [batch_size * self.config.lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        # Use view-independent text embeddings in LoRA
        text_embeddings_cond, _ = text_embeddings.chunk(2)
        if self.config.lora_cfg_training and np.random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        noise_pred = self.unet_lora.forward(
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings_cond.repeat(
                self.config.lora_n_timestamp_samples, 1, 1
            ),
            class_labels=camera_condition.view(batch_size, -1).repeat(
                self.config.lora_n_timestamp_samples, 1
            ),
            cross_attention_kwargs={"scale": 1.0},
        ).sample
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    def bridge_stage_two(
        self,
        im,
        prompt=None,
        cfg_scale=30,
        extra_tgt_prompts=", detailed high resolution, high quality, sharp",
        extra_src_prompts=", oversaturated, smooth, pixelated, cartoon, foggy, hazy, blurry, bad structure, noisy, malformed",
        return_dict=False,
    ):
        # Process text
        self.update_text_features(
            tgt_prompt=prompt + extra_tgt_prompts, src_prompt=prompt + extra_src_prompts
        )
        tgt_text_embedding = self.tgt_text_feature
        src_text_embedding = self.src_text_feature

        batch_size = im.shape[0]
        t, _ = self.sample_timestep(batch_size)
        im = im.repeat_interleave(self.reward_strategy.n_noises, dim=0).to(self.dtype)
        new_batch_size = im.shape[0]
        noise = torch.randn_like(im)
        latents_noisy = self.scheduler.add_noise(im, noise, t)
        text_embeddings = torch.cat(
            tensors=[
                tgt_text_embedding.expand(new_batch_size, -1, -1),
                src_text_embedding.expand(new_batch_size, -1, -1)
            ],
            dim=0
        )
        rewards = None
        # Avoid exploring direction if the reward strategy is NoRewardStrategy
        if not isinstance(self.reward_strategy, NoRewardStrategy):
            rewards = self.explore_direction(latents_noisy, prompt, t, text_embeddings)
            indices = self.reward_strategy.extract_indices_to_consider(rewards)

            # Update variables according to the best and worst noises
            im = tensor_from_indices(im, indices)
            latents_noisy = tensor_from_indices(latents_noisy, indices)
            new_batch_size = im.shape[0]
            text_embeddings = torch.cat(
                tensors=[
                    tgt_text_embedding.expand(new_batch_size, -1, -1),
                    src_text_embedding.expand(new_batch_size, -1, -1)
                ],
                dim=0
            )


        latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
        noise_pred = self.unet.forward(
            latent_model_input,
            t.to(self.device),
            encoder_hidden_states=text_embeddings,
        ).sample
        noise_pred_tgt, noise_pred_src = noise_pred.chunk(2)

        w = 1 - self.scheduler.alphas_cumprod[t].to(self.device)
        grad = w * cfg_scale * (noise_pred_tgt - noise_pred_src)

        losses = self._compute_losses(im, grad, batch_size)
        loss = self.reward_strategy.extract_loss(losses, rewards)

        if return_dict:
            dic = {"loss": loss, "grad": grad, "t": t}
            return dic
        else:
            return loss
