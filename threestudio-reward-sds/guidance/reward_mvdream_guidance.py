from dataclasses import dataclass
import numpy as np
from mvdream.ldm.modules.diffusionmodules.util import extract_into_tensor
import threestudio
import torch.nn.functional as F
from mvdream.camera_utils import normalize_camera
from mvdream.model_zoo import build_model
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C
from threestudio.utils.typing import *
import torch
from .utils import config_to_reward_strategy, config_to_reward_model
from .reward_strategy import NoRewardStrategy

@threestudio.register("reward-mvdream-multiview-diffusion-guidance")
class RewardMultiviewDiffusionGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        model_name: str = (
            "sd-v2.1-base-4view"  # Check mvdream.model_zoo.PRETRAINED_MODELS
        )
        ckpt_path: Optional[
            str
        ] = None  # Path to local checkpoint (None for loading from url)
        guidance_scale: float = 50.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_view: int = 4
        image_size: int = 256
        recon_loss: bool = True
        recon_std_rescale: float = 0.5

        reward_strategy: str = "none"
        reward_model: str = "image-reward"
        n_noises: int = 6
        reward_duration: float =  0.5
        num_of_training_steps: int = 10000

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Diffusion ...")

        self.model = build_model(self.cfg.model_name, ckpt_path=self.cfg.ckpt_path).to(
            self.device
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None

        self.reward_strategy = config_to_reward_strategy(self.cfg)
        self.reward_model = config_to_reward_model(self.cfg)
        self.last_reward_step = int(self.cfg.num_of_training_steps * self.cfg.reward_duration)
        self.cur_step = 0
        # Used when last_reward_step has passed
        self.default_no_reward_strategy = NoRewardStrategy()
        threestudio.info(f"Loaded Multiview Diffusion!")

    def get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"]
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(
                f"Unknown camera_condition_type={self.cfg.camera_condition_type}"
            )
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs)
        )
        return latents  # [B, 4, 32, 32] Latent space image

    def decode_latents(self, latents):
        imgs = self.model.decode_first_stage(latents)
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        return imgs

    def diffusion_denoising_loop(self, noisy_latent, cur_t, context, num_steps=15):
        """
        Perform the reverse diffusion process to denoise a latent and recover x0.

        Args:
            noisy_latent (torch.Tensor): The starting latent tensor at timestep `cur_t`.
            cur_t (int): The starting timestep.
            context (dict): The conditioning context for the model.
            num_steps (int): The number of denoising steps to
        Returns:
            torch.Tensor: The denoised latent x0.
        """
        x = noisy_latent
        full_timesteps = self.model.num_timesteps
        # Build a fixed schedule over the full chain.
        # Here we generate num_steps timesteps evenly spaced between 0 and T-1.
        full_schedule = np.linspace(0, full_timesteps - 1, num_steps, dtype=int)
        # Reverse it so that it goes from high to low (e.g., [999, 932, ..., 0]).
        full_schedule = full_schedule[::-1].tolist()

        # Now, if cur_t is lower than the highest scheduled timestep, filter to only those <= cur_t.
        scheduled_steps = [t for t in full_schedule if t <= cur_t]

        # If cur_t is not in the schedule (i.e. we’re starting later than the schedule's first point),
        # Prepend it to simulate a jump.
        if not scheduled_steps or scheduled_steps[0] != cur_t:
            scheduled_steps = [cur_t] + scheduled_steps
        for i in range(len(scheduled_steps) - 1):
            t = scheduled_steps[i]
            t_next = scheduled_steps[i + 1]

            # Create tensors for the current and next timesteps.
            t_tensor = torch.tensor([t], device=x.device, dtype=torch.long)
            t_expand = t_tensor.repeat(context['context'].shape[0])

            t_next_tensor = torch.tensor([t_next], device=x.device, dtype=torch.long)
            x_input = torch.cat([x] * 2)
            # Use the model to predict the noise at the current timestep.
            noise_pred = self.model.apply_model(x_input, t_expand, context)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            # Compute the predicted x0 from the noisy latent and the noise prediction.
            x0_pred = self.model.predict_start_from_noise(
                x_input[:x_input.shape[0] // 2],
                t_expand[:t_expand.shape[0] // 2],
                noise_pred
            )

            # Get the coefficients for the current timestep.
            sqrt_at = extract_into_tensor(self.model.sqrt_alphas_cumprod, t_tensor, x.shape)
            sqrt_1_at = extract_into_tensor(self.model.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)

            # Get the coefficients for the next timestep.
            sqrt_at_next = extract_into_tensor(self.model.sqrt_alphas_cumprod, t_next_tensor, x.shape)
            sqrt_1_at_next = extract_into_tensor(self.model.sqrt_one_minus_alphas_cumprod, t_next_tensor, x.shape)

            # Compute the predicted noise epsilon.
            eps_pred = (x - sqrt_at * x0_pred) / sqrt_1_at

            # Deterministic DDIM update:
            x = sqrt_at_next * x0_pred + sqrt_1_at_next * eps_pred

        return x

    @torch.no_grad()
    def explore_direction(
        self,
        latents,  # Noisy latents
        prompt,
        cur_t,
        context,
        num_inference_steps=15
    ):
        x0_latents = self.diffusion_denoising_loop(latents, cur_t, context, num_inference_steps)
        images = self.decode_latents(x0_latents).to(torch.float32)
        scores = self.reward_model.compute_scores(images, prompt)
        return torch.tensor(scores).mean().to(self.device)

    def _has_last_reward_step_passed(self):
        return self.cur_step >= self.last_reward_step

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        **kwargs,
    ):
        camera = c2w
        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )

        if input_is_latent:
            latents = rgb
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = (
                    F.interpolate(
                        rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
                    )
                    * 2
                    - 1
                )
            else:
                # Interp to 256x256 to be fed into vae.
                pred_rgb = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size, self.cfg.image_size),
                    mode="bilinear",
                    align_corners=False,
                )
                # Encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        # Sample timestep
        if timestep is None:
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [1],
                dtype=torch.long,
                device=latents.device,
            )
        else:
            assert 0 <= timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
        t_expand = t.repeat(text_embeddings.shape[0])

        cur_reward_strategy = self.reward_strategy if not self._has_last_reward_step_passed() else self.default_no_reward_strategy
        losses = torch.zeros(cur_reward_strategy.n_noises, device=latents.device)
        grads = torch.zeros(cur_reward_strategy.n_noises, device=latents.device)
        rewards = torch.zeros(cur_reward_strategy.n_noises, device=latents.device)
        with torch.no_grad():
            # Save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(camera)
                camera = camera.repeat(2, 1).to(text_embeddings)
                context = {
                    "context": text_embeddings,
                    "camera": camera,
                    "num_frames": self.cfg.n_view,
                }
            else:
                context = {"context": text_embeddings}
        for i in range(cur_reward_strategy.n_noises):
            # Predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # Add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.model.q_sample(latents, t, noise)

                # Avoid computing reward if not needed
                if not isinstance(cur_reward_strategy, NoRewardStrategy):
                    reward = self.explore_direction(latents_noisy, prompt_utils.prompt, t, context)
                    rewards[i] = reward
                # Pred noise
                latent_model_input = torch.cat([latents_noisy] * 2)
                noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

            # Perform guidance
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)  # Note: flipped compared to stable-dreamfusionn
            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            if self.cfg.recon_loss:
                # Reconstruct x0
                latents_recon = self.model.predict_start_from_noise(
                    latents_noisy, t, noise_pred
                )

                # Clip or rescale x0
                if self.cfg.recon_std_rescale > 0:
                    latents_recon_nocfg = self.model.predict_start_from_noise(
                        latents_noisy, t, noise_pred_text
                    )
                    latents_recon_nocfg_reshape = latents_recon_nocfg.view(
                        -1, self.cfg.n_view, *latents_recon_nocfg.shape[1:]
                    )
                    latents_recon_reshape = latents_recon.view(
                        -1, self.cfg.n_view, *latents_recon.shape[1:]
                    )
                    factor = (
                        latents_recon_nocfg_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8
                    ) / (latents_recon_reshape.std([1, 2, 3, 4], keepdim=True) + 1e-8)

                    latents_recon_adjust = latents_recon.clone() * factor.squeeze(
                        1
                    ).repeat_interleave(self.cfg.n_view, dim=0)
                    latents_recon = (
                        self.cfg.recon_std_rescale * latents_recon_adjust
                        + (1 - self.cfg.recon_std_rescale) * latents_recon
                    )

                loss = (
                    0.5
                    * F.mse_loss(latents, latents_recon.detach(), reduction="sum")
                    / latents.shape[0]
                )
                grad = torch.autograd.grad(loss, latents, retain_graph=True)[0]

            else:
                # Original SDS
                # w(t), sigma_t^2
                w = 1 - self.model.alphas_cumprod[t]
                grad = w * (noise_pred - noise)

                # CLIP grad for stable training?
                if self.grad_clip_val is not None:
                    grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
                grad = torch.nan_to_num(grad)

                target = (latents - grad).detach()
                loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]
            losses[i] = loss
            grads[i] = grad.norm()

        loss, grad = cur_reward_strategy.extract_loss(losses, grads, rewards)
        return {
            "loss_sds": loss,
            "grad_norm": grad,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.cur_step = global_step
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
