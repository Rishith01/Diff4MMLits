"""
train.py

Core diffusion utilities for training (forward process, loss, and schedules)
used in the Enhanced Chunked Conditional Diffusion Training pipeline.

Replaces the minimal DDPM wrapper with one that matches:
 - cosine beta schedule
 - v-parameterization support
 - conditional inputs
 - consistent batch indexing
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Beta Schedules
# ============================================================

def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    """
    Linear schedule for beta_t between `start` and `end`.
    """
    return torch.linspace(start, end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule (used in your main training pipeline)
    smoother than linear.
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-4, 0.9999)


# ============================================================
# Core DDPM process
# ============================================================

class Diffusion:
    """
    Implements forward diffusion, v-target computation, and loss function.
    Fully consistent with the main tumour CT generation training logic.
    """
    def __init__(self, timesteps=1000, device="cuda", use_cosine=True):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.tensor(
            cosine_beta_schedule(timesteps) if use_cosine else linear_beta_schedule(timesteps),
            dtype=torch.float32, device=device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_cum = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum = torch.sqrt(self.alpha_cum)
        self.sqrt_one_minus_alpha_cum = torch.sqrt(1.0 - self.alpha_cum)

    # --------------------------------------------------------
    def q_sample(self, x0, noise, t):
        """
        Forward diffusion step:
            x_t = sqrt(alphā_t) * x0 + sqrt(1 - alphā_t) * noise
        """
        a = self.sqrt_alpha_cum[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_cum[t].view(-1, 1, 1, 1)
        return a * x0 + b * noise

    # --------------------------------------------------------
    def get_v_target(self, x0, noise, t):
        """
        Compute v-parameterization target:
            v = sqrt(alphā_t) * noise - sqrt(1 - alphā_t) * x0
        """
        a = self.sqrt_alpha_cum[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_cum[t].view(-1, 1, 1, 1)
        return a * noise - b * x0

    # --------------------------------------------------------
    def predict_x0_from_v(self, xt, v_pred, t):
        """
        Reconstruct x0 (clean image) from v prediction:
            x0 = sqrt(alphā_t) * x_t - sqrt(1 - alphā_t) * v_pred
        """
        a = self.sqrt_alpha_cum[t].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_cum[t].view(-1, 1, 1, 1)
        return a * xt - b * v_pred

    # --------------------------------------------------------
    def p_losses(self, model, x_start, cond, t, use_v=False, drop_cond=False):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x_start.device)
        t = t.long()
        if t.dim() == 0:
            t = t.unsqueeze(0)
        B = x_start.shape[0]
        if t.shape[0] != B:
            t = t.expand(B)

        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, noise, t)

        pred_noise = model(x_t, cond, t, drop_cond=drop_cond)

        target = noise
        core_loss = F.mse_loss(pred_noise, noise)

        # correct x0_hat formula for eps-prediction:
        sqrt_alpha_cum = self.sqrt_alpha_cum[t].view(B,1,1,1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cum[t].view(B,1,1,1)

        x0_hat = (x_t - sqrt_one_minus * pred_noise) / sqrt_alpha_cum
        x0_hat = torch.clamp(x0_hat, -1, 1)

        return core_loss, pred_noise, x0_hat, target


    
    def sample(self, model, cond, shape, timesteps=1000, guidance_scale=2.0, device="cuda"):
        model.eval()
        B = cond.shape[0]
        x = torch.randn(shape, device=device)

        for t_idx in reversed(range(timesteps)):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)

            eps_cond   = model(x, cond, t, drop_cond=False)
            eps_uncond = model(x, cond, t, drop_cond=True)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            beta = self.betas[t].view(B,1,1,1)
            alpha = self.alphas[t].view(B,1,1,1)
            alpha_cum = self.alpha_cum[t].view(B,1,1,1)

            # posterior variance
            sigma = torch.sqrt(beta)

            # --- correct DDPM update ---
            x_prev = (1.0 / torch.sqrt(alpha)) * (
                        x - (beta / torch.sqrt(1 - alpha_cum)) * eps
                    )

            if t_idx > 0:
                x = x_prev + sigma * torch.randn_like(x)
            else:
                x = x_prev

        return x

        
# def visualize_sample(
#     diffusion,
#     model,
#     healthy_t,
#     target_t,
#     tumor_mask_t,
#     liver_mask_t,
#     pass_n,
#     stage="mid",
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#     force_size=(256,256),
#     timesteps=50,
#     guidance_scale=2.0,
#     vis_dir=None
# ):
#     """
#     Visualize model inference for a given healthy slice and condition.

#     Args:
#         diffusion: Diffusion object (handles sampling)
#         model: trained EnhancedUNet
#         healthy_t, target_t, tumor_mask_t, liver_mask_t: tensors
#         pass_n: current pass index (int)
#         stage: "mid" or "end"
#         device: torch device
#         force_size: (H, W) tuple
#         timesteps: number of DDPM steps for visualization
#         guidance_scale: classifier-free guidance scale
#         vis_dir: Path to save the figure (optional)
#     """
#     model.eval()
#     B = 1

#     cond = torch.cat([healthy_t[:1].to(device), liver_mask_t[:1].to(device)], dim=1)
#     cond = cond.float()

#     # Ensure exact shape: (1, 2, H, W)
#     cond = cond[:1, :2]                    # enforce B=1, C=2
#     cond = cond.float().to(device)

#     assert cond.ndim == 4, f"cond wrong shape: {cond.shape}"


#     # Sample from diffusion model
#     shape = (B, 1, *force_size)
#     with torch.no_grad():
#         generated = diffusion.sample(
#             model=model,
#             cond=cond,
#             shape=shape,
#             timesteps=timesteps,
#             guidance_scale=guidance_scale,
#             device=device
#         )

#     gen_img = generated[0, 0].cpu().numpy()
#     cond_img = healthy_t[0, 0].cpu().numpy()
#     target_img = target_t[0, 0].cpu().numpy()

#     # Optional: scale for display clarity
#     def to_disp(img):
#         img = (img + 1) / 2.0  # scale [-1,1] -> [0,1]
#         return np.clip(img, 0, 1)

#     gen_disp, cond_disp, target_disp = map(to_disp, [gen_img, cond_img, target_img])

#     # Plot
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     axs[0].imshow(cond_disp, cmap='gray')
#     axs[0].set_title('Condition (Healthy)')
#     axs[1].imshow(target_disp, cmap='gray')
#     axs[1].set_title('Target (Ground Truth)')
#     axs[2].imshow(gen_disp, cmap='gray')
#     axs[2].set_title('Generated (Reconstruction)')
#     for ax in axs:
#         ax.axis('off')

#     plt.tight_layout()
#     if vis_dir is not None:
#         save_path = vis_dir / f"pass_{pass_n+1}_{stage}_sample_fullCT.png"
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         print(f"✅ Visualization saved at: {save_path}")
#     plt.close()
#     torch.cuda.empty_cache()



