import torch
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def visualize_sample(
    diffusion,
    model,
    healthy_t,
    target_t,
    tumor_mask_t,
    liver_mask_t,
    pass_n,
    stage="mid",
    device="cuda",
    force_size=(256,256),
    timesteps=40,
    guidance_scale=2.0,
    vis_dir=None
):
    # -------------------------------------------------------
    # Unwrap DataParallel
    # -------------------------------------------------------
    net = model.module if hasattr(model, "module") else model
    net.eval()

    # -------------------------------------------------------
    # Conditioning vector (same as in training)
    # -------------------------------------------------------
    healthy_1 = healthy_t[:1].to(device)
    liver_1   = liver_mask_t[:1].to(device)
    cond = torch.cat([healthy_1, liver_1], dim=1).float()

    H, W = force_size

    # -------------------------------------------------------
    # (1) Full final DDPM sampling
    # -------------------------------------------------------
    final_sample = diffusion.sample(
        model=net,
        cond=cond,
        shape=(1,1,H,W),
        timesteps=timesteps,
        guidance_scale=guidance_scale,
        device=device
    )[0,0].cpu().numpy()


    # -------------------------------------------------------
    # (2) Training-style x0_hat reconstruction
    # -------------------------------------------------------
    t = torch.tensor([timesteps//2], device=device).long()

    # forward noisy sample
    noise = torch.randn_like(healthy_1)
    x_t = diffusion.q_sample(healthy_1, noise, t)

    # model prediction (v-prediction)
    v_pred = net(x_t, cond, t)

    # reconstruct x0_hat using YOUR diffusion code
    x0_hat_tensor = diffusion.predict_x0_from_v(
        x_t,
        v_pred,
        t
    )
    x0_hat = x0_hat_tensor[0,0].cpu().numpy()

    # -------------------------------------------------------
    # (3) Healthy + ground truth
    # -------------------------------------------------------
    healthy_img = healthy_t[0,0].cpu().numpy()
    target_img  = target_t[0,0].cpu().numpy()

    def disp(img):
        return np.clip((img + 1) / 2, 0, 1)

    healthy_disp = disp(healthy_img)
    target_disp  = disp(target_img)
    x0_hat_disp  = disp(x0_hat)
    final_disp   = disp(final_sample)

    # -------------------------------------------------------
    # Plot 4-panel visualization
    # -------------------------------------------------------
    fig, axs = plt.subplots(1, 4, figsize=(22,6))

    axs[0].imshow(healthy_disp, cmap="gray"); axs[0].set_title("Healthy (Condition)")
    axs[1].imshow(target_disp, cmap="gray");  axs[1].set_title("Ground Truth")
    axs[2].imshow(x0_hat_disp, cmap="gray");  axs[2].set_title("Training x0_hat")
    axs[3].imshow(final_disp, cmap="gray");   axs[3].set_title("Final Reconstruction")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    if vis_dir is not None:
        save_path = vis_dir / f"recon_pass{pass_n}_{stage}.png"
        plt.savefig(save_path, dpi=130)
        print("Saved:", save_path)
    plt.close()
