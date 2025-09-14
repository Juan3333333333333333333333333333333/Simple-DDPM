import torch
from forward import (
    get_index_from_list,
    sqrt_one_minus_alphas_cumprod,
    betas,
    posterior_variance,
    sqrt_recip_alphas,
)
import matplotlib.pyplot as plt
from dataloader import show_tensor_image
from unet import SimpleUnet
import os
from tqdm import tqdm


@torch.no_grad()
def sample_timestep(model, x, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if torch.all(t == 0):
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, device, img_size, T):
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    num_images = 10
    stepsize = int(T / num_images)

    for i in reversed(range(0, T)):
        t = torch.tensor([i], device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i / stepsize) + 1)
            show_tensor_image(img.detach().cpu())
    plt.savefig("sample.png")


@torch.no_grad()
def generate_and_save_images(model, device, img_size, T, num_images=1000, output_dir="generated_images"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    batch_size = 16
    num_batches = (num_images + batch_size - 1) // batch_size

    print(f"Generating {num_images} images in {num_batches} batches...")

    for batch_idx in tqdm(range(num_batches), desc="Generating images"):
        current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
        img_batch = torch.randn((current_batch_size, 3, img_size, img_size), device=device)

        for i in reversed(range(0, T)):
            t = torch.tensor([i] * current_batch_size, device=device, dtype=torch.long)
            img_batch = sample_timestep(model, img_batch, t)

        img_batch = torch.clamp(img_batch, -1.0, 1.0)
        img_batch = (img_batch + 1) / 2

        for img_idx in range(current_batch_size):
            image_idx_global = batch_idx * batch_size + img_idx
            filename = f"{image_idx_global:06d}.png"
            filepath = os.path.join(output_dir, filename)

            plt.figure(figsize=(3, 3))
            plt.imshow(img_batch[img_idx].permute(1, 2, 0).cpu().numpy())
            plt.axis('off')
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
            plt.close()

    print(f"Successfully generated {num_images} images in '{output_dir}' directory")


@torch.no_grad()
def generate_single_image(model, device, img_size, T):
    img = torch.randn((1, 3, img_size, img_size), device=device)

    for i in reversed(range(0, T)):
        t = torch.tensor([i], device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)

    img = torch.clamp(img, -1.0, 1.0)
    img = (img + 1) / 2

    return img


if __name__ == "__main__":
    img_size = 64
    T = 1000
    model = SimpleUnet()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.load_state_dict(torch.load("trained_models/ddpm_mse_epochs_100.pth", map_location=device))
    model.to(device)
    model.eval()

    generate_and_save_images(
        model=model,
        device=device,
        img_size=img_size,
        T=T,
        num_images=1000,
        output_dir="fid_generated_images"
    )

    sample_plot_image(model=model, device=device, img_size=img_size, T=T)

    single_image = generate_single_image(model, device, img_size, T)
    plt.figure(figsize=(5, 5))
    plt.imshow(single_image[0].permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig("single_generated.png")
    plt.show()