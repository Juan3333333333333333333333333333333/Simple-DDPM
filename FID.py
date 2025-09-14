import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision import transforms
import torch.nn as nn
import os
from PIL import Image
from tqdm import tqdm
import time


class FIDCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = self.load_inception()
        self.transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        print(f"FID calculator initialized on {device}")

    def load_inception(self):
        model = inception_v3(pretrained=True, transform_input=False)
        model = model.to(self.device)
        model.eval()
        model.fc = nn.Identity()
        return model

    def get_features_batch(self, image_paths, batch_size=32, desc="Extracting features"):
        features_list = []

        with tqdm(total=len(image_paths), desc=desc, unit="img") as pbar:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = []

                for img_path in batch_paths:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img_tensor = self.transform(img).to(self.device)
                        batch_images.append(img_tensor)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue

                if batch_images:
                    batch_tensor = torch.stack(batch_images)
                    with torch.no_grad():
                        features = self.model(batch_tensor)
                    features_list.append(features.cpu().numpy())

                pbar.update(len(batch_paths))
                pbar.set_postfix({"Batch": f"{i // batch_size + 1}/{(len(image_paths) - 1) // batch_size + 1}"})

        if features_list:
            return np.concatenate(features_list, axis=0)
        else:
            return np.array([])

    def calculate_fid_with_progress(self, real_features, fake_features, desc="Calculating FID"):
        with tqdm(total=4, desc=desc) as pbar:
            pbar.set_description("Calculating means")
            mu_real = np.mean(real_features, axis=0)
            mu_fake = np.mean(fake_features, axis=0)
            pbar.update(1)
            time.sleep(0.1)

            pbar.set_description("Calculating covariances")
            sigma_real = np.cov(real_features, rowvar=False)
            sigma_fake = np.cov(fake_features, rowvar=False)
            pbar.update(1)
            time.sleep(0.1)

            pbar.set_description("Computing matrix sqrt")
            covmean = sqrtm(sigma_real.dot(sigma_fake))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            pbar.update(1)
            time.sleep(0.1)

            pbar.set_description("Final FID computation")
            diff = mu_real - mu_fake
            fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
            pbar.update(1)

        return fid


def get_image_paths(directory, max_images=None):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_paths = []

    print(f"Scanning directory: {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
                if max_images and len(image_paths) >= max_images:
                    break
        if max_images and len(image_paths) >= max_images:
            break

    return image_paths


def calculate_fid_with_progress(real_dir, fake_dir, batch_size=32, max_images=5000, device='cuda'):
    print("🚀 Starting FID calculation...")
    print("=" * 50)

    fid_calc = FIDCalculator(device)

    print("📁 Collecting image paths...")
    real_paths = get_image_paths(real_dir, max_images)
    fake_paths = get_image_paths(fake_dir, max_images)

    if not real_paths:
        raise ValueError(f"No images found in real directory: {real_dir}")
    if not fake_paths:
        raise ValueError(f"No images found in fake directory: {fake_dir}")

    print(f"📊 Found {len(real_paths)} real images and {len(fake_paths)} fake images")
    print("=" * 50)

    real_features = fid_calc.get_features_batch(
        real_paths,
        batch_size=batch_size,
        desc="📸 Extracting real image features"
    )

    print("=" * 50)

    fake_features = fid_calc.get_features_batch(
        fake_paths,
        batch_size=batch_size,
        desc="🎨 Extracting generated image features"
    )

    print("=" * 50)

    if len(real_features) == 0 or len(fake_features) == 0:
        raise ValueError("Failed to extract features from images")

    print(f"✅ Feature extraction complete")
    print(f"   Real features shape: {real_features.shape}")
    print(f"   Fake features shape: {fake_features.shape}")
    print("=" * 50)

    fid_score = fid_calc.calculate_fid_with_progress(
        real_features,
        fake_features,
        desc="🧮 Calculating FID score"
    )

    print("=" * 50)
    print(f"🎉 FID Calculation Complete!")
    print(f"   FID Score: {fid_score:.2f}")
    print("=" * 50)

    return fid_score


def compare_multiple_fid(real_dir, fake_dirs, batch_size=32, max_images=5000, device='cuda'):
    results = {}

    for fake_dir in tqdm(fake_dirs, desc="📊 Comparing multiple models"):
        try:
            fid_score = calculate_fid_with_progress(real_dir, fake_dir, batch_size, max_images, device)
            results[fake_dir] = fid_score
            print(f"✅ {os.path.basename(fake_dir)}: FID = {fid_score:.2f}")
        except Exception as e:
            print(f"❌ Error processing {fake_dir}: {e}")
            results[fake_dir] = None

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    real_images_dir = "ae4dc-main/anime-data/anime-data/anime-faces"
    fake_images_dir = "fid_generated_images"

    try:
        fid_score = calculate_fid_with_progress(
            real_images_dir,
            fake_images_dir,
            batch_size=32,
            max_images=5000,
            device=device
        )

        print(f"\n📈 Final FID Score: {fid_score:.2f}")

        if fid_score < 10:
            quality = "Excellent (接近真实数据)"
        elif fid_score < 30:
            quality = "Good"
        elif fid_score < 50:
            quality = "Fair"
        elif fid_score < 100:
            quality = "Poor"
        else:
            quality = "Very Poor"

        print(f"🏆 Quality Assessment: {quality}")

    except Exception as e:
        print(f"❌ Error during FID calculation: {e}")