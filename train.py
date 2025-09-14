import csv
import os
from datetime import datetime
from Model import UNet
from forward import forward_diffusion_sample
from unet import SimpleUnet
from dataloader import load_dataset
import torch.nn.functional as F
import torch
from torch.optim import Adam
import logging
import os

logging.basicConfig(level=logging.INFO)

def init_log_csv(log_file='experiment_log.csv'):
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'date', 'model', 'dataset', 'image_size',
                'batch_size', 'lr', 'timesteps', 'epochs',
                'final_loss', 'best_loss','notes'
            ]
            writer.writerow(header)
    return log_file
def log_experiment_to_csv(exp_data, log_file='experiment_log.csv'):
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        row = [
             exp_data['date'], exp_data['model'], exp_data['dataset'], exp_data['image_size'],
            exp_data['batch_size'], exp_data['lr'],exp_data['timesteps'],  exp_data['epochs'],
            exp_data['final_loss'], exp_data['best_loss'],exp_data.get('notes', '')
        ]
        writer.writerow(row)

save_dir = "./trained_models"
save_path = os.path.join(save_dir, "ddpm_mse_epochs_100.pth")
os.makedirs(save_dir, exist_ok=True)

def get_loss(model, x_0, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.mse_loss(noise, noise_pred)


if __name__ == "__main__":
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    T = 1000
    BATCH_SIZE = 128
    epochs = 100

    dataloader = load_dataset(batch_size=BATCH_SIZE)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    log_file = init_log_csv()
    final_loss = None
    best_loss = float('inf')


    for epoch in range(epochs):
        for batch_idx, (batch, _) in enumerate(dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, batch, t, device=device)
            if loss.item() < best_loss:
                best_loss = loss.item()
            final_loss=loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                logging.info(f"Epoch {epoch} | Batch index {batch_idx:03d} Loss: {loss.item()}")

    experiment_data = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'model': 'DDPM',
        'dataset': 'ae4dc',
        'image_size': 64,
        'batch_size': BATCH_SIZE,
        'lr': 1e-3,
        'timesteps': T,
        'epochs': epochs ,
        'final_loss': final_loss,
        'best_loss': best_loss,
        'notes': 'First attempt, baseline model'
    }
    log_experiment_to_csv(experiment_data , log_file='experiment_log.csv')
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到 {save_path}")