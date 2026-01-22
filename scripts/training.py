import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F_transforms
import segmentation_models_pytorch as smp
import argparse
import time
import matplotlib.pyplot as plt
import random
from models.mitUnet import modelchm
from utils.functions import *

def set_seed(seed=2204):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Semilla fijada en: {seed}")



if __name__ == '__main__':
    set_seed(1005150) 
    parser = argparse.ArgumentParser(description= "Training")
    parser.add_argument('--exp_name', type=str, required= True, help='Name of the experiment')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'mse', 'huber', 'logcosh'], help='Tipo de función de pérdida a utilizar')
    parser.add_argument('--h5_path', default='./dataset_h5_files/', type=str, help='Ruta a los archivos HDF5')
    parser.add_argument('--num_epochs', default=150, type=int, help='Número de épocas')
    parser.add_argument('--batch_size', default=32, type=int, help='Tamaño del lote')
    parser.add_argument('--learning_rate', default=5e-4, type=float, help='Learning rate')
    args = parser.parse_args()

    base_path = os.path.join('results', args.exp_name)
    ckpt_dir  = os.path.join(base_path, 'checkpoints')
    vis_dir   = os.path.join(base_path, 'val_patches')

    os.makedirs(ckpt_dir, exist_ok= True)
    os.makedirs(vis_dir, exist_ok= True)
    with open(os.path.join(base_path, 'config.txt'), 'w') as f:
        f.write(str(args))

    H5_TRAIN_PATH = os.path.join(args.h5_path, 'train_dataset.h5')
    H5_VAL_PATH = os.path.join(args.h5_path, 'val_dataset.h5')
    
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Inicializando datasets...", flush=True)
    train_dataset = H5Dataset(H5_TRAIN_PATH, augmentation=True)
    val_dataset = H5Dataset(H5_VAL_PATH, augmentation=False)

    print(f"Creando DataLoaders con batch_size={BATCH_SIZE}...", flush=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Inicializando modelo...", flush=True)
   
    model = UNet(img_size=256, num_channels=3, num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #Logica de la loss function
    if args.loss == 'l1':
        criterion = nn.L1Loss(reduction='none')
    elif args.loss == 'mse':
        criterion = nn.MSELoss(reduction='none')
    elif args.loss == 'huber':
        criterion = nn.HuberLoss(reduction= 'none')
    elif args.loss == 'logcosh':
        criterion = log_cosh_loss
    scaler = torch.cuda.amp.GradScaler() 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping_patience = 15
    epochs_no_improve = 0
    best_val_mae = float('inf')
    train_losses = []
    val_losses = []
    val_maes = []
    val_rmses = []

    try:
        print("Comenzando el entrenamiento...", flush=True)
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            train_valid_batches = 0
            start_time = time.time()
            
            for i, (rgb_batch, chm_batch, final_mask) in enumerate(train_loader):
                batch_start = time.time()
                rgb_batch = rgb_batch.to(device)
                chm_batch = chm_batch.to(device)
                final_mask = final_mask.to(device)
                
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    predictions = model(rgb_batch)['out']
                    predictions_masked = predictions.squeeze(1)[final_mask]
                    targets_masked = chm_batch.squeeze(1)[final_mask]
                    
                    if targets_masked.numel() > 0:
                        main_loss = criterion(predictions_masked, targets_masked).mean()
                    else:
                        main_loss = torch.tensor(0.0, device=device)
                    loss = main_loss
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                if targets_masked.numel() > 0:
                    running_loss += loss.item()
                    train_valid_batches += 1
                    print(f"Época {epoch+1}/{NUM_EPOCHS}, Lote {i+1}/{len(train_loader)}, Perdida: {loss.item():.4f}, Tiempo: {time.time() - batch_start:.2f}s", flush=True)
                else:
                    print(f"Época {epoch+1}/{NUM_EPOCHS}, Lote {i+1}/{len(train_loader)}, saltando", flush=True)
            
            avg_train_loss = running_loss / train_valid_batches if train_valid_batches > 0 else float('inf')
            train_losses.append(avg_train_loss)
            
            model.eval()
            val_mae = 0.0
            val_rmse = 0.0
            val_loss = 0.0
            valid_batches = 0
            vis_images = []
            with torch.no_grad():
                for i, (rgb_batch, chm_batch, final_mask) in enumerate(val_loader):
                    rgb_batch = rgb_batch.to(device)
                    chm_batch = chm_batch.to(device)
                    final_mask = final_mask.to(device)
                    predictions = model(rgb_batch)['out']
                    
                    predictions_masked = predictions.squeeze(1)[final_mask]
                    targets_masked = chm_batch.squeeze(1)[final_mask]
                    
                    if targets_masked.numel() > 0:
                        loss_batch = criterion(predictions_masked, targets_masked)
                        val_loss += loss_batch.mean().item()
                        
                        chm_min_val = val_dataset.chm_min
                        chm_max_val = val_dataset.chm_max
                        predictions_denorm = predictions_masked * (chm_max_val - chm_min_val) + chm_min_val
                        targets_denorm = targets_masked * (chm_max_val - chm_min_val) + chm_min_val
                        
                        mae_batch = torch.abs(predictions_denorm - targets_denorm)
                        val_mae += mae_batch.mean().item()
                        rmse_batch = torch.sqrt(torch.mean((predictions_denorm - targets_denorm)**2))
                        val_rmse += rmse_batch.item()
                        valid_batches += 1

                        if i == 0 and len(vis_images) < 2:
                            vis_images.append((rgb_batch.cpu(), predictions.cpu(), chm_batch.cpu()))
            
            avg_val_mae = val_mae / valid_batches if valid_batches > 0 else float('inf')
            avg_val_rmse = val_rmse / valid_batches if valid_batches > 0 else float('inf')
            avg_val_loss = val_loss / valid_batches if valid_batches > 0 else float('inf')
            val_maes.append(avg_val_mae)
            val_rmses.append(avg_val_rmse)
            val_losses.append(avg_val_loss)

            print(f"epoca {epoch+1}/{NUM_EPOCHS}, perdida promedio: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f} m, Val RMSE: {avg_val_rmse:.4f} m, Tiempo total: {time.time() - start_time:.2f}s", flush=True)

            scheduler.step(avg_val_mae)
            
            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                epochs_no_improve = 0
                file_name = f'chm-epoch{epoch+1}-mae{avg_val_mae:.4f}.pth'
                save_path = os.path.join(ckpt_dir, file_name)
                torch.save(model.state_dict(), save_path)
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
                print(f"Modelo guardado en {save_path}", flush=True)
            else:
                epochs_no_improve += 1
                print(f"Mejora no encontrada. Épocas sin mejorar: {epochs_no_improve}/{early_stopping_patience}", flush=True)
                if epochs_no_improve >= early_stopping_patience:
                    print("Early stopping activado. Terminando el entrenamiento.", flush=True)
                    break

            for idx, (rgb, pred, chm) in enumerate(vis_images[:2]):
                rgb = rgb[0].numpy().transpose(1, 2, 0)  # (H, W, 3)
                pred = pred[0, 0].numpy() * (val_dataset.chm_max - val_dataset.chm_min)  # (H, W), en metros
                chm = chm[0, 0].numpy() * (val_dataset.chm_max - val_dataset.chm_min)  # (H, W), en metros
                
                rgb = rgb * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                rgb = np.clip(rgb, 0, 1)
                
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
                ax1.imshow(rgb)
                ax1.set_title('RGB')
                ax1.axis('off')
                
                ax2.imshow(pred, cmap='viridis', vmin=0, vmax=val_dataset.chm_max)
                ax2.set_title('CHM prediction (m)')
                ax2.axis('off')
                
                ax3.imshow(chm, cmap='viridis', vmin=0, vmax=val_dataset.chm_max)
                ax3.set_title('CHM Real (m)')
                ax3.axis('off')
                
                plt.savefig(os.path.join(vis_dir, f'epoch{epoch+1}_patch{idx+1}.png'))
                plt.close()

    except Exception as e:
        print(f"Se ha producido un error durante el entrenamiento: {e}", flush=True)
        
    finally:
        train_dataset.close()
        val_dataset.close()
        torch.cuda.empty_cache()
        print("Archivos HDF5 cerrados. Entrenamiento finalizado.", flush=True)

        print("Generando gráficos de métricas...", flush=True)
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, train_losses, 'b', label='Training Loss')
        plt.plot(epochs_range, val_losses, 'orange', label='Validation Loss')
        plt.title('Loss function')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(base_path, 'training_loss_plot.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, val_maes, 'r', label='MAE (m)')
        plt.plot(epochs_range, val_rmses, 'g', label='RMSE (m)')
        plt.title('Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Error ')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(base_path, 'validation_metrics_plot.png'))
        plt.close()
