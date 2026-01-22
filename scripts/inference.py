import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.mitUnet import modelchm 


def run_inference(model_path, image_path, output_dir, device):
    CHM_MIN = 0.0
    CHM_MAX = 60 

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = modelchm(img_size=256, num_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img_name = os.path.basename(image_path)
    raw_img = Image.open(image_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out']
        pred_chm = output.squeeze().cpu().numpy() * (CHM_MAX - CHM_MIN) + CHM_MIN
      
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(raw_img.resize((256, 256)))
    ax1.set_title(f'Original RGB: {img_name}')
    ax1.axis('off')

    im = ax2.imshow(pred_chm, cmap='viridis')
    ax2.set_title('Predicted CHM (meters)')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, label='Height (m)')

    save_name = f"pred_{img_name.split('.')[0]}.png"
    plt.savefig(os.path.join(output_dir, save_name))
    plt.close()
    print(f"Resultado guardado en: {os.path.join(output_dir, save_name)}")

if __name__ == '__main__':
    # Configuración rápida
    MODEL_WEIGHTS = './results/experimento1/checkpoints/best_model.pth'
    INPUT_FOLDER = './test_images/'
    OUTPUT_FOLDER = './predictions/'
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for img_file in os.listdir(INPUT_FOLDER):
        if img_file.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            path = os.path.join(INPUT_FOLDER, img_file)
            run_inference(MODEL_WEIGHTS, path, OUTPUT_FOLDER, device)
