import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

def extract_features(model_name='resnet50', layer_name='avgpool', input_size=224):
    """
    Extract features from images using pretrained model
    Args:
        model_name: 'resnet50', 'vgg16', or 'vgg19' 
        layer_name: layer to extract features from
        input_size: input image size
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load pretrained model
    print(f"Loading pretrained {model_name}...")
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # ResNet50 final features: 2048
        feature_dim = 2048
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # VGG16 final features: 4096
        feature_dim = 4096
    elif model_name == 'vgg19':
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # VGG19 final features: 4096
        feature_dim = 4096
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = model.to(device)
    model.eval()
    
    # Create feature extractor
    if model_name.startswith('resnet'):
        if layer_name == 'avgpool':
            new_model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    else:  # VGG models
        if layer_name == 'features':
            new_model = torch.nn.Sequential(*list(model.features.children())).to(device)
        elif layer_name == 'classifier':
            # Get features before the last layer
            new_model = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
    
    # Get model's preprocessing
    if model_name.startswith('resnet'):
        weights = models.ResNet50_Weights.DEFAULT if model_name == 'resnet50' else models.ResNet18_Weights.DEFAULT
    elif model_name == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT
    else:  # vgg19
        weights = models.VGG19_Weights.DEFAULT
        
    preprocess = weights.transforms()
    
    # Load and process images
    features_list = []
    img_files = []
    
    print("\nProcessing images from 'chair3' directory...")
    for filename in tqdm(sorted(os.listdir('chair3'))):
        if filename.endswith('.png'):
            img_path = os.path.join('chair3', filename)
            img_files.append(filename)
            
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                features = new_model(img_tensor)
                
            # Convert to numpy and flatten
            features = features.squeeze().cpu().numpy()
            features_list.append(features)
            
    # Stack all features
    features_array = np.stack(features_list)
    
    # Save features and filenames
    save_name = f'chair_features_{model_name}'
    np.save(f'{save_name}.npy', features_array)
    with open(f'{save_name}_files.txt', 'w') as f:
        f.write('\n'.join(img_files))
        
    print(f'\nFeatures shape: {features_array.shape}')
    print(f'Features saved to: {save_name}.npy')
    print(f'File list saved to: {save_name}_files.txt')
    return features_array

if __name__ == "__main__":
    # Extract features using ResNet50 (2048-d features)
    features_2048 = extract_features(model_name='vgg16')
    
    # Optionally, extract features using VGG16 (4096-d features, closer to paper's model)
    # features_4096 = extract_features(model_name='vgg16')
