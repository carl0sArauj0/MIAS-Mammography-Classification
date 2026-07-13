import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

def get_model(num_classes=3, feature_extract=False):
    """
    Crea un modelo basado en ResNet50 para Transfer Learning.
    
    Args:
        num_classes (int): Número de clases (Normal, Benigno, Maligno).
        feature_extract (bool): Si es True, congela los pesos de la base y solo 
                               entrena la nueva capa final. Si es False, hace Fine-tuning.
    """
    
    # 1. Cargar ResNet50 con los mejores pesos actuales de ImageNet
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    # 2. Control de congelación de capas 
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes) 
    )
    
    return model

if __name__ == "__main__":

    net = get_model(num_classes=3)
    print(net)
    print("\nModelo cargado exitosamente.")