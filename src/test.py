import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import MIASDataset
from model import get_model

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_df = pd.read_csv('../data/processed/test_split.csv')
    
    # transformaciones
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_ds = MIASDataset(test_df, '../data/raw/', transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # cargar modelo
    model = get_model(num_classes=3)
    model.load_state_dict(torch.load('../results/models/best_mias_model.pth'))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # inferencia
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # generar reporte
    target_names = ['Normal', 'Benigno', 'Maligno']
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(all_labels, all_preds, target_names=target_names, labels=[0,1,2]))

    # matriz confusión
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión - Clasificación de Mamografías')
    plt.savefig('../results/plots/confusion_matrix.png')
    plt.show()

if __name__ == '__main__':
    evaluate_model()