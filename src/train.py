import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt


from dataset import MIASDataset
from model import get_model

def train_model():
    # 1. Configuración de Dispositivo 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    # 2. Cargar Metadata y Dividir Datos
    df = pd.read_csv('../data/info_mias.csv')
    
    # Dividir en Train (80%) y Validation (30%) 
    train_df, val_df = train_test_split(
        df, 
        test_size=0.25, 
        stratify=df['target'], 
        random_state=42
    )
    
    # Guardamos el val_df como nuestro set de prueba para después
    val_df.to_csv('../data/processed/test_split.csv', index=False)
    print(f"Datos de entrenamiento: {len(train_df)} | Datos de validación: {len(val_df)}")

    # 3. Data Augmentation para el set de entrenamiento 
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 4. Crear Datasets y DataLoaders
    image_dir = '../data/raw/'
    train_ds = MIASDataset(train_df, image_dir, transform=data_transforms['train'])
    val_ds = MIASDataset(val_df, image_dir, transform=data_transforms['val'])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # 5. Inicializar Modelo, Pérdida y Optimizador
    model = get_model(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 

    # 6. Bucle de Entrenamiento
    num_epochs = 20
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validación
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct.double() / len(val_loader.dataset)

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.item())

        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}')

        # Guardar el mejor modelo
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), '../results/models/best_mias_model.pth')
            print("--- Modelo guardado ---")

    # Guardar CSV de test para el script de evaluación final
    val_df.to_csv('../data/processed/test_split.csv', index=False)
    print("Entrenamiento completado.")
    return history

if __name__ == '__main__':
    # Crear carpeta de resultados si no existe
    os.makedirs('../results/models', exist_ok=True)
    os.makedirs('../results/plots', exist_ok=True)
    os.makedirs('../data/processed', exist_ok=True)
    
    stats = train_model()
    
    # Graficar resultados
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(stats['train_loss'], label='Train Loss')
    plt.plot(stats['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Pérdida por Época')
    
    plt.subplot(1, 2, 2)
    plt.plot(stats['val_acc'], label='Val Accuracy')
    plt.legend(); plt.title('Precisión de Validación')
    plt.savefig('../results/plots/training_curves.png')
    plt.show()