import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from dataset import MIASDataset
from model import get_model

def train_model():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Iniciando entrenamiento en: {device}")

    # Rutas relativas robustas
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_path, 'data', 'info_mias.csv')
    img_dir = os.path.join(base_path, 'data', 'raw')
    model_save_path = os.path.join(base_path, 'results', 'models', 'best_mias_model.pth')
    test_csv_save = os.path.join(base_path, 'data', 'processed', 'test_split.csv')

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_csv_save), exist_ok=True)

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.20, stratify=df['target'], random_state=42)
    val_df.to_csv(test_csv_save, index=False)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = MIASDataset(train_df, img_dir, transform=train_transform)
    val_ds = MIASDataset(val_df, img_dir, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = get_model(num_classes=3).to(device)
    
    weights = torch.tensor([1.0, 2.5, 3.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=2e-5) 

    num_epochs = 60
    best_acc = 0

    print("-" * 30)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, preds = torch.max(model(inputs), 1)
                correct += torch.sum(preds == labels.data)
        
        acc = correct.double() / len(val_df)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_save_path)
            status = "⭐ Mejorado"
        else: status = ""

        print(f"Época {epoch+1}/{num_epochs} | Loss: {running_loss/len(train_ds):.4f} | Acc Val: {acc:.4f} {status}")

if __name__ == '__main__':
    train_model()