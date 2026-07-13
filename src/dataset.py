import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import glob
from PIL import Image

class MIASDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): Metadata con columnas 'file_name' y 'target'.
            root_dir (string): Directorio con las imágenes .pgm.
            transform (callable, optional): Transformaciones de PyTorch (Resize, ToTensor, etc).
        """
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _apply_clahe(self, img):
        """Mejora del contraste médico."""
        # Normalizar a 8 bits
        img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Aplicar CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(img_8u)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. Obtener ID del archivo del CSV
        img_id = self.df.iloc[idx]['file_name']
        
        # 2. Localizar el archivo real (manejando mdb001 vs mdb001lm.pgm)
        search_pattern = os.path.join(self.root_dir, f"{img_id}*.pgm")
        files = glob.glob(search_pattern)
        
        if not files:
            raise FileNotFoundError(f"No se encontró la imagen para el ID: {img_id}")
        
        img_path = files[0]

        # 3. Leer imagen con OpenCV
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        # 4. Preprocesamiento médico
        image = self._apply_clahe(image)
        
        # 5. Convertir a RGB (3 canales)
        # IMPORTANTE: Los modelos de Transfer Learning (ResNet, etc.) esperan 3 canales
        # aunque la imagen sea en blanco y negro.
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 6. Convertir a objeto PIL para que las transforms de PyTorch funcionen correctamente
        image = Image.fromarray(image)

        # 7. Obtener etiqueta
        label = int(self.df.iloc[idx]['target'])

        # 8. Aplicar transformaciones (Resize, Normalización de ImageNet, etc.)
        if self.transform:
            image = self.transform(image)

        return image, label