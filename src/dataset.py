import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import glob
from PIL import Image

class MIASDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _apply_clahe(self, img):
        img_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(img_8u)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['file_name']
        search_pattern = os.path.join(self.root_dir, f"{img_id}*.pgm")
        files = glob.glob(search_pattern)
        if not files:
            raise FileNotFoundError(f"No se encontró la imagen: {img_id}")
        
        image = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
        image = self._apply_clahe(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image)

        label = int(self.df.iloc[idx]['target'])
        if self.transform:
            image = self.transform(image)
        return image, label