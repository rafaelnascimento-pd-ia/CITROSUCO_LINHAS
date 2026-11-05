import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==============================================================================
# CLASSE DE DATASET DO PYTORCH: OrangeDataset
# Responsável por gerenciar os pares de (imagem aérea, máscara de linhas)
# e fornecê-los ao modelo sob demanda durante o treino, validação e teste.
# ==============================================================================
class OrangeDataset(Dataset):
    """
    Dataset customizado do PyTorch para segmentação de linhas de plantio de citros.
    Herda da classe base 'torch.utils.data.Dataset'.
    """
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Construtor da classe:
        - images_dir: Diretório que contém os recortes (tiles) das ortofotos em PNG.
        - masks_dir: Diretório que contém as máscaras binárias das linhas em PNG.
        - transform: Pipeline de transformações e data augmentation (Albumentations).
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        # 1. Busca e mapeia todos os arquivos .png nas pastas de imagens e máscaras
        # sorted() garante que a imagem 'tile_001.png' corresponda à máscara 'tile_001.png'
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        
        # 2. Verificação de integridade: checa se há exatamente uma máscara para cada imagem
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Diferença entre número de imagens ({len(self.image_paths)}) e máscaras ({len(self.mask_paths)})!"

    def __len__(self):
        """
        Retorna o número total de amostras (tiles) no dataset.
        Permite ao PyTorch e ao DataLoader saber quantas iterações compõem uma época.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Carrega e processa uma única amostra com base no seu índice (idx):
        1. Lê a imagem RGB e a máscara de linhas do disco.
        2. Aplica as transformações (data augmentation e normalização).
        3. Retorna o par (imagem, máscara) como tensores prontos para a GPU.
        """
        # Obtém o caminho do arquivo correspondente ao índice
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Leitura da imagem com OpenCV: por padrão lê em BGR, convertemos para RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Leitura da máscara binária em escala de cinza (1 canal com valores de 0 ou 255)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Se um pipeline de transformações (Albumentations) foi fornecido:
        if self.transform:
            # Aplica aumentação sincronizada: rotaciona/inverte a imagem e a máscara exatamente juntas
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Fallback caso não seja passado transform:
            # Converte a imagem para float [0, 1] e formato de canais do PyTorch: (C, H, W)
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float32)
            
            # Converte a máscara para float [0, 1] e adiciona a dimensão de canal (1, H, W)
            mask = mask.astype(np.float32) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            
        return image, mask


# ==============================================================================
# PIPELINE DE TRANSFORMAÇÕES E DATA AUGMENTATION (Albumentations)
# Previne overfitting e ensina a rede neural a reconhecer linhas em diferentes
# iluminações, inclinações e orientações geográficas.
# ==============================================================================
def get_transforms(img_size=256, is_train=True):
    """
    Retorna o pipeline de transformações do Albumentations.
    - is_train=True: Inclui data augmentation (giros, rotações, variações de brilho).
    - is_train=False: Apenas normalização matemática padrão ImageNet (para validação/teste).
    """
    # Médias e desvios padrão oficiais do ImageNet (compatíveis com a ResNet pré-treinada)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    if is_train:
        # Pipeline completo de treino (com Data Augmentation)
        return A.Compose([
            # 1. Espelhamento horizontal e vertical (50% de chance cada)
            # Ajuda o modelo a entender linhas em qualquer sentido de plantio
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # 2. Deslocamento leve, variação de zoom e rotação de até 45 graus
            # Simula drones voando em ângulos e altitudes ligeiramente diferentes
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            
            # 3. Variação de iluminação (30% de chance de aplicar uma das opções)
            # Simula fotos tiradas com sol forte, nuvens ou sombras no pomar
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=1.0),
            ], p=0.3),
            
            # 4. Normalização com base nos valores do ImageNet (z-score: (x - mean) / std)
            A.Normalize(mean=mean, std=std),
            
            # 5. Converte matriz NumPy (H, W, C) para Tensor PyTorch (C, H, W)
            ToTensorV2()
        ])
    else:
        # Pipeline de validação e teste: NÃO altera o conteúdo visual das imagens,
        # apenas normaliza e converte para Tensor para avaliação justa das métricas.
        return A.Compose([
            # Apenas Normalização e conversão para Tensor
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

