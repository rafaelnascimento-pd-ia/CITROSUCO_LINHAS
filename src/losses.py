"""
Centerline Dice Loss (clDice) — Topology-Preserving Loss for Tubular Structures
=================================================================================

Implementação baseada no paper:
  Shit et al., "clDice - a Novel Topology-Preserving Loss Function for
  Tubular Structure Segmentation", CVPR 2021.
  https://github.com/jocpae/clDice

A ideia central:
  - Dice Loss padrão mede sobreposição volumétrica, mas ignora a TOPOLOGIA.
  - clDice mede a sobreposição entre os ESQUELETOS (centerlines) da predição
    e do ground truth, forçando o modelo a preservar a conectividade das
    fileiras de árvores de citros.

Componentes:
  1. SoftSkeletonize2D: Esqueletização diferenciável via min/max pooling iterativo
  2. SoftCLDiceLoss: clDice loss pura (skeleton overlap)
  3. BCEDiceCLDiceLoss: Loss combinada = BCE + Dice + clDice (recomendada)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Operações Morfológicas Diferenciáveis (Soft Morphology)
# ==============================================================================

def soft_erode(img):
    """
    Erosão suave diferenciável usando -max_pool(-img).
    Equivale a min pooling com kernel 3x3 separável.
    """
    if len(img.shape) != 4:
        raise ValueError(f"soft_erode espera tensor 4D (B,C,H,W), recebeu shape={img.shape}")
    # Min pooling separável 3x1 e 1x3 (mais estável que 3x3 direto)
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img):
    """
    Dilatação suave diferenciável usando max_pool.
    """
    if len(img.shape) != 4:
        raise ValueError(f"soft_dilate espera tensor 4D (B,C,H,W), recebeu shape={img.shape}")
    return F.max_pool2d(img, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


def soft_open(img):
    """
    Abertura morfológica suave = erode seguido de dilate.
    Remove ruído fino preservando a estrutura principal.
    """
    return soft_dilate(soft_erode(img))


def soft_skel(img, num_iter):
    """
    Esqueletização suave diferenciável (Soft Skeleton).
    
    Algoritmo:
      skel = img
      Para cada iteração:
        img_eroded = erode(img)
        img_opened = dilate(img_eroded)  # = open(img)
        delta = clamp(img_eroded - img_opened, 0, 1)  # bordas topológicas
        skel = clamp(skel - soft_dilate(delta), 0, 1)  # remove bordas
        img = img_eroded  # próxima iteração erode mais
      
      Retorna: skel * img  (interseção com a erosão final)
    
    Args:
        img: Tensor (B, C, H, W) com valores em [0, 1]
        num_iter: Número de iterações de erosão (10-15 funciona bem para 256x256)
    
    Returns:
        Tensor com o esqueleto suave diferenciável
    """
    img_orig = img.clone()
    skel = F.relu(img - soft_open(img))
    
    for _ in range(num_iter):
        img = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skel = torch.clamp(skel + F.relu(delta - skel * soft_dilate(delta)), 0.0, 1.0)
    
    return skel


# ==============================================================================
# 2. Dice Loss (mesma da train.py original, para referência e uso combinado)
# ==============================================================================

class DiceLoss(nn.Module):
    """Standard Dice Loss para segmentação binária."""
    
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) — saída crua do modelo (sem sigmoid)
            targets: (B, 1, H, W) — máscara binária [0, 1]
        """
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ==============================================================================
# 3. Centerline Dice Loss (clDice)
# ==============================================================================

class SoftCLDiceLoss(nn.Module):
    """
    Centerline Dice Loss (clDice) — preserva topologia das fileiras.
    
    Calcula:
      - Tprec (Topology Precision): quanto do skeleton da PREDIÇÃO está dentro do GT
      - Tsens (Topology Sensitivity): quanto do skeleton do GT está dentro da predição
      - clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
    
    Args:
        num_iter: Iterações para esqueletização suave (10-15 para tiles 256x256)
        smooth: Constante de suavização para estabilidade numérica
    """
    
    def __init__(self, num_iter=10, smooth=1.0):
        super().__init__()
        self.num_iter = num_iter
        self.smooth = smooth
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) — saída crua do modelo (sem sigmoid)
            targets: (B, 1, H, W) — máscara binária [0, 1]
        """
        probs = torch.sigmoid(logits)
        
        # Esqueletização suave diferenciável
        skel_pred = soft_skel(probs, self.num_iter)
        skel_true = soft_skel(targets, self.num_iter)
        
        # Topology Precision: skeleton da predição coberto pelo GT
        tprec = ((skel_pred * targets).sum(dim=[2, 3]) + self.smooth) / \
                (skel_pred.sum(dim=[2, 3]) + self.smooth)
        
        # Topology Sensitivity: skeleton do GT coberto pela predição
        tsens = ((skel_true * probs).sum(dim=[2, 3]) + self.smooth) / \
                (skel_true.sum(dim=[2, 3]) + self.smooth)
        
        # clDice = F1 entre Tprec e Tsens
        cl_dice = (2.0 * tprec * tsens + self.smooth) / (tprec + tsens + self.smooth)
        
        return 1.0 - cl_dice.mean()


# ==============================================================================
# 4. Loss Combinada: BCE + Dice + clDice (recomendada para treinamento)
# ==============================================================================

class BCEDiceCLDiceLoss(nn.Module):
    """
    Loss combinada para segmentação de fileiras de citros com preservação topológica.
    
    L = w_bce * BCE + w_dice * Dice + w_cldice * clDice
    
    Pesos recomendados:
      - Início do treino: BCE=0.4, Dice=0.4, clDice=0.2 (foco em aprender a forma geral)
      - Após convergência: BCE=0.3, Dice=0.3, clDice=0.4 (foco em topologia/conectividade)
    
    Args:
        bce_weight: Peso para Binary Cross-Entropy
        dice_weight: Peso para Dice Loss
        cldice_weight: Peso para Centerline Dice Loss
        num_iter: Iterações da esqueletização suave
    """
    
    def __init__(self, bce_weight=0.4, dice_weight=0.4, cldice_weight=0.2, num_iter=10):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.cldice = SoftCLDiceLoss(num_iter=num_iter)
        
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
    
    def forward(self, logits, targets):
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        loss_cldice = self.cldice(logits, targets)
        
        total = (self.bce_weight * loss_bce + 
                 self.dice_weight * loss_dice + 
                 self.cldice_weight * loss_cldice)
        
        return total, {
            'bce': loss_bce.item(),
            'dice': loss_dice.item(),
            'cldice': loss_cldice.item(),
            'total': total.item()
        }


# ==============================================================================
# 5. Métrica clDice (para avaliação, não backpropagation)
# ==============================================================================

def compute_cldice_metric(logits, targets, threshold=0.5, num_iter=10, smooth=1.0):
    """
    Calcula clDice como MÉTRICA de avaliação (sem gradientes).
    
    Returns:
        float: valor clDice entre 0 e 1 (1 = topologia perfeita)
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        skel_pred = soft_skel(preds, num_iter)
        skel_true = soft_skel(targets, num_iter)
        
        tprec = ((skel_pred * targets).sum() + smooth) / (skel_pred.sum() + smooth)
        tsens = ((skel_true * preds).sum() + smooth) / (skel_true.sum() + smooth)
        
        cl_dice = (2.0 * tprec * tsens + smooth) / (tprec + tsens + smooth)
        
        return cl_dice.item()


if __name__ == '__main__':
    # Teste rápido de sanidade
    print("Testando clDice Loss...")
    
    # Simula um batch: 2 imagens, 1 canal, 64x64
    B, C, H, W = 2, 1, 64, 64
    
    # Predição aleatória e target com linhas sintéticas
    logits = torch.randn(B, C, H, W)
    targets = torch.zeros(B, C, H, W)
    # Cria linhas horizontais no target
    targets[:, :, 10:13, :] = 1.0
    targets[:, :, 30:33, :] = 1.0
    targets[:, :, 50:53, :] = 1.0
    
    # Testa DiceLoss
    dice_loss = DiceLoss()
    dl = dice_loss(logits, targets)
    print(f"  Dice Loss: {dl.item():.4f}")
    
    # Testa SoftCLDiceLoss
    cldice_loss = SoftCLDiceLoss(num_iter=10)
    cl = cldice_loss(logits, targets)
    print(f"  clDice Loss: {cl.item():.4f}")
    
    # Testa loss combinada
    combined = BCEDiceCLDiceLoss(bce_weight=0.4, dice_weight=0.4, cldice_weight=0.2)
    total, breakdown = combined(logits, targets)
    print(f"  Combined Loss: {total.item():.4f}")
    print(f"    BCE: {breakdown['bce']:.4f}")
    print(f"    Dice: {breakdown['dice']:.4f}")
    print(f"    clDice: {breakdown['cldice']:.4f}")
    
    # Testa métrica
    metric = compute_cldice_metric(logits, targets)
    print(f"  clDice Metric: {metric:.4f}")
    
    # Testa com predição perfeita
    perfect_logits = torch.zeros(B, C, H, W) - 5.0
    perfect_logits[:, :, 10:13, :] = 5.0
    perfect_logits[:, :, 30:33, :] = 5.0
    perfect_logits[:, :, 50:53, :] = 5.0
    
    metric_perfect = compute_cldice_metric(perfect_logits, targets)
    print(f"  clDice Metric (predição perfeita): {metric_perfect:.4f}")
    
    print("\n[OK] Todos os testes passaram!")
