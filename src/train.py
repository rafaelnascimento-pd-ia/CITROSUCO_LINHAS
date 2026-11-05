import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import OrangeDataset, get_transforms
from src.model import get_segmentation_model
from src.losses import BCEDiceCLDiceLoss, compute_cldice_metric



def compute_metrics(logits, targets, threshold=0.5, eps=1e-5):
    """
    Computes IoU (Jaccard), Dice (F1-score) and clDice for binary segmentation.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    # Guarda targets 4D antes do flatten para o clDice
    targets_4d = targets if targets.dim() == 4 else targets.view(logits.shape)
    
    # Flatten spatial dimensions para IoU e Dice
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    
    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    
    dice_union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
    dice = (2. * intersection + eps) / (dice_union + eps)
    
    # clDice metric (topology-aware) — precisa de tensores 4D (B,C,H,W)
    cldice = compute_cldice_metric(logits, targets_4d, threshold=threshold)
    
    return iou.mean().item(), dice.mean().item(), cldice

def train_model(data_dir="data", models_dir="models", encoder_name="resnet18",
                epochs=10, batch_size=2, lr=1e-4, patience=20, min_delta=0.001, device=None, progress_callback=None,
                logger=print, num_orthos=None):
    """
    Trains a UNet++ model and saves the best checkpoint with timestamp and number of orthomosaics.
    """
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(models_dir, exist_ok=True)
    progress_file = os.path.join(data_dir, "training_progress.json")
    
    # Auto-detect number of orthomosaics from dataset_info.json if not provided
    if num_orthos is None:
        info_path = os.path.join(data_dir, "dataset_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    num_orthos = d.get("num_orthomosaics", 1)
            except Exception:
                num_orthos = 1
        else:
            num_orthos = 1
            
    ortho_tag = f"{num_orthos}orthos" if num_orthos > 1 else "1ortho"
    logger(f"Treinamento configurado com {num_orthos} ortomosaico(s). Tag: {ortho_tag}")
    
    # Reset progress file
    with open(progress_file, "w") as f:
        json.dump({"epoch": 0, "total_epochs": epochs, "status": "Iniciando", "history": []}, f)
        
    # Setup Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger(f"Dispositivo selecionado: {device}")
    if device.type == "cuda":
        logger(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        logger(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
    # 1. Load Data Loaders
    logger("Preparando DataLoaders...")
    train_dataset = OrangeDataset(
        images_dir=os.path.join(data_dir, "train", "images"),
        masks_dir=os.path.join(data_dir, "train", "masks"),
        transform=get_transforms(is_train=True)
    )
    
    val_dataset = OrangeDataset(
        images_dir=os.path.join(data_dir, "val", "images"),
        masks_dir=os.path.join(data_dir, "val", "masks"),
        transform=get_transforms(is_train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=0, pin_memory=True if device.type == "cuda" else False
    )
    
    logger(f"  Tamanho do dataset de treino: {len(train_dataset)} tiles ({len(train_loader)} batches)")
    logger(f"  Tamanho do dataset de validação: {len(val_dataset)} tiles ({len(val_loader)} batches)")
    
    # 2. Instantiate Model, Loss and Optimizer
    logger(f"Instanciando UNet++ com encoder {encoder_name}...")
    model = get_segmentation_model(encoder_name=encoder_name)
    model.to(device)
    
    # Loss combinada: BCE + Dice + clDice (preserva topologia das fileiras)
    criterion = BCEDiceCLDiceLoss(
        bce_weight=0.4,   # Binary Cross-Entropy: volume geral
        dice_weight=0.4,   # Dice: sobreposição regional
        cldice_weight=0.2, # clDice: preservação da topologia/conectividade
        num_iter=10        # iterações da esqueletização suave
    )
    logger(f"  Loss: BCE(0.4) + Dice(0.4) + clDice(0.2) [topology-preserving]")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Use PyTorch AMP for Mixed Precision (reduces memory usage and speeds training)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))
    
    best_val_iou = 0.0
    is_best = False
    patience_counter = 0
    history = []
    
    from tqdm import tqdm

    # 3. Training Loop
    logger("Iniciando loop de treinamento...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_iou_accum = 0.0
        train_dice_accum = 0.0
        train_cldice_accum = 0.0
        
        pbar_train = tqdm(train_loader, desc=f"Época {epoch:02d}/{epochs:02d} [Treino]", unit="batch", leave=False)
        for inputs, targets in pbar_train:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)
            targets = targets.float() / 255.0
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(inputs)
                loss, loss_breakdown = criterion(outputs, targets)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_val = loss.item()
            train_loss += loss_val * inputs.size(0)
            
            # Metrics
            iou, dice, cldice_m = compute_metrics(outputs, targets)
            train_iou_accum += iou * inputs.size(0)
            train_dice_accum += dice * inputs.size(0)
            train_cldice_accum += cldice_m * inputs.size(0)

            # Live update in tqdm progress bar
            pbar_train.set_postfix({
                "loss": f"{loss_val:.4f}",
                "iou": f"{iou:.4f}",
                "clD": f"{loss_breakdown['cldice']:.3f}"
            })
            
        train_loss /= len(train_dataset)
        train_iou = train_iou_accum / len(train_dataset)
        train_dice = train_dice_accum / len(train_dataset)
        train_cldice = train_cldice_accum / len(train_dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_iou_accum = 0.0
        val_dice_accum = 0.0
        val_cldice_accum = 0.0
        
        pbar_val = tqdm(val_loader, desc=f"Época {epoch:02d}/{epochs:02d} [Validação]", unit="batch", leave=False)
        with torch.no_grad():
            for inputs, targets in pbar_val:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)
                targets = targets.float() / 255.0
                
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    outputs = model(inputs)
                    loss, loss_breakdown = criterion(outputs, targets)
                    
                loss_val = loss.item()
                val_loss += loss_val * inputs.size(0)
                
                # Metrics
                iou, dice, cldice_m = compute_metrics(outputs, targets)
                val_iou_accum += iou * inputs.size(0)
                val_dice_accum += dice * inputs.size(0)
                val_cldice_accum += cldice_m * inputs.size(0)

                pbar_val.set_postfix({
                    "val_loss": f"{loss_val:.4f}",
                    "val_iou": f"{iou:.4f}",
                    "clD": f"{loss_breakdown['cldice']:.3f}"
                })
                
        val_loss /= len(val_dataset)
        val_iou = val_iou_accum / len(val_dataset)
        val_dice = val_dice_accum / len(val_dataset)
        val_cldice = val_cldice_accum / len(val_dataset)
        
        epoch_time = time.time() - epoch_start
        
        logger(f"Epoca {epoch:02d}/{epochs:02d} ({epoch_time:.1f}s) | "
               f"Train Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Dice: {train_dice:.4f}, clDice: {train_cldice:.4f} | "
               f"Val Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Dice: {val_dice:.4f}, clDice: {val_cldice:.4f}")
               
        # Save best model with unique descriptive name
        is_best = val_iou > (best_val_iou + (min_delta or 0.0))
        if is_best:
            best_val_iou = val_iou
            patience_counter = 0
            best_path = os.path.join(models_dir, "best_model.pth")
            archive_best_path = os.path.join(models_dir, f"best_model_{timestamp}_{ortho_tag}.pth")
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_dice': val_dice,
                'val_cldice': val_cldice,
                'encoder_name': encoder_name,
                'num_orthomosaics': num_orthos,
                'timestamp': timestamp
            }
            torch.save(checkpoint_data, best_path)
            torch.save(checkpoint_data, archive_best_path)
            logger(f"  --> Novo melhor modelo salvo! (Val IoU: {val_iou:.4f})")
            logger(f"      [Arquivo salvo: {archive_best_path}]")
        else:
            patience_counter += 1
            if patience and patience > 0:
                logger(f"  [Early Stopping] Sem melhoria significativa (> {min_delta*100:.2f}%) por {patience_counter}/{patience} época(s).")
            
        # Log to history
        epoch_data = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_iou": float(train_iou),
            "train_dice": float(train_dice),
            "train_cldice": float(train_cldice),
            "val_loss": float(val_loss),
            "val_iou": float(val_iou),
            "val_dice": float(val_dice),
            "val_cldice": float(val_cldice),
            "time": float(epoch_time),
            "is_best": bool(is_best)
        }
        history.append(epoch_data)
        
        # Update progress file for Web UI
        progress_info = {
            "epoch": epoch,
            "total_epochs": epochs,
            "status": "Treinando",
            "history": history
        }
        with open(progress_file, "w") as f:
            json.dump(progress_info, f, indent=2)
            
        if progress_callback:
            progress_callback(progress_info)

        # Early Stopping check
        if patience and patience > 0 and patience_counter >= patience:
            logger(f"\n[EARLY STOPPING ATIVADO] O modelo não apresentou melhoria no Val IoU nas últimas {patience} épocas consecutivas.")
            logger(f"Encerrando treinamento antecipadamente na época {epoch}/{epochs}.")
            break
            
    # Finalize model training
    logger(f"Treinamento concluído. Melhor Val IoU: {best_val_iou:.4f}")

    # 4. Optional Test Set Evaluation (10% test cego)
    test_img_dir = os.path.join(data_dir, "test", "images")
    test_mask_dir = os.path.join(data_dir, "test", "masks")
    test_metrics = {}
    if os.path.exists(test_img_dir) and len(os.listdir(test_img_dir)) > 0:
        logger("\n=== Avaliando Melhor Modelo no Conjunto de TESTE CEGO (10%) ===")
        test_dataset = OrangeDataset(
            images_dir=test_img_dir,
            masks_dir=test_mask_dir,
            transform=get_transforms(is_train=False)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=0, pin_memory=True if device.type == "cuda" else False
        )
        
        # Load best model weights for test evaluation
        best_path = os.path.join(models_dir, "best_model.pth")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            
        model.eval()
        test_loss = 0.0
        test_iou_accum = 0.0
        test_dice_accum = 0.0
        test_cldice_accum = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)
                targets = targets.float() / 255.0
                
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    outputs = model(inputs)
                    loss, _ = criterion(outputs, targets)
                    
                test_loss += loss.item() * inputs.size(0)
                iou, dice, cldice_m = compute_metrics(outputs, targets)
                test_iou_accum += iou * inputs.size(0)
                test_dice_accum += dice * inputs.size(0)
                test_cldice_accum += cldice_m * inputs.size(0)
                
        test_loss /= len(test_dataset)
        test_iou = test_iou_accum / len(test_dataset)
        test_dice = test_dice_accum / len(test_dataset)
        test_cldice = test_cldice_accum / len(test_dataset)
        
        logger(f"  [RESULTADO TESTE CEGO] {len(test_dataset)} tiles avaliados:")
        logger(f"  --> Test Loss: {test_loss:.4f} | Test IoU: {test_iou:.4f} | Test Dice: {test_dice:.4f} | Test clDice: {test_cldice:.4f}\n")
        test_metrics = {
            "test_loss": float(test_loss),
            "test_iou": float(test_iou),
            "test_dice": float(test_dice),
            "test_cldice": float(test_cldice),
            "test_tiles": len(test_dataset)
        }
    
    # Save final model check point with descriptive name
    final_path = os.path.join(models_dir, "model_final.pth")
    archive_final_path = os.path.join(models_dir, f"model_final_{timestamp}_{ortho_tag}.pth")
    final_checkpoint_data = {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'val_iou': val_iou,
        'val_dice': val_dice,
        'test_metrics': test_metrics,
        'encoder_name': encoder_name,
        'num_orthomosaics': num_orthos,
        'timestamp': timestamp
    }
    torch.save(final_checkpoint_data, final_path)
    torch.save(final_checkpoint_data, archive_final_path)
    logger(f"Modelo final salvo em: {final_path}")
    logger(f"Backup do modelo final salvo em: {archive_final_path}")
    
    # Update progress to finished
    progress_info["status"] = "Concluído"
    progress_info["test_metrics"] = test_metrics
    with open(progress_file, "w") as f:
        json.dump(progress_info, f, indent=2)
        
    return archive_best_path if is_best else final_path

if __name__ == "__main__":
    # Test execution
    train_model(epochs=1, batch_size=2)
