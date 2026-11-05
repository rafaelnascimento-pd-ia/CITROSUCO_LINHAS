# -*- coding: utf-8 -*-
"""
Monitor em tempo real do treinamento de IA (Citrosuco)
Uso: .venv\\Scripts\\python.exe scripts/monitor_treinamento.py
"""
import os
import sys
import time
import json
import subprocess

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            encoding='utf-8', timeout=2
        ).strip()
        parts = [p.strip() for p in out.split(',')]
        if len(parts) >= 5:
            return {
                "name": parts[0],
                "mem_used": f"{parts[1]} MB",
                "mem_total": f"{parts[2]} MB",
                "gpu_util": f"{parts[3]}%",
                "temp": f"{parts[4]}°C"
            }
    except Exception:
        pass
    return None

def main():
    progress_file = os.path.join("data", "training_progress.json")
    print("Iniciando monitor de treinamento... Pressione Ctrl + C para sair.\n")
    
    while True:
        clear_screen()
        print("=" * 80)
        print("         MONITORAMENTO EM TEMPO REAL - IA LINHAS CITROSUCO")
        print("=" * 80)
        
        gpu = get_gpu_info()
        if gpu:
            print(f" GPU: {gpu['name']} | Uso: {gpu['gpu_util']} | VRAM: {gpu['mem_used']}/{gpu['mem_total']} | Temp: {gpu['temp']}")
            print("-" * 80)

        if not os.path.exists(progress_file):
            print("[INFO] Aguardando criação do arquivo de progresso (data/training_progress.json)...")
            time.sleep(5)
            continue
            
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[AVISO] Lendo arquivo... ({e})")
            time.sleep(3)
            continue
            
        status = data.get("status", "Desconhecido")
        curr_epoch = data.get("epoch", 0)
        total_epochs = data.get("total_epochs", 60)
        history = data.get("history", [])
        
        progress_pct = (curr_epoch / total_epochs) * 100 if total_epochs > 0 else 0
        bar_len = 35
        filled_len = int(bar_len * curr_epoch // total_epochs) if total_epochs > 0 else 0
        bar = "█" * filled_len + "░" * (bar_len - filled_len)
        
        print(f" Status: {status.upper()} | Época: {curr_epoch}/{total_epochs} ({progress_pct:.1f}%)")
        print(f" Progresso: [{bar}]")
        print("-" * 80)
        
        if history:
            best_epoch_data = max(history, key=lambda x: x.get("val_iou", 0))
            best_cldice = best_epoch_data.get('val_cldice', 0) * 100
            print(f" 🏆 MELHOR MODELO: Época {best_epoch_data['epoch']} | Val IoU: {best_epoch_data['val_iou']*100:.2f}% | Val Dice: {best_epoch_data['val_dice']*100:.2f}% | Val clDice: {best_cldice:.2f}%")
            print("-" * 88)
            print(f" {'ÉPOCA':<6} | {'TR LOSS':<8} | {'TR IOU':<7} | {'VAL LOSS':<9} | {'VAL IOU':<8} | {'VAL DICE':<9} | {'VAL CLDICE':<11} | {'TEMPO':<7} | {'STATUS'}")
            print("-" * 88)
            
            # Mostra as últimas 12 épocas
            for h in history[-12:]:
                ep = h.get("epoch", 0)
                t_loss = h.get("train_loss", 0)
                t_iou = h.get("train_iou", 0) * 100
                v_loss = h.get("val_loss", 0)
                v_iou = h.get("val_iou", 0) * 100
                v_dice = h.get("val_dice", 0) * 100
                v_cldice = h.get("val_cldice", 0) * 100
                t_sec = h.get("time", 0)
                mins = int(t_sec // 60)
                secs = int(t_sec % 60)
                tag = "⭐ RECORDE" if h.get("is_best") else ""
                print(f" {ep:02d}/{total_epochs:02d}  | {t_loss:<8.4f} | {t_iou:>5.1f}%  | {v_loss:<9.4f} | {v_iou:>6.2f}%  | {v_dice:>6.2f}%  | {v_cldice:>7.2f}%    | {mins:02d}m{secs:02d}s  | {tag}")
        else:
            print("[INFO] Histórico de épocas sendo gerado...")
            
        print("=" * 80)
        print("Atualizando a cada 5 segundos... (Pressione Ctrl + C para encerrar o monitor)")
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nMonitor encerrado.")
