import os
import sys
import time
import json
import zipfile
import threading
import shutil
import glob
from flask import Flask, request, jsonify, send_file, Response, send_from_directory
import numpy as np
import rasterio
import rasterio.warp
import geopandas as gpd
import torch

# Adjust path to import src modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.preprocess import run_preprocessing
from src.train import train_model
from src.inference import run_inference
from src.postprocess import run_postprocessing

app = Flask(__name__, static_folder="static")

# Directories setup
UPLOAD_FOLDER = "data/uploads"
OUTPUT_FOLDER = "data/outputs"
DEBUG_FOLDER = "data/debug"
MODELS_FOLDER = "models"

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, DEBUG_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global task state tracking
# Used to communicate progress and logs to the frontend
TASK_STATES = {
    "preprocess": {"status": "idle", "progress": 0, "logs": [], "error": None, "results": None},
    "train": {"status": "idle", "progress": 0, "logs": [], "error": None, "history": []},
    "inference": {"status": "idle", "progress": 0, "logs": [], "error": None, "results": None},
    "postprocess": {"status": "idle", "progress": 0, "logs": [], "error": None, "results": None}
}

# Logger helper that appends to TASK_STATES
def get_logger(task_type):
    def log_msg(msg):
        timestamp = time.strftime("[%H:%M:%S]")
        full_msg = f"{timestamp} {msg}"
        print(f"[{task_type.upper()}] {msg}")
        TASK_STATES[task_type]["logs"].append(full_msg)
        if len(TASK_STATES[task_type]["logs"]) > 800:
            TASK_STATES[task_type]["logs"].pop(0)
    return log_msg

# Helper: calculate tile bounds in Web Mercator (EPSG:3857)
def tile_bounds_3857(x, y, z):
    num_tiles = 1 << z
    # Circumference of the Earth at equator
    circ = 40075016.68557849
    tile_size = circ / num_tiles
    
    # Coordinates of top-left corner
    min_x = -20037508.342789244 + x * tile_size
    max_x = -20037508.342789244 + (x + 1) * tile_size
    max_y = 20037508.342789244 - y * tile_size
    min_y = 20037508.342789244 - (y + 1) * tile_size
    
    return min_x, min_y, max_x, max_y

# Shared GPKG raster handles to avoid opening/closing on every tile request
gpkg_handles = {}
gpkg_lock = threading.Lock()

def get_gpkg_src(gpkg_filename):
    with gpkg_lock:
        if gpkg_filename not in gpkg_handles:
            if os.path.exists(gpkg_filename):
                gpkg_handles[gpkg_filename] = rasterio.open(gpkg_filename)
            else:
                print(f"[WARN] File '{gpkg_filename}' not found!")
                return None
        return gpkg_handles[gpkg_filename]


def load_prob_map_from_tif(path):
    with rasterio.open(path) as src:
        prob_map = src.read(1).astype(np.float32)
        if prob_map.max() > 1.0:
            prob_map = prob_map / 255.0
        return prob_map, src.transform, src.crs

# Clean up raster handles on exit
import atexit
@atexit.register
def close_gpkg_handles():
    with gpkg_lock:
        for src in gpkg_handles.values():
            src.close()
        gpkg_handles.clear()

# --- UI Server ---

@app.route('/')
def index():
    return send_from_directory("static", "index.html")

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory("static", path)

# --- GPKG Discovery ---

@app.route('/api/list-gpkgs')
def list_gpkgs():
    gpkgs = glob.glob("*.gpkg")
    return jsonify({"gpkgs": gpkgs})

# --- Tile Server Endpoint (On-the-fly reprojected raster tiles) ---

@app.route('/api/tiles/<gpkg_name>/<int:z>/<int:x>/<int:y>')
def serve_tile(gpkg_name, z, x, y):
    src = get_gpkg_src(gpkg_name)
    if src is None:
        return "GeoPackage not found", 404
        
    try:
        # 1. Get Web Mercator bounds of the requested Leaflet tile
        min_x, min_y, max_x, max_y = tile_bounds_3857(x, y, z)
        
        # 2. Check if the requested tile overlaps the raster bounds in Web Mercator
        # Reproject GPKG bounds to Web Mercator for quick overlap checking
        # GPKG is in EPSG:32722. Let's do a quick check
        # (For speed, we let rasterio.warp.reproject handle it. If it doesn't overlap,
        # out_data will remain filled with 0 / transparency)
        
        out_shape = (4, 256, 256) # RGBA output tile
        out_data = np.zeros(out_shape, dtype=np.uint8)
        
        dst_transform = rasterio.transform.from_bounds(min_x, min_y, max_x, max_y, 256, 256)
        
        with gpkg_lock:
            # Warp from UTM (32722) to Web Mercator (3857)
            # Read first 3 bands (RGB)
            rasterio.warp.reproject(
                source=rasterio.band(src, [1, 2, 3]),
                destination=out_data[:3], # Write to RGB
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs='EPSG:3857',
                resampling=rasterio.warp.Resampling.bilinear,
                src_nodata=0,
                dst_nodata=0
            )
            
        # Create alpha channel: 255 where there is image data, 0 where there is black/nodata
        # If all 3 channels are 0, set alpha to 0 (transparent)
        mask = (out_data[0] > 0) | (out_data[1] > 0) | (out_data[2] > 0)
        out_data[3][mask] = 255
        
        # If tile is completely empty, we can return 204 or a blank transparent PNG
        if not mask.any():
            # Return a blank transparent 1x1 png or just let cv2 encode the empty 256x256
            pass
            
        # Encode as PNG using memory file
        import cv2
        # Convert RGBA to BGRA for cv2 encoding
        bgra = cv2.merge([out_data[2], out_data[1], out_data[0], out_data[3]])
        _, img_encoded = cv2.imencode('.png', bgra)
        
        return Response(img_encoded.tobytes(), mimetype='image/png')
        
    except Exception as e:
        # print(f"Error serving tile {z}/{x}/{y}: {e}")
        # Return transparent PNG on error or out of bounds
        empty_tile = np.zeros((256, 256, 4), dtype=np.uint8)
        _, img_encoded = cv2.imencode('.png', empty_tile)
        return Response(img_encoded.tobytes(), mimetype='image/png')

# --- GeoJSON Layer Serving ---

@app.route('/api/vector/<layer>')
def get_vector_layer(layer):
    """
    Returns layers (talhao, linhas_centrais, or predicoes) as GeoJSON.
    """
    try:
        if layer == "talhao":
            path = glob.glob(os.path.join(UPLOAD_FOLDER, "talhao.*"))
        elif layer == "linhas":
            path = glob.glob(os.path.join(UPLOAD_FOLDER, "linhas_centrais.*"))
        elif layer == "predicoes":
            path = [os.path.join(OUTPUT_FOLDER, "predicao_linhas.geojson")]
        elif layer == "synthetic_talhao":
            path = [os.path.join("data/synthetic", "talhao.geojson")]
        elif layer == "synthetic_linhas":
            path = [os.path.join("data/synthetic", "linhas_centrais.geojson")]
        else:
            return jsonify({"error": f"Layer '{layer}' inválido!"}), 400
            
        # Find files matching the request (we want to support shapefile or geojson uploads)
        geojson_file = None
        for p in path:
            if p.endswith('.geojson'):
                geojson_file = p
                break
                
        # If it's a shapefile, load it in geopandas and export to geojson string
        if not geojson_file:
            shp_files = [p for p in path if p.endswith('.shp')]
            if shp_files:
                gdf = gpd.read_file(shp_files[0])
                # Ensure CRS is reprojected to WGS 84 for Leaflet
                if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
                return Response(gdf.to_json(), mimetype='application/json')
            else:
                return jsonify({"status": "empty", "message": f"Nenhum vetor carregado para a camada: {layer}"}), 200
                
        # Return GeoJSON directly from disk
        with open(geojson_file, "r") as f:
            geojson_data = json.load(f)
        return jsonify(geojson_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- File Upload ---

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """
    Handles upload of talhao (boundary) and linhas_centrais (centerlines) vectors.
    Supports .geojson or zipped shapefiles.
    """
    if 'file' not in request.files or 'type' not in request.form:
        return jsonify({"error": "Parâmetros inválidos. Envie 'file' e 'type' ('talhao', 'linhas' ou 'path_linhas')."}), 400
        
    file = request.files['file']
    vector_type = request.form['type']
    
    if file.filename == '':
        return jsonify({"error": "Nome de arquivo vazio!"}), 400
        
    if vector_type not in ["talhao", "linhas", "path_linhas"]:
        return jsonify({"error": "Tipo de vetor inválido! Use 'talhao', 'linhas' ou 'path_linhas'."}), 400
        
    filename = file.filename.lower()
    
    # Target prefix based on type
    if vector_type == "talhao":
        prefix = "talhao"
    elif vector_type == "linhas":
        prefix = "linhas_centrais"
    else:
        prefix = "path_linhas"
    
    try:
        # Clear previous files of this type
        for f in glob.glob(os.path.join(UPLOAD_FOLDER, f"{prefix}.*")):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)
                
        if filename.endswith('.geojson'):
            dest_path = os.path.join(UPLOAD_FOLDER, f"{prefix}.geojson")
            file.save(dest_path)
            # Verify file structure
            gpd.read_file(dest_path)
            return jsonify({"status": "success", "message": f"Vetor '{vector_type}' salvo com sucesso como GeoJSON."})
            
        elif filename.endswith('.zip'):
            # Unzip and look for shapefiles
            zip_path = os.path.join(UPLOAD_FOLDER, f"temp_{prefix}.zip")
            file.save(zip_path)
            
            extract_dir = os.path.join(UPLOAD_FOLDER, f"temp_{prefix}")
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
            # Search for .shp file
            shp_files = []
            for root, dirs, files in os.walk(extract_dir):
                for f in files:
                    if f.lower().endswith('.shp'):
                        shp_files.append(os.path.join(root, f))
                        
            if not shp_files:
                raise ValueError("Nenhum arquivo .shp encontrado dentro do arquivo ZIP!")
                
            # Rename shapefile components to standard prefix
            shp_base = os.path.splitext(shp_files[0])[0]
            extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg']
            for ext in extensions:
                src_file = shp_base + ext
                if os.path.exists(src_file):
                    shutil.copy(src_file, os.path.join(UPLOAD_FOLDER, f"{prefix}{ext}"))
                    
            # Clean up temp
            os.remove(zip_path)
            shutil.rmtree(extract_dir)
            
            # Verify shapefile
            gpd.read_file(os.path.join(UPLOAD_FOLDER, f"{prefix}.shp"))
            return jsonify({"status": "success", "message": f"Shapefile '{vector_type}' descompactado e salvo com sucesso."})
            
        else:
            return jsonify({"error": "Formato inválido. Use .geojson ou .zip (para Shapefile)."}), 400
            
    except Exception as e:
        return jsonify({"error": f"Falha ao validar arquivo: {str(e)}"}), 500

@app.route('/api/upload-mask', methods=['POST'])
def upload_mask():
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado!"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nome de arquivo vazio!"}), 400
    if not file.filename.lower().endswith('.tif'):
        return jsonify({"error": "O arquivo deve ter extensão .tif!"}), 400

    try:
        dest_path = os.path.join(UPLOAD_FOLDER, "input_mask.tif")
        file.save(dest_path)
        return jsonify({"status": "success", "message": "Máscara carregada com sucesso."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Process Status APIs ---

@app.route('/api/status')
def get_all_status():
    # If training is active, read progress from JSON file
    progress_file = "data/training_progress.json"
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                train_progress = json.load(f)
                TASK_STATES["train"]["progress"] = int((train_progress["epoch"] / train_progress["total_epochs"]) * 100)
                TASK_STATES["train"]["status"] = train_progress["status"]
                TASK_STATES["train"]["history"] = train_progress["history"]
        except Exception:
            pass
            
    # Check if predictions are ready
    predictions_ready = os.path.exists(os.path.join(OUTPUT_FOLDER, "predicao_linhas.geojson"))
    
    return jsonify({
        "tasks": TASK_STATES,
        "predictions_ready": predictions_ready,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    })

# --- SSE Logs Streaming ---

@app.route('/api/stream-logs/<task_type>')
def stream_logs(task_type):
    """
    Server-Sent Events endpoint to stream logs to the client UI.
    """
    if task_type not in TASK_STATES:
        return "Invalid task type", 400
        
    def event_stream():
        last_idx = 0
        while True:
            logs = TASK_STATES[task_type]["logs"]
            if len(logs) > last_idx:
                for i in range(last_idx, len(logs)):
                    yield f"data: {logs[i]}\n\n"
                last_idx = len(logs)
            time.sleep(0.5)
            
    return Response(event_stream(), mimetype="text/event-stream")

# --- Pipeline Control Threads ---

def preprocess_worker(gpkg_filename, buffer_size, tile_size, stride, append):
    logger = get_logger("preprocess")
    TASK_STATES["preprocess"]["status"] = "running"
    TASK_STATES["preprocess"]["error"] = None
    TASK_STATES["preprocess"]["progress"] = 10
    
    try:
        # Base name of the GPKG without extension (e.g., '64000')
        gpkg_basename = os.path.splitext(gpkg_filename)[0]
        
        # 1. Try to find uploaded files first
        talhao_files = glob.glob(os.path.join(UPLOAD_FOLDER, "talhao.geojson")) + glob.glob(os.path.join(UPLOAD_FOLDER, "talhao.shp"))
        lines_files = glob.glob(os.path.join(UPLOAD_FOLDER, "linhas_centrais.geojson")) + glob.glob(os.path.join(UPLOAD_FOLDER, "linhas_centrais.shp"))
            
        # 2. Fallback to auto-detecting from data_set folder based on GPKG name
        talhao_path = None
        lines_path = None
        
        if talhao_files:
            talhao_path = talhao_files[0]
        else:
            # Check data_set folder and VETORES subfolder for geojson or shp
            patterns = [
                os.path.join("data_set", gpkg_basename, "VETORES", "*talho*.geojson"),
                os.path.join("data_set", gpkg_basename, "VETORES", "*talho*.shp"),
                os.path.join("data_set", gpkg_basename, "*talho*.geojson"),
                os.path.join("data_set", gpkg_basename, "*talho*.shp"),
            ]
            ds_talhao = []
            for pat in patterns:
                ds_talhao = glob.glob(pat)
                if ds_talhao:
                    break
            if ds_talhao:
                talhao_path = ds_talhao[0]
                logger(f"Talhão auto-detectado: {talhao_path}")
                
        if lines_files:
            lines_path = lines_files[0]
        else:
            # Check data_set folder and VETORES subfolder for geojson or shp
            patterns = [
                os.path.join("data_set", gpkg_basename, "VETORES", "*linha*.geojson"),
                os.path.join("data_set", gpkg_basename, "VETORES", "*linha*.shp"),
                os.path.join("data_set", gpkg_basename, "*linha*.geojson"),
                os.path.join("data_set", gpkg_basename, "*linha*.shp"),
            ]
            ds_linhas = []
            for pat in patterns:
                ds_linhas = glob.glob(pat)
                if ds_linhas:
                    break
            if ds_linhas:
                lines_path = ds_linhas[0]
                logger(f"Linhas auto-detectadas: {lines_path}")
                
        # 3. Last fallback: synthetic data
        if not talhao_path or not lines_path:
            logger("[WARN] Vetores não encontrados (nem em uploads, nem em data_set/). Usando vetores sintéticos.")
            if not os.path.exists("data/synthetic/talhao.geojson"):
                from src.generate_synthetic_data import generate_data
                generate_data()
            talhao_path = "data/synthetic/talhao.geojson"
            lines_path = "data/synthetic/linhas_centrais.geojson"
            
        TASK_STATES["preprocess"]["progress"] = 30
        
        results = run_preprocessing(
            raster_path=gpkg_filename,
            lines_path=lines_path,
            talhao_path=talhao_path,
            buffer_size=buffer_size,
            tile_size=tile_size,
            stride=stride,
            append=append,
            logger=logger
        )
        
        # Only log 100% success at the end
        TASK_STATES["preprocess"]["progress"] = 100
        TASK_STATES["preprocess"]["status"] = "success"
        TASK_STATES["preprocess"]["results"] = results
        logger("Preprocessamento finalizado com sucesso!")
        
    except Exception as e:
        import traceback
        err_msg = str(e)
        logger(f"ERRO no preprocessamento: {err_msg}")
        logger(traceback.format_exc())
        TASK_STATES["preprocess"]["status"] = "failed"
        TASK_STATES["preprocess"]["error"] = err_msg

@app.route('/api/start-preprocess', methods=['POST'])
def start_preprocess():
    if TASK_STATES["preprocess"]["status"] == "running":
        return jsonify({"error": "Preprocessamento já em andamento!"}), 400
        
    data = request.json or {}
    gpkg_filename = data.get("gpkg", "63996.gpkg")
    buffer_size = float(data.get("buffer_size", 1.5))
    tile_size = int(data.get("tile_size", 256))
    # Stride is half of tile_size for overlap, or equal for no overlap
    stride = int(data.get("stride", tile_size // 2))
    append = bool(data.get("append", False))
    
    # Clear logs
    TASK_STATES["preprocess"]["logs"] = []
    
    thread = threading.Thread(target=preprocess_worker, args=(gpkg_filename, buffer_size, tile_size, stride, append))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started"})

def train_worker(encoder_name, epochs, batch_size, lr):
    logger = get_logger("train")
    TASK_STATES["train"]["status"] = "running"
    TASK_STATES["train"]["error"] = None
    TASK_STATES["train"]["progress"] = 5
    
    try:
        # Check if tiles are generated
        train_img_dir = "data/train/images"
        if not os.path.exists(train_img_dir) or len(glob.glob(os.path.join(train_img_dir, "*.png"))) == 0:
            raise ValueError("Dados de treino não encontrados! Execute o Pré-processamento primeiro.")
            
        train_model(
            data_dir="data",
            models_dir=MODELS_FOLDER,
            encoder_name=encoder_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            logger=logger
        )
        
        TASK_STATES["train"]["progress"] = 100
        TASK_STATES["train"]["status"] = "success"
        logger("Treinamento finalizado com sucesso!")
        
    except Exception as e:
        import traceback
        err_msg = str(e)
        logger(f"ERRO no treinamento: {err_msg}")
        logger(traceback.format_exc())
        TASK_STATES["train"]["status"] = "failed"
        TASK_STATES["train"]["error"] = err_msg

@app.route('/api/start-train', methods=['POST'])
def start_train():
    if TASK_STATES["train"]["status"] == "running":
        return jsonify({"error": "Treinamento já em andamento!"}), 400
        
    data = request.json or {}
    encoder_name = data.get("encoder", "resnet18")
    epochs = int(data.get("epochs", 10))
    batch_size = int(data.get("batch_size", 2))
    lr = float(data.get("lr", 1e-4))
    
    # Clear logs
    TASK_STATES["train"]["logs"] = []
    TASK_STATES["train"]["history"] = []
    
    thread = threading.Thread(target=train_worker, args=(encoder_name, epochs, batch_size, lr))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started"})

def inference_worker(gpkg_filename, checkpoint_type, tile_size, stride, threshold, min_area, close_size, open_size, skip_postprocess, export_as_lines=False, profile="tiro_reto", calc_entrelinhas=False, extend_to_boundary=None, min_row_spacing_m=4.0, max_row_spacing_m=8.5):
    logger = get_logger("inference")
    TASK_STATES["inference"]["status"] = "running"
    TASK_STATES["inference"]["error"] = None
    TASK_STATES["inference"]["progress"] = 10
    
    try:
        gpkg_basename = os.path.splitext(gpkg_filename)[0]
        
        # Determine checkpoint path
        if checkpoint_type == "best":
            checkpoint_path = "models/best_model.pth"
        else:
            checkpoint_path = "models/model_final.pth"
            
        if not os.path.exists(checkpoint_path):
            pths = glob.glob("models/*.pth")
            if pths:
                checkpoint_path = pths[0]
                logger(f"[WARN] Checkpoint selecionado não encontrado. Usando alternativo: {checkpoint_path}")
            else:
                raise ValueError("Nenhum checkpoint (.pth) encontrado! Treine um modelo primeiro ou envie um arquivo .pth.")
                
        # 1. Try uploaded talhao first
        talhao_files = glob.glob(os.path.join(UPLOAD_FOLDER, "talhao.geojson")) + glob.glob(os.path.join(UPLOAD_FOLDER, "talhao.shp"))
        
        talhao_path = None
        if talhao_files:
            talhao_path = talhao_files[0]
        else:
            patterns = [
                os.path.join("data_set", gpkg_basename, "VETORES", "*talho*.geojson"),
                os.path.join("data_set", gpkg_basename, "VETORES", "*talho*.shp"),
                os.path.join("data_set", gpkg_basename, "*talho*.geojson"),
                os.path.join("data_set", gpkg_basename, "*talho*.shp"),
            ]
            ds_talhao = []
            for pat in patterns:
                ds_talhao = glob.glob(pat)
                if ds_talhao:
                    break
            if ds_talhao:
                talhao_path = ds_talhao[0]
                logger(f"Talhão auto-detectado: {talhao_path}")
            
        if not talhao_path:
            logger("[WARN] Vetor de Talhão não encontrado. Usando talhão sintético.")
            talhao_path = "data/synthetic/talhao.geojson"
            
        # Check for uploaded path_linhas vector
        sentido_files = glob.glob(os.path.join(UPLOAD_FOLDER, "path_linhas.geojson")) + glob.glob(os.path.join(UPLOAD_FOLDER, "path_linhas.shp"))
        path_linhas = sentido_files[0] if sentido_files else None
            
        TASK_STATES["inference"]["progress"] = 25
        
        # 1. Slide window prediction (Inference)
        prob_map, cropped_transform, raster_crs, talhao_geom = run_inference(
            raster_path=gpkg_filename,
            talhao_path=talhao_path,
            checkpoint_path=checkpoint_path,
            tile_size=tile_size,
            stride=stride,
            logger=logger
        )
        
        TASK_STATES["inference"]["progress"] = 75
        
        # 2. Post-processing and exporting
        results = run_postprocessing(
            prob_map=prob_map,
            transform=cropped_transform,
            crs=raster_crs,
            talhao_geom=talhao_geom,
            output_dir="data",
            threshold=threshold,
            min_area_m2=min_area,
            close_size=close_size,
            open_size=open_size,
            skip_postprocess=skip_postprocess,
            export_as_lines=export_as_lines,
            path_linhas=path_linhas,
            profile=profile,
            calc_entrelinhas=calc_entrelinhas,
            extend_to_boundary=extend_to_boundary,
            min_row_spacing_m=min_row_spacing_m,
            max_row_spacing_m=max_row_spacing_m,
            logger=logger
        )
        
        TASK_STATES["inference"]["progress"] = 100
        TASK_STATES["inference"]["status"] = "success"
        TASK_STATES["inference"]["results"] = results
        logger(f"Inferência e pós-processamento concluídos com sucesso [Perfil: {profile}]!")
        
    except Exception as e:
        import traceback
        err_msg = str(e)
        logger(f"ERRO na inferência: {err_msg}")
        logger(traceback.format_exc())
        TASK_STATES["inference"]["status"] = "failed"
        TASK_STATES["inference"]["error"] = err_msg
 
@app.route('/api/start-inference', methods=['POST'])
def start_inference():
    if TASK_STATES["inference"]["status"] == "running":
        return jsonify({"error": "Inferência já em andamento!"}), 400
        
    data = request.json or {}
    gpkg_filename = data.get("gpkg", "63996.gpkg")
    checkpoint_type = data.get("checkpoint", "best")
    tile_size = int(data.get("tile_size", 256))
    stride = int(data.get("stride", tile_size // 2))
    
    # Post-process parameters
    threshold = float(data.get("threshold", 0.5))
    min_area = float(data.get("min_area", 10.0))
    close_size = int(data.get("close_size", 5))
    open_size = int(data.get("open_size", 3))
    
    skip_postprocess = bool(data.get("skip_postprocess", False))
    export_as_lines = bool(data.get("export_as_lines", False))
    profile = str(data.get("profile", "tiro_reto"))
    calc_entrelinhas = bool(data.get("calc_entrelinhas", False))
    extend_to_boundary = data.get("extend_to_boundary", None)
    min_row_spacing_m = float(data.get("min_row_spacing_m", 4.0))
    max_row_spacing_m = float(data.get("max_row_spacing_m", 8.5))
    
    # Clear logs
    TASK_STATES["inference"]["logs"] = []
    
    thread = threading.Thread(target=inference_worker, args=(
        gpkg_filename, checkpoint_type, tile_size, stride, threshold, min_area, close_size, open_size, skip_postprocess, export_as_lines, profile, calc_entrelinhas, extend_to_boundary, min_row_spacing_m, max_row_spacing_m
    ))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started"})


def postprocess_worker(input_raster_path, talhao_path, threshold, min_area, close_size, open_size, skip_postprocess, export_as_lines=False, profile="tiro_reto", calc_entrelinhas=False, extend_to_boundary=None, min_row_spacing_m=4.0, max_row_spacing_m=8.5):
    logger = get_logger("postprocess")
    TASK_STATES["postprocess"]["status"] = "running"
    TASK_STATES["postprocess"]["error"] = None
    TASK_STATES["postprocess"]["progress"] = 10

    try:
        if not os.path.exists(input_raster_path):
            raise ValueError(f"Arquivo de máscara não encontrado: {input_raster_path}")

        prob_map, transform, raster_crs = load_prob_map_from_tif(input_raster_path)
        logger(f"Máscara carregada de: {input_raster_path}")

        if not talhao_path or not os.path.exists(talhao_path):
            logger("[WARN] Talhão não encontrado. Usando talhão sintético.")
            talhao_path = "data/synthetic/talhao.geojson"

        talhao_gdf = gpd.read_file(talhao_path).to_crs(raster_crs)
        talhao_geom = talhao_gdf.unary_union

        # Check for uploaded path_linhas vector
        sentido_files = glob.glob(os.path.join(UPLOAD_FOLDER, "path_linhas.geojson")) + glob.glob(os.path.join(UPLOAD_FOLDER, "path_linhas.shp"))
        path_linhas = sentido_files[0] if sentido_files else None

        TASK_STATES["postprocess"]["progress"] = 40

        results = run_postprocessing(
            prob_map=prob_map,
            transform=transform,
            crs=raster_crs,
            talhao_geom=talhao_geom,
            output_dir="data",
            threshold=threshold,
            min_area_m2=min_area,
            close_size=close_size,
            open_size=open_size,
            skip_postprocess=skip_postprocess,
            export_as_lines=export_as_lines,
            path_linhas=path_linhas,
            profile=profile,
            calc_entrelinhas=calc_entrelinhas,
            extend_to_boundary=extend_to_boundary,
            min_row_spacing_m=min_row_spacing_m,
            max_row_spacing_m=max_row_spacing_m,
            logger=logger
        )
        TASK_STATES["postprocess"]["progress"] = 100
        TASK_STATES["postprocess"]["status"] = "success"
        TASK_STATES["postprocess"]["results"] = results
        logger(f"Pós-processamento concluído com sucesso [Perfil: {profile}]!")

    except Exception as e:
        import traceback
        err_msg = str(e)
        logger(f"ERRO no pós-processamento: {err_msg}")
        logger(traceback.format_exc())
        TASK_STATES["postprocess"]["status"] = "failed"
        TASK_STATES["postprocess"]["error"] = err_msg


@app.route('/api/start-postprocess', methods=['POST'])
def start_postprocess():
    if TASK_STATES["postprocess"]["status"] == "running":
        return jsonify({"error": "Pós-processamento já em andamento!"}), 400

    data = request.json or {}
    input_raster = data.get("input_raster", os.path.join(UPLOAD_FOLDER, "input_mask.tif"))
    talhao_path = data.get("talhao_path", None)
    threshold = float(data.get("threshold", 0.5))
    min_area = float(data.get("min_area", 10.0))
    close_size = int(data.get("close_size", 5))
    open_size = int(data.get("open_size", 3))
    skip_postprocess = bool(data.get("skip_postprocess", False))
    export_as_lines = bool(data.get("export_as_lines", False))
    profile = str(data.get("profile", "tiro_reto"))
    calc_entrelinhas = bool(data.get("calc_entrelinhas", False))
    extend_to_boundary = data.get("extend_to_boundary", None)
    min_row_spacing_m = float(data.get("min_row_spacing_m", 4.0))
    max_row_spacing_m = float(data.get("max_row_spacing_m", 8.5))

    TASK_STATES["postprocess"]["logs"] = []

    thread = threading.Thread(target=postprocess_worker, args=(
        input_raster, talhao_path, threshold, min_area, close_size, open_size, skip_postprocess, export_as_lines, profile, calc_entrelinhas, extend_to_boundary, min_row_spacing_m, max_row_spacing_m
    ))
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started"})


# --- Export & Download APIs ---

@app.route('/api/download/model/<model_type>')
def download_model(model_type):
    filename = "best_model.pth" if model_type == "best" else "model_final.pth"
    path = os.path.join(MODELS_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "Model file not found", 404

@app.route('/api/download/raster')
def download_raster():
    path = os.path.join(OUTPUT_FOLDER, "predicao_linhas.tif")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name="predicao_linhas.tif")
    return "Raster output not found", 404

@app.route('/api/download/vector/geojson')
def download_vector_geojson():
    path = os.path.join(OUTPUT_FOLDER, "predicao_linhas.geojson")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name="predicao_linhas.geojson")
    return "Vector GeoJSON output not found", 404

@app.route('/api/download/vector/entrelinhas_geojson')
def download_vector_entrelinhas_geojson():
    path = os.path.join(OUTPUT_FOLDER, "predicao_entrelinhas.geojson")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name="predicao_entrelinhas.geojson")
    return "Vector Entrelinhas GeoJSON not found", 404

@app.route('/api/download/vector/shapefile')
def download_vector_shapefile():
    """
    Zips shapefile components and downloads them.
    """
    shp_base_dir = os.path.join(OUTPUT_FOLDER, "shp")
    zip_path = os.path.join(OUTPUT_FOLDER, "predicao_linhas_shp.zip")
    
    shp_file = os.path.join(shp_base_dir, "predicao_linhas.shp")
    if not os.path.exists(shp_file):
        return "Shapefile not found", 404
        
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for root, dirs, files in os.walk(shp_base_dir):
                for f in files:
                    if f.startswith("predicao_linhas."):
                        zip_ref.write(os.path.join(root, f), f)
                        
        return send_file(zip_path, as_attachment=True, download_name="predicao_linhas_shp.zip")
    except Exception as e:
        return f"Error creating zip: {str(e)}", 500

@app.route('/api/download/vector/entrelinhas_shapefile')
def download_vector_entrelinhas_shapefile():
    """
    Zips entrelinhas shapefile components and downloads them.
    """
    shp_base_dir = os.path.join(OUTPUT_FOLDER, "shp")
    zip_path = os.path.join(OUTPUT_FOLDER, "predicao_entrelinhas_shp.zip")
    
    shp_file = os.path.join(shp_base_dir, "predicao_entrelinhas.shp")
    if not os.path.exists(shp_file):
        return "Shapefile not found", 404
        
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for root, dirs, files in os.walk(shp_base_dir):
                for f in files:
                    if f.startswith("predicao_entrelinhas."):
                        zip_ref.write(os.path.join(root, f), f)
                        
        return send_file(zip_path, as_attachment=True, download_name="predicao_entrelinhas_shp.zip")
    except Exception as e:
        return f"Error creating zip: {str(e)}", 500

# --- Checkpoint Management ---

@app.route('/api/upload-checkpoint', methods=['POST'])
def upload_checkpoint():
    """
    Allows user to upload an external pretrained model checkpoint .pth
    """
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado!"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nome de arquivo vazio!"}), 400
    if not file.filename.endswith('.pth'):
        return jsonify({"error": "O arquivo deve ter extensão .pth!"}), 400
        
    try:
        # Save to models folder as uploaded_model.pth
        path = os.path.join(MODELS_FOLDER, "best_model.pth")
        # Overwrite or back up
        file.save(path)
        return jsonify({"status": "success", "message": "Checkpoint .pth carregado com sucesso como 'best_model.pth'."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Local web app run
    app.run(host='127.0.0.1', port=5000, debug=True)
