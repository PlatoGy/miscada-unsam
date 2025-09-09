from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
import shutil
import os
import uuid
import subprocess
import traceback
from resize import resize_image

# PyTorch CUDA Setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
STATIC_OUTPUT_DIR = "static/outputs"
CHECKPOINT = "checkpoints/unsam_sa1b_4perc_ckpt_200k.pth"
MODEL_SCRIPT = "whole_image_segmentation/demo_whole_image.py"
CONFIG_FILE = "whole_image_segmentation/configs/maskformer2_R50_infer_light.yaml"


os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STATIC_OUTPUT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/unsam")
async def segment_image(file: UploadFile = File(...)):
    # Store the uploaded file
    image_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
    output_path = os.path.join(STATIC_OUTPUT_DIR, f"{image_id}_out.jpg")

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    resize_image(image_path, max_size=512)

    cmd = [
        "python", MODEL_SCRIPT,
        "--config-file", CONFIG_FILE,
        "--input", image_path,
        "--output", output_path,
        "--opts",
        "MODEL.WEIGHTS", CHECKPOINT,
        "MODEL.DEVICE", "cpu"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("UnSAM Whole-image Segmentation Error:", e)
        print(traceback.format_exc())
        return {"error": "UnSAM Whole-image Segmentation Error"}

    # Return the output image URL
    image_url = f"/static/outputs/{image_id}_out.jpg"
    return {"image_url": image_url}

@app.post("/point_unsam")
async def point_unsam(file: UploadFile = File(...)):
    image_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{image_id}.jpg")
    output_dir = os.path.join(STATIC_OUTPUT_DIR, image_id)
    os.makedirs(output_dir, exist_ok=True)

    # Store the uploaded file
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cmd = [
        "python", "promptable_segmentation/point_segmentation.py",
        "--image_path", input_path,
        "--config_path", "promptable_segmentation/configs/semantic_sam_only_sa-1b_swinT.yaml",
        "--ckpt_path", "checkpoints/unsam_plus_promptable_sa1b_1perc_ckpt_100k.pth",
        "--output_dir", output_dir
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("UnSAM Prompt Segmentation Error:", e)
        print(traceback.format_exc())
        return JSONResponse({"error": "UnSAM Prompt Segmentation Error "}, status_code=500)

    base_url = f"/static/outputs/{image_id}"
    result = {
        "marked_image": f"{base_url}/marked_input.jpg",
        "segmentation_results": [
            f"{base_url}/seg_result_{i}.jpg" for i in range(4)
        ]
    }
    return result