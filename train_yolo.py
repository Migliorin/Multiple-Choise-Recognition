from pathlib import Path
from ultralytics import YOLO

DATA_YAML = "/Users/lucas/Documents/Projetos/Multiple-Choise-Recognition/preprocess/dataset_tamaulipas_base/data.yaml"

MODEL = "yolo26n.pt"

EPOCHS = 32
IMG_SIZE = 640
BATCH_SIZE = 2
WORKERS = 4

DEVICE = "mps"

PROJECT = "runs_yolo"
NAME = "tamaulipas_yolo11n"


data_path = Path(DATA_YAML)

if not data_path.exists():
    raise FileNotFoundError(f"data.yaml não encontrado: {data_path}")

model = YOLO(MODEL)

model.train(
    data=str(data_path),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    project=PROJECT,
    name=NAME,
    workers=WORKERS,
    cache=False,
    patience=30,
    pretrained=True,
    optimizer="auto",
    verbose=True
)