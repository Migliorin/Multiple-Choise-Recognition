from pathlib import Path
import argparse
import math
import random
import yaml
import cv2
import matplotlib.pyplot as plt

from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_data_yaml(data_yaml_path: str) -> dict:
    data_yaml_path = Path(data_yaml_path).resolve()

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("data.yml inválido: o conteúdo precisa ser um dicionário YAML.")

    if "val" not in data:
        raise ValueError("data.yml inválido: campo 'val' não encontrado.")

    return data


def resolve_dataset_path(data_yaml_path: str, value: str | list[str]) -> list[Path]:
    """
    Resolve caminhos do data.yml.

    Suporta:
    - val: images/val
    - val: /caminho/absoluto/images/val
    - val: arquivo.txt com lista de imagens
    - val: [path1, path2]
    - path: /dataset/root + val: images/val
    """
    data_yaml_path = Path(data_yaml_path).resolve()
    data = load_data_yaml(data_yaml_path)

    yaml_dir = data_yaml_path.parent

    base_path = data.get("path", yaml_dir)
    base_path = Path(base_path)

    if not base_path.is_absolute():
        base_path = (yaml_dir / base_path).resolve()

    values = value if isinstance(value, list) else [value]

    resolved = []
    for item in values:
        p = Path(item)

        if not p.is_absolute():
            candidate_from_base = (base_path / p).resolve()
            candidate_from_yaml = (yaml_dir / p).resolve()

            if candidate_from_base.exists():
                p = candidate_from_base
            else:
                p = candidate_from_yaml.resolve()

        resolved.append(p)

    return resolved


def collect_images_from_path(path: Path) -> list[Path]:
    """
    Coleta imagens de:
    - diretório;
    - arquivo .txt com caminhos;
    - arquivo de imagem único.
    """
    images = []

    if path.is_dir():
        images = [
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]

    elif path.is_file() and path.suffix.lower() == ".txt":
        txt_dir = path.parent

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:
            img_path = Path(line)

            if not img_path.is_absolute():
                img_path = (txt_dir / img_path).resolve()

            if img_path.exists() and img_path.suffix.lower() in IMAGE_EXTS:
                images.append(img_path)

    elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        images = [path]

    return sorted(images)


def get_validation_images(data_yaml_path: str) -> list[Path]:
    data = load_data_yaml(data_yaml_path)
    val_paths = resolve_dataset_path(data_yaml_path, data["val"])

    images = []
    for p in val_paths:
        images.extend(collect_images_from_path(p))

    images = sorted(set(images))

    if not images:
        raise FileNotFoundError(
            "Nenhuma imagem de validação encontrada. "
            "Verifique o campo 'val' no data.yml."
        )

    return images


def print_metrics(metrics):
    """
    Métricas principais para detecção.
    Ultralytics geralmente expõe:
    metrics.box.mp     -> mean precision
    metrics.box.mr     -> mean recall
    metrics.box.map50  -> mAP@0.50
    metrics.box.map75  -> mAP@0.75
    metrics.box.map    -> mAP@0.50:0.95
    """

    print("\n========== MÉTRICAS DE VALIDAÇÃO ==========")

    if hasattr(metrics, "box") and metrics.box is not None:
        box = metrics.box

        values = {
            "Precision": getattr(box, "mp", None),
            "Recall": getattr(box, "mr", None),
            "mAP50": getattr(box, "map50", None),
            "mAP75": getattr(box, "map75", None),
            "mAP50-95": getattr(box, "map", None),
        }

        for name, value in values.items():
            if value is not None:
                print(f"{name:12s}: {value:.4f}")

        maps = getattr(box, "maps", None)
        if maps is not None:
            print("\nAP por classe, quando disponível:")
            for i, ap in enumerate(maps):
                print(f"Classe {i:>3}: {ap:.4f}")

    else:
        print(metrics)

    print("==========================================\n")


def make_prediction_grid(
    model: YOLO,
    image_paths: list[Path],
    output_path: str,
    n_images: int = 12,
    conf: float = 0.25,
    imgsz: int = 640,
    seed: int = 42,
    show: bool = True,
):
    random.seed(seed)

    n_images = min(n_images, len(image_paths))
    sampled_images = random.sample(image_paths, n_images)

    results = model.predict(
        source=[str(p) for p in sampled_images],
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )

    plotted_images = []

    for result in results:
        # result.plot() retorna imagem anotada em BGR.
        annotated_bgr = result.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        plotted_images.append(annotated_rgb)

    cols = min(4, n_images)
    rows = math.ceil(n_images / cols)

    plt.figure(figsize=(cols * 5, rows * 5))

    for i, img in enumerate(plotted_images):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.axis("off")

        filename = Path(sampled_images[i]).name
        ax.set_title(filename, fontsize=9)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Grid salvo em: {output_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Validação + inferência em imagens de validação com YOLO."
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Caminho para o data.yml."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Caminho para o modelo YOLO, ex: runs/detect/train/weights/best.pt."
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Tamanho de imagem usado na validação/inferência."
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold para predição."
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold para NMS/validação."
    )

    parser.add_argument(
        "--n",
        type=int,
        default=12,
        help="Número de imagens no grid."
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device: 'cpu', '0', '0,1', etc. Se omitido, Ultralytics decide."
    )

    parser.add_argument(
        "--output",
        default="outputs/valid_predictions_grid.jpg",
        help="Arquivo de saída do grid."
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Não abre a janela do matplotlib, apenas salva o grid."
    )

    args = parser.parse_args()

    model = YOLO(args.model)

    print("Rodando validação...")
    metrics = model.val(
        data=args.data,
        split="val",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )

    print_metrics(metrics)

    print("Coletando imagens de validação...")
    valid_images = get_validation_images(args.data)
    print(f"Total de imagens de validação encontradas: {len(valid_images)}")

    print("Rodando inferência e montando grid...")
    make_prediction_grid(
        model=model,
        image_paths=valid_images,
        output_path=args.output,
        n_images=args.n,
        conf=args.conf,
        imgsz=args.imgsz,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()