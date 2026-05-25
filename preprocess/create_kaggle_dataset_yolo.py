import argparse
import os
import random
import shutil
from pathlib import Path


EXTENSOES_IMAGEM = [
    ".jpg", ".jpeg", ".png",
    ".JPG", ".JPEG", ".PNG"
]


def listar_txts(input_dir):
    input_dir = Path(input_dir)

    txt_files = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                txt_files.append(Path(root) / file)

    return txt_files


def encontrar_imagem_correspondente(txt_file):
    base_path = txt_file.with_suffix("")

    for ext in EXTENSOES_IMAGEM:
        image_file = base_path.with_suffix(ext)

        if image_file.exists():
            return image_file

    return None


def criar_estrutura_yolo(dataset_dir):
    dataset_dir = Path(dataset_dir)

    dirs = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "valid",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "valid",
    ]

    for dir_ in dirs:
        dir_.mkdir(parents=True, exist_ok=True)


def copiar_par(txt_file, image_file, dataset_dir, split):
    dataset_dir = Path(dataset_dir)

    destino_txt = dataset_dir / "labels" / split / txt_file.name
    destino_img = dataset_dir / "images" / split / image_file.name

    shutil.copy2(txt_file, destino_txt)
    shutil.copy2(image_file, destino_img)


def salvar_data_yaml(dataset_dir, class_names):
    dataset_dir = Path(dataset_dir)
    data_yaml = dataset_dir / "data.yaml"

    with open(data_yaml, "w", encoding="utf-8") as file_:
        file_.write(f"path: {dataset_dir.resolve()}\n")
        file_.write("train: images/train\n")
        file_.write("val: images/valid\n")
        file_.write("\n")
        file_.write(f"nc: {len(class_names)}\n")
        file_.write("names:\n")

        for idx, class_name in enumerate(class_names):
            file_.write(f"  {idx}: {class_name}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Cria um dataset YOLO para o Kaggle a partir de imagens e anotacoes ja formatadas."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Diretorio contendo imagens e arquivos .txt no formato YOLO."
    )

    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Diretorio de saida no formato YOLO. Padrao: dataset"
    )

    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.2,
        help="Proporcao dos dados para validacao. Padrao: 0.2"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para embaralhamento. Padrao: 42"
    )

    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["A", "B", "C", "D"],
        help="Nomes das classes. Exemplo: --class-names A B C D"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    dataset_dir = Path(args.dataset_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"input-dir nao existe: {input_dir}")

    if not 0 < args.valid_size < 1:
        raise ValueError("--valid-size precisa estar entre 0 e 1.")

    criar_estrutura_yolo(dataset_dir)

    txt_files = listar_txts(input_dir)

    if not txt_files:
        print("Nenhum arquivo .txt encontrado.")
        return

    pares_validos = []
    arquivos_sem_imagem = []

    for txt_file in txt_files:
        image_file = encontrar_imagem_correspondente(txt_file)

        if image_file is None:
            arquivos_sem_imagem.append(txt_file)
        else:
            pares_validos.append((txt_file, image_file))

    if not pares_validos:
        print("Nenhum par valido de .txt + imagem encontrado.")
        return

    random.seed(args.seed)
    random.shuffle(pares_validos)

    total = len(pares_validos)
    total_valid = int(total * args.valid_size)

    valid_files = pares_validos[:total_valid]
    train_files = pares_validos[total_valid:]

    for txt_file, image_file in train_files:
        copiar_par(
            txt_file=txt_file,
            image_file=image_file,
            dataset_dir=dataset_dir,
            split="train"
        )

    for txt_file, image_file in valid_files:
        copiar_par(
            txt_file=txt_file,
            image_file=image_file,
            dataset_dir=dataset_dir,
            split="valid"
        )

    salvar_data_yaml(
        dataset_dir=dataset_dir,
        class_names=args.class_names
    )

    if arquivos_sem_imagem:
        debug_file = dataset_dir / "arquivos_sem_imagem.txt"

        with open(debug_file, "w", encoding="utf-8") as file_:
            for txt_file in arquivos_sem_imagem:
                file_.write(str(txt_file) + "\n")

        print(f"{len(arquivos_sem_imagem)} arquivos .txt ficaram sem imagem correspondente.")
        print(f"Lista salva em: {debug_file}")

    print("Dataset YOLO criado com sucesso.")
    print(f"Total de arquivos .txt encontrados: {len(txt_files)}")
    print(f"Total usado no dataset: {total}")
    print(f"Train: {len(train_files)}")
    print(f"Valid: {len(valid_files)}")
    print(f"Saida: {dataset_dir.resolve()}")


if __name__ == "__main__":
    main()
