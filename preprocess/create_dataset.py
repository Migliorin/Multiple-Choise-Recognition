import os
import shutil
import argparse
import random
from pathlib import Path


EXTENSOES_IMAGEM = [
    ".jpg", ".jpeg", ".png",
    ".JPG", ".JPEG", ".PNG"
]


def listar_txts(labels_dir):
    labels_dir = Path(labels_dir)

    txt_files = []

    for root, _, files in os.walk(labels_dir):
        for file in files:
            if file.lower().endswith(".txt"):
                txt_files.append(Path(root) / file)

    return txt_files


def encontrar_imagem_correspondente(txt_file, labels_dir):
    """
    Procura a imagem correspondente ao .txt usando a mesma estrutura relativa.

    Exemplo:
        labels_dir/School001/Grade12/File0005.txt
        labels_dir/School001/Grade12/File0005.jpg
    """

    labels_dir = Path(labels_dir)

    relative_path = txt_file.relative_to(labels_dir)
    relative_without_ext = relative_path.with_suffix("")

    for ext in EXTENSOES_IMAGEM:
        rotated_image_path = labels_dir / relative_without_ext.with_suffix(ext)

        if rotated_image_path.exists():
            return rotated_image_path

    return None


def nome_flatten_arquivo(file_path, base_dir, nova_extensao=None):
    """
    Transforma:
        School001/Grade12/File0005.txt

    Em:
        School001_Grade12_File0005.txt
    """

    file_path = Path(file_path)
    base_dir = Path(base_dir)

    relative_path = file_path.relative_to(base_dir)
    partes = list(relative_path.parts)

    nome = "_".join(partes)

    if nova_extensao is not None:
        nome = str(Path(nome).with_suffix(nova_extensao))

    return nome


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


def copiar_par(txt_file, image_file, labels_dir, dataset_dir, split):
    dataset_dir = Path(dataset_dir)

    nome_base_txt = nome_flatten_arquivo(
        file_path=txt_file,
        base_dir=labels_dir,
        nova_extensao=".txt"
    )

    extensao_imagem = image_file.suffix

    nome_base_img = nome_flatten_arquivo(
        file_path=image_file,
        base_dir=labels_dir,
        nova_extensao=extensao_imagem
    )

    destino_txt = dataset_dir / "labels" / split / nome_base_txt
    destino_img = dataset_dir / "images" / split / nome_base_img

    shutil.copy2(txt_file, destino_txt)
    shutil.copy2(image_file, destino_img)


def salvar_data_yaml(dataset_dir, class_names):
    dataset_dir = Path(dataset_dir)
    data_yaml = dataset_dir / "data.yaml"

    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {dataset_dir.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/valid\n")
        f.write("\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write("names:\n")

        for idx, class_name in enumerate(class_names):
            f.write(f"  {idx}: {class_name}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Cria um dataset YOLO sem subpastas, renomeando labels e imagens."
    )

    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Diretório contendo os arquivos .txt. Exemplo: /caminho/output"
    )

    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Diretório de saída no formato YOLO. Padrão: dataset"
    )

    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.2,
        help="Proporção dos dados para validação. Padrão: 0.2"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para embaralhamento. Padrão: 42"
    )

    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["object"],
        help="Nomes das classes. Exemplo: --class-names classe0 classe1 classe2 classe3"
    )

    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    dataset_dir = Path(args.dataset_dir)

    if not labels_dir.exists():
        raise FileNotFoundError(f"labels-dir não existe: {labels_dir}")

    if not 0 < args.valid_size < 1:
        raise ValueError("--valid-size precisa estar entre 0 e 1.")

    criar_estrutura_yolo(dataset_dir)

    txt_files = listar_txts(labels_dir)

    if not txt_files:
        print("Nenhum arquivo .txt encontrado.")
        return

    pares_validos = []
    arquivos_sem_imagem = []

    for txt_file in txt_files:
        image_file = encontrar_imagem_correspondente(
            txt_file=txt_file,
            labels_dir=labels_dir
        )

        if image_file is None:
            arquivos_sem_imagem.append(txt_file)
        else:
            pares_validos.append((txt_file, image_file))

    if not pares_validos:
        print("Nenhum par válido de .txt + imagem encontrado.")
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
            labels_dir=labels_dir,
            dataset_dir=dataset_dir,
            split="train"
        )

    for txt_file, image_file in valid_files:
        copiar_par(
            txt_file=txt_file,
            image_file=image_file,
            labels_dir=labels_dir,
            dataset_dir=dataset_dir,
            split="valid"
        )

    salvar_data_yaml(
        dataset_dir=dataset_dir,
        class_names=args.class_names
    )

    if arquivos_sem_imagem:
        debug_file = dataset_dir / "arquivos_sem_imagem.txt"

        with open(debug_file, "w", encoding="utf-8") as f:
            for txt_file in arquivos_sem_imagem:
                f.write(str(txt_file) + "\n")

        print(f"{len(arquivos_sem_imagem)} arquivos .txt ficaram sem imagem correspondente.")
        print(f"Lista salva em: {debug_file}")

    print("Dataset YOLO criado com sucesso.")
    print(f"Total de arquivos .txt encontrados: {len(txt_files)}")
    print(f"Total usado no dataset: {total}")
    print(f"Train: {len(train_files)}")
    print(f"Valid: {len(valid_files)}")
    print(f"Saída: {dataset_dir.resolve()}")


if __name__ == "__main__":
    main()
