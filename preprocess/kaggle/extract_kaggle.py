import argparse
import os

import cv2
import pandas as pd
from tqdm import tqdm

from kaggle import Kaggle


def salvar_anotacoes(output_file, anotacoes):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file_:
        for item in anotacoes:
            file_.write(f"{item}\n")


def salvar_imagem(output_file, imagem):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not cv2.imwrite(output_file, imagem):
        raise IOError(f"Falha ao salvar imagem em: {output_file}")


def nome_base_unico(image_id):
    return f"kaggle_{int(image_id):06d}"


def carregar_registros(csv_file):
    dataframe = pd.read_csv(csv_file)
    dataframe = dataframe.dropna()

    registros = []

    for row in dataframe.values:
        image_id = int(row[0])
        labels = [str(label).strip().upper() for label in row[1:]]

        if "BLANK" in labels or "MULTI" in labels:
            continue

        if len(labels) != 30:
            continue

        registros.append((image_id, labels))

    return registros


def processar_registros(registros, input_dir, output_dir, list_debug, pbar):
    kaggle = Kaggle()

    for image_id, labels in registros:
        image_file = os.path.join(input_dir, f"{image_id}.jpg")
        base_name = nome_base_unico(image_id)
        output_image_file = os.path.join(output_dir, f"{base_name}.jpg")
        output_txt_file = os.path.join(output_dir, f"{base_name}.txt")

        try:
            if os.path.exists(output_image_file) and os.path.exists(output_txt_file):
                continue

            anotacoes, imagem_rotacionada = kaggle(labels, image_file)

            salvar_imagem(output_image_file, imagem_rotacionada)
            salvar_anotacoes(output_txt_file, anotacoes)

        except Exception as error:
            list_debug.append((image_file, str(error)))

        finally:
            pbar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Gera imagens e anotacoes YOLO do dataset Kaggle."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Diretorio com as imagens do Kaggle. Exemplo: OMR_Image_Resized"
    )

    parser.add_argument(
        "--csv-file",
        required=True,
        help="Arquivo CSV com as respostas. Exemplo: OMR_Ans.csv"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Diretorio de saida para imagens e anotacoes."
    )

    parser.add_argument(
        "--debug-file",
        default="debug_erros.txt",
        help="Arquivo para salvar erros. Padrao: debug_erros.txt."
    )

    args = parser.parse_args()

    registros = carregar_registros(args.csv_file)

    if not registros:
        print("Nenhum registro valido encontrado.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    list_debug = []

    barra_progresso = tqdm(
        total=len(registros),
        desc="Extraindo coordenadas",
        unit="imagem"
    )

    try:
        processar_registros(
            registros=registros,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            list_debug=list_debug,
            pbar=barra_progresso
        )

    finally:
        barra_progresso.close()

    if list_debug:
        debug_path = os.path.join(args.output_dir, args.debug_file)

        with open(debug_path, "w", encoding="utf-8") as file_:
            for image_file, error in list_debug:
                file_.write(f"{image_file}\t{error}\n")

        print(f"Processamento concluido com erros. Debug salvo em: {debug_path}")
    else:
        print("Processamento concluido sem erros.")


if __name__ == "__main__":
    main()
