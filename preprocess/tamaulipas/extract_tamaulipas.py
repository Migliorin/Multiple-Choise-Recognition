import os
import argparse
import threading
import numpy as np
from tqdm import tqdm

from preprocess.tamaulipas.tamaulipas import Tamaulipas


def listar_imagens(input_dir):
    image_files = []

    for root_, _, files_ in os.walk(input_dir):
        for file_ in files_:
            if file_.lower().endswith((".png", ".jpg", ".jpeg")):
                image_files.append(os.path.join(root_, file_))

    return image_files


def caminho_saida_txt(image_file, input_dir, output_dir):
    """
    Cria um caminho de saída baseado no nome/caminho relativo da imagem.

    Exemplo:
        input_dir/images/a/b/img001.jpg
        output_dir/a/b/img001.txt
    """

    relative_path = os.path.relpath(image_file, input_dir)
    relative_path_without_ext = os.path.splitext(relative_path)[0]

    output_file = os.path.join(output_dir, relative_path_without_ext + ".txt")
    return output_file


def salvar_coordenadas(output_file, coordenadas):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        if isinstance(coordenadas, (list, tuple)):
            for item in coordenadas:
                f.write(f"{item}\n")
        else:
            f.write(str(coordenadas))


def extract_function(image_files, input_dir, output_dir, list_debug, pbar):
    tamaulipas = Tamaulipas()

    for image_file in image_files:
        output_file = caminho_saida_txt(
            image_file=image_file,
            input_dir=input_dir,
            output_dir=output_dir
        )

        try:
            # Se já existe o .txt, considera que a coordenada já foi extraída
            if os.path.exists(output_file):
                continue

            coordenadas = tamaulipas(image_file)

            if coordenadas is None:
                continue 
            
            salvar_coordenadas(
                output_file=output_file,
                coordenadas=coordenadas
            )

        except Exception as error:
            list_debug.append((image_file, str(error)))

        finally:
            pbar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Extrai coordenadas de imagens usando Tamaulipas e salva os resultados em arquivos .txt."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Diretório raiz contendo as imagens."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Diretório onde os arquivos .txt serão salvos."
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Número de threads para processamento. Padrão: 8."
    )

    parser.add_argument(
        "--debug-file",
        default="debug_erros.txt",
        help="Arquivo para salvar imagens que deram erro. Padrão: debug_erros.txt."
    )

    args = parser.parse_args()

    image_files = listar_imagens(args.input_dir)

    if not image_files:
        print("Nenhuma imagem encontrada.")
        return

    list_debug = []
    threads_list = []

    num_threads = min(args.num_threads, len(image_files))

    blocos = np.array_split(image_files, num_threads)

    descricao_progresso = "Extraindo coordenadas"
    barra_progresso = tqdm(
        total=len(image_files),
        desc=descricao_progresso,
        unit="imagem"
    )
    os.makedirs(args.output_dir,exist_ok=True)

    try:
        for bloco in blocos:
            thread = threading.Thread(
                target=extract_function,
                args=(
                    list(bloco),
                    args.input_dir,
                    args.output_dir,
                    list_debug,
                    barra_progresso
                )
            )

            thread.start()
            threads_list.append(thread)

        for thread in threads_list:
            thread.join()

    finally:
        barra_progresso.close()

    if list_debug:
        debug_path = os.path.join(args.output_dir, args.debug_file)
        os.makedirs(args.output_dir, exist_ok=True)

        with open(debug_path, "w", encoding="utf-8") as f:
            for image_file, error in list_debug:
                f.write(f"{image_file}\t{error}\n")

        print(f"Processamento concluído com erros. Debug salvo em: {debug_path}")
    else:
        print("Processamento concluído sem erros.")


if __name__ == "__main__":
    main()