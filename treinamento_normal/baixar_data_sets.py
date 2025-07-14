import tensorflow_datasets as tfds
import tensorflow as tf
import os
from PIL import Image
import numpy as np

# CONFIGURAÇÕES
QUANTIDADE = 1400               #1400       # Quantas imagens salvar
DESTINO = "dataset/train/NAO_PULMAO"  # Pasta de destino
START_INDEX = 1400                  # A partir de qual imagem do CIFAR começar

os.makedirs(DESTINO, exist_ok=True)

# Carrega o CIFAR-10
ds = tfds.load('cifar10', split='train', shuffle_files=False)  # NÃO embaralha

# Converte para numpy e salva a partir do índice desejado
contador = 0
indice_atual = 0

for exemplo in tfds.as_numpy(ds):
    if indice_atual < START_INDEX:
        indice_atual += 1
        continue  # pula até o índice inicial

    if contador >= QUANTIDADE:
        break

    imagem = exemplo['image']
    img = Image.fromarray(imagem)
    img = img.resize((224, 224))
    img.save(f"{DESTINO}/nao_pulmao_{START_INDEX + contador}.jpg")
    contador += 1
    indice_atual += 1

print(f"Salvo {contador} imagens a partir de índice {START_INDEX} em: {DESTINO}")
