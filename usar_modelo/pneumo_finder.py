import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class DetectorDePneumonia:
    def __init__(self, caminho_modelo, tamanho_img=(224, 224)):
        """
        Inicializa o detector de pneumonia carregando o modelo treinado.

        Parâmetros:
        - caminho_modelo: Caminho para o arquivo do modelo salvo (.keras).
        - tamanho_img: Tamanho esperado das imagens (default = (224, 224)).
        """
        self.modelo = load_model(caminho_modelo)
        self.tamanho_img = tamanho_img

    def _preprocessar_imagem(self, caminho_imagem):
        """
        Pré-processa a imagem para o formato esperado pela rede neural.

        Parâmetros:
        - caminho_imagem: Caminho para a imagem a ser processada.

        Retorna:
        - imagem preparada para predição (tensor 4D).
        """
        img = image.load_img(caminho_imagem, target_size=self.tamanho_img)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normaliza
        return np.expand_dims(img_array, axis=0)  # Adiciona dimensão para o batch

    def diagnosticar_imagem(self, caminho_imagem):
        """
        Diagnostica uma única imagem, indicando se é NORMAL ou PNEUMONIA.

        Parâmetros:
        - caminho_imagem: Caminho da imagem a ser diagnosticada.

        Retorna:
        - Uma tupla (classe_predita, confianca) indicando o diagnóstico.
        """
        imagem_processada = self._preprocessar_imagem(caminho_imagem)
        pred = self.modelo.predict(imagem_processada)[0][0]
        classe = "PNEUMONIA" if pred > 0.5 else "NORMAL"
        confianca = pred if pred > 0.5 else 1 - pred
        print(f"{os.path.basename(caminho_imagem)}: {classe} (confiança: {confianca:.2f})")
        return classe, confianca

    def diagnosticar_pasta(self, pasta_imgs):
        """
        Diagnostica todas as imagens dentro de uma pasta, indicando se é NORMAL ou PNEUMONIA.

        Parâmetros:
        - pasta_imgs: Caminho da pasta com as imagens a serem diagnosticadas.
        """
        for nome_arquivo in os.listdir(pasta_imgs):
            caminho = os.path.join(pasta_imgs, nome_arquivo)
            if os.path.isfile(caminho):
                imagem_processada = self._preprocessar_imagem(caminho)
                pred = self.modelo.predict(imagem_processada)[0][0]
                classe = "PNEUMONIA" if pred > 0.5 else "NORMAL"
                confianca = pred if pred > 0.5 else 1 - pred
                print(f"{nome_arquivo}: {classe} (confiança: {confianca:.2f})")
