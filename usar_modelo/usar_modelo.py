import os
import pneumo_finder as pf
# from tensorflow.keras.models import load_model

# Define o caminho do modelo com base na localização real do script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Pega o caminho da pasta 'usar_modelo'
CAMINHO_MODELO = os.path.join(BASE_DIR, "..", "best_model.keras")

# # Carrega o modelo com segurança
# modelo = load_model("best_model.keras")
# modelo.summary()

detector = pf.DetectorDePneumonia(CAMINHO_MODELO)

# detector.diagnosticar_imagem(os.path.join(BASE_DIR, "imgs", "1_normal1.jpeg"))
# detector.diagnosticar_imagem(os.path.join(BASE_DIR, "imgs", "3_pneumonia1.jpeg"))
# detector.diagnosticar_imagem(os.path.join(BASE_DIR, "imgs", "2_normal2.jpeg"))
detector.diagnosticar_imagem(os.path.join(BASE_DIR, "imgs", "4_pneumonia2.jpeg"))
# detector.diagnosticar_imagem(os.path.join(BASE_DIR, "imgs", "6_normal3.jpeg"))

# detector.diagnosticar_pasta(os.path.join(BASE_DIR, "imgs"))
