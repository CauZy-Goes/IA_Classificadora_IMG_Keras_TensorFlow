from preprocessamento import carregar_dados  # Importa a função para carregar e pré-processar os dados
from modelo import criar_modelo  # Importa a função para criar a arquitetura da Rede Neural Convolucional (CNN)
from treinamento import treinar_modelo  # Importa a função responsável por treinar o modelo
from avaliacao import avaliar_modelo, plotar_historico  # Importa as funções para avaliar o modelo e gerar gráficos de desempenho


# Caminhos do dataset dividido
# Define os caminhos das pastas onde estão os dados de treinamento, validação e teste
train_dir = "C:\\Users\\cauac\\dev\\Faculdade\\IA\\IA_Classificadora_IMG_Keras_TensorFlow\\dataset\\train"
val_dir = "C:\\Users\\cauac\\dev\\Faculdade\\IA\\IA_Classificadora_IMG_Keras_TensorFlow\\dataset\\val"
test_dir = "C:\\Users\\cauac\\dev\\Faculdade\\IA\\IA_Classificadora_IMG_Keras_TensorFlow\\dataset\\test"

# Passo 1: Carregar os dados
# Chama a função para carregar e pré-processar os dados das pastas de treinamento, validação e teste.
# Essa função deve realizar operações como redimensionamento das imagens, normalização e divisão em batches.
train_gen, val_gen, test_gen = carregar_dados(train_dir, val_dir, test_dir)

# Passo 2: Criar o modelo
# Cria a arquitetura do modelo CNN utilizando a função `criar_modelo`, que deve incluir camadas convolucionais,
# pooling, densas (fully connected), e uma camada de saída para classificação binária.
model = criar_modelo()

# Passo 3: Treinar o modelo
# Treina o modelo utilizando os dados de treinamento e validação.
# O treinamento é acompanhado por callbacks como salvamento do melhor modelo e early stopping.
history = treinar_modelo(model, train_gen, val_gen)

# Passo 4: Avaliar o modelo
# Avalia o desempenho do modelo nos dados de teste.
# A função deve calcular métricas como acurácia, precisão, recall e F1-score, além de gerar uma matriz de confusão.
avaliar_modelo(model, test_gen)

# Passo 5: Plotar os gráficos de histórico
# Plota os gráficos de acurácia e perda ao longo das épocas de treinamento e validação, utilizando o histórico gerado.
# Isso ajuda a analisar o desempenho do modelo e identificar possíveis problemas como overfitting.
plotar_historico(history)
