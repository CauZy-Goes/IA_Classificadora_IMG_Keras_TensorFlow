from preprocessamento import carregar_dados
from modelo import criar_modelo
from treinamento import treinar_modelo
from avaliacao import avaliar_modelo, plotar_historico

# Caminhos do dataset
train_dir = "C:\\Users\cauac\\dev\\Faculdade\\IA\\IA_Classificadora_IMG_Keras_TensorFlow\\dataset\\train"
val_dir = "C:\\Users\cauac\\dev\\Faculdade\\IA\\IA_Classificadora_IMG_Keras_TensorFlow\\dataset\\val"
test_dir = "C:\\Users\cauac\\dev\\Faculdade\\IA\\IA_Classificadora_IMG_Keras_TensorFlow\\dataset\\test"

# Passo 1: Carregar os dados
train_gen, val_gen, test_gen = carregar_dados(train_dir, val_dir, test_dir)

# Passo 2: Criar o modelo
model = criar_modelo()

# Passo 3: Treinar o modelo
history = treinar_modelo(model, train_gen, val_gen)

# Passo 4: Avaliar o modelo
avaliar_modelo(model, test_gen)

# Passo 5: Plotar os gráficos de histórico
plotar_historico(history)
