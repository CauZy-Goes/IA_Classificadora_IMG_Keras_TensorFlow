import matplotlib.pyplot as plt  # Biblioteca para criação de gráficos
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay  # Métricas e exibição da matriz de confusão

def avaliar_modelo(model, test_gen):
    """
    Avalia o modelo no conjunto de teste e exibe métricas de desempenho, como perda, acurácia,
    relatório de classificação e matriz de confusão.

    Parâmetros:
    - model: Modelo treinado a ser avaliado.
    - test_gen: Gerador de dados do conjunto de teste.
    """
    # Avaliação no conjunto de teste:
    # - Calcula a perda (loss) e a acurácia (accuracy) no conjunto de teste.
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Teste - Loss: {test_loss}, Accuracy: {test_acc}")  # Exibe os resultados

    # Predição:
    # - Realiza predições no conjunto de teste e converte as probabilidades em classes (0 ou 1).
    y_pred = (model.predict(test_gen) > 0.5).astype("int32")  # Classifica com base em um limiar de 0.5
    y_true = test_gen.classes  # Obtém os rótulos verdadeiros do conjunto de teste

    # Relatório de Classificação:
    # - Gera métricas como precisão (precision), recall, F1-score e suporte para cada classe.
    # - 'target_names' atribui nomes às classes.
    print(classification_report(y_true, y_pred, target_names=['Não é Pulmão', 'Pulmão']))

    # Matriz de Confusão:
    # - Mostra a relação entre as predições do modelo e os valores reais.
    cm = confusion_matrix(y_true, y_pred)  # Calcula a matriz de confusão
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não é Pulmão', 'Pulmão'])
    disp.plot(cmap=plt.cm.Blues)  # Exibe a matriz com um mapa de cor azul para melhor visualização
    plt.show()  # Mostra a matriz de confusão plotada

def plotar_historico(history):
    """
    Plota as curvas de perda e acurácia para os conjuntos de treino e validação ao longo das épocas.

    Parâmetros:
    - history: Histórico de treinamento gerado pelo método model.fit().
    """
    # Configura o tamanho do gráfico
    plt.figure(figsize=(12, 5))

    # Subgráfico 1: Acurácia
    plt.subplot(1, 2, 1)  # Configura o primeiro gráfico em uma grade 1x2
    plt.plot(history.history['accuracy'], label='Train Accuracy')  # Acurácia no conjunto de treino
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Acurácia no conjunto de validação
    plt.legend()  # Exibe a legenda
    plt.title('Accuracy over epochs')  # Define o título do gráfico

    # Subgráfico 2: Perda
    plt.subplot(1, 2, 2)  # Configura o segundo gráfico em uma grade 1x2
    plt.plot(history.history['loss'], label='Train Loss')  # Perda no conjunto de treino
    plt.plot(history.history['val_loss'], label='Validation Loss')  # Perda no conjunto de validação
    plt.legend()  # Exibe a legenda
    plt.title('Loss over epochs')  # Define o título do gráfico

    # Exibe os gráficos gerados
    plt.show()
