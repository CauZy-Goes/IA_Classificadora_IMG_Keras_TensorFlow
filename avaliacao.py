import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def avaliar_modelo(model, test_gen):
    """Avalia o modelo e exibe métricas de desempenho."""
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Teste - Loss: {test_loss}, Accuracy: {test_acc}")

    # Relatório de Classificação
    y_pred = (model.predict(test_gen) > 0.5).astype("int32")
    y_true = test_gen.classes
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NORMAL', 'PNEUMONIA'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def plotar_historico(history):
    """Plota as curvas de perda e acurácia."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss over epochs')
    plt.show()
