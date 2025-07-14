from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping  # Importa os callbacks para monitoramento do treinamento

def treinar_modelo(model, train_gen, val_gen, epochs=15):
    """
    Treina o modelo e implementa estratégias para salvar o melhor modelo e prevenir overfitting.

    Parâmetros:
    - model: O modelo compilado que será treinado.
    - train_gen: Gerador de dados para o conjunto de treino.
    - val_gen: Gerador de dados para o conjunto de validação.
    - epochs: Número máximo de épocas para o treinamento (default = 20).

    Retorna:
    - history: Histórico do treinamento, incluindo métricas como perda e acurácia para treino e validação.
    """
    # Lista de callbacks para monitorar o treinamento:
    callbacks = [
        # ModelCheckpoint:
        # - Salva o modelo apenas quando sua acurácia na validação (val_accuracy) atinge um novo melhor valor.
        # - O arquivo 'best_model.keras' conterá o melhor modelo encontrado durante o treinamento.
        # = mode = max, novo valor maximo
        ModelCheckpoint('normal_model.keras', save_best_only=True, monitor='val_accuracy', mode='max'),

        # EarlyStopping:
        # - Interrompe o treinamento antecipadamente se o desempenho no conjunto de validação (val_loss) parar de melhorar.
        # - O parâmetro 'patience=5' significa que o treinamento será interrompido após 5 épocas sem melhora no val_loss.
        # - 'restore_best_weights=True' garante que o modelo final usará os pesos da melhor época.
        EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True)
    ]

    # Treinamento do modelo:
    # - train_gen fornece os dados de treino em mini-lotes.
    # - val_gen fornece os dados de validação em mini-lotes.
    # - epochs define o número máximo de épocas para o treinamento.
    # - callbacks monitora e salva o progresso do treinamento.
    history = model.fit(
        train_gen,          # Dados de treino
        epochs=epochs,      # Número máximo de épocas
        validation_data=val_gen,  # Dados de validação
        callbacks=callbacks  # Lista de callbacks para monitoramento
    )

    return history  # Retorna o histórico do treinamento
