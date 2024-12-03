from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def treinar_modelo(model, train_gen, val_gen, epochs=20):
    """Treina o modelo e salva o melhor desempenho."""
    callbacks = [
        ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )

    return history
