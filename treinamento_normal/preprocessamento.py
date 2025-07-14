from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Importa a classe para gerar e pré-processar imagens

def carregar_dados(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    """
    Carrega e pré-processa os dados de treinamento, validação e teste.

    Parâmetros:
    - train_dir: Caminho para a pasta de treinamento.
    - val_dir: Caminho para a pasta de validação.
    - test_dir: Caminho para a pasta de teste.
    - img_size: Dimensão das imagens a serem redimensionadas (largura, altura). Default é (224, 224).
    - batch_size: Número de amostras a serem processadas por lote. Default é 32.

    Retorna:
    - train_gen: Gerador de dados de treinamento.
    - val_gen: Gerador de dados de validação.
    - test_gen: Gerador de dados de teste.
    """
    # Inicializa o gerador de imagens para o conjunto de treinamento com aumentação de dados
    # O rescale normaliza os pixels para o intervalo [0, 1], enquanto os outros parâmetros aplicam aumentações como:
    # - Rotação de até 15 graus
    # - Deslocamento horizontal e vertical de até 10% da imagem
    # - Zoom de até 10%
    # - Flip horizontal
    # - Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,  # Normalização
        rotation_range=15,  # Rotação aleatória
        width_shift_range=0.1,  # Deslocamento horizontal
        height_shift_range=0.1,  # Deslocamento vertical
        zoom_range=0.1,  # Zoom aleatório
        horizontal_flip=True  # Flip horizontal
    )

    # Inicializa o gerador de imagens para os conjuntos de validação e teste, sem aumentação de dados
    # Apenas normaliza os pixels no intervalo [0, 1], dividindo por 255
    # - Sem Data Augmentation pois queremos medir o desempenho real do modelo em dados que ele nunca viu.
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Cria o gerador de dados de treinamento
    train_gen = train_datagen.flow_from_directory(
        train_dir,  # Diretório contendo as imagens de treinamento organizadas em subpastas por classe
        target_size=img_size,  # Redimensiona as imagens para o tamanho especificado (224x224 por padrão)
        batch_size=batch_size,  # Número de amostras por lote
        class_mode='binary'  # Classificação binária: NORMAL ou PNEUMONIA
    )

    # Cria o gerador de dados de validação
    val_gen = val_test_datagen.flow_from_directory(
        val_dir,  # Diretório contendo as imagens de validação organizadas em subpastas por classe
        target_size=img_size,  # Redimensiona as imagens para o tamanho especificado
        batch_size=batch_size,  # Número de amostras por lote
        class_mode='binary'  # Classificação binária
    )

    # Cria o gerador de dados de teste
    test_gen = val_test_datagen.flow_from_directory(
        test_dir,  # Diretório contendo as imagens de teste organizadas em subpastas por classe
        target_size=img_size,  # Redimensiona as imagens para o tamanho especificado
        batch_size=batch_size,  # Número de amostras por lote
        class_mode='binary',  # Classificação binária
        shuffle=False  # Não embaralha os dados, útil para avaliação e geração de previsões consistentes
    )

    # Retorna os geradores de dados
    return train_gen, val_gen, test_gen
