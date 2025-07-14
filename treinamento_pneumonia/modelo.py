from tensorflow.keras.models import Sequential  # Importa o modelo sequencial (camadas empilhadas linearmente)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Importa as camadas necessárias

def criar_modelo(input_shape=(224, 224, 3)):
    """
    Cria e compila um modelo de rede neural convolucional (CNN) para classificação binária.

    Parâmetros:
    - input_shape: Tupla que define o formato das entradas do modelo (altura, largura, canais de cor).
                   O default é (224, 224, 3), indicando imagens coloridas de 224x224.

    Retorna:
    - model: Um modelo CNN compilado e pronto para treinamento.
    """
    model = Sequential([  # Define um modelo sequencial, onde as camadas são empilhadas uma após a outra
        # Primeira camada convolucional:
        # - 32 filtros = é o suficiente para aprender caracteristicas simples como bordas e texturas
        # - tamanho de filtro 3 x 3 = capturam detalhes mais finos e são muito usados em redes modernas
        # - Função de ativação ReLU (para introduzir não-linearidade) permitindo que ele aprenda padrões complexos
        # - input_shape define o tamanho das imagens de entrada
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        # Camada de pooling (redução de dimensionalidade):
        # - Reduz as dimensões da imagem pela metade, usando uma janela de 2x2 preservando as características mais importantes
        # - ajuda a reduzir a complexidade computacional e a prevenir overfitting.
        MaxPooling2D((2, 2)),

        # Segunda camada convolucional:
        # - 64 filtros de tamanho 3x3 = padrões mais complexos, como formas mais detalhadas e partes dos objetos.
        # - Função de ativação ReLU, continua a modelar não-linearidades e acelera o aprendizado
        Conv2D(64, (3, 3), activation='relu'),
        # Segunda camada de pooling:
        MaxPooling2D((2, 2)),

        # Terceira camada convolucional:
        # - 128 filtros de tamanho 3x3 capturar padrões ainda mais complexos
        # - Função de ativação ReLU
        Conv2D(128, (3, 3), activation='relu'),
        # Terceira camada de pooling:
        MaxPooling2D((2, 2)),

        # Camada Flatten:
        # - Achata a saída das camadas anteriores (matrizes) em um vetor unidimensional
        # - conectá-lo à camada dense layer, que espera uma entrada de vetor linear.
        Flatten(),

        # Camada totalmente conectada (Fully Connected Layer):
        # - 128 neurônios aprender as relações complexas entre as características extraídas pelas camadas convolucionais
        #  e de pooling.
        # - Função de ativação ReLU
        Dense(128, activation='relu'),

        # Dropout:
        # - Desativa 50% dos neurônios da camada anterior durante o treinamento (para reduzir overfitting)
        # - Isso ajuda a evitar que o modelo se ajuste demais aos dados de treinamento (overfitting) e melhora a
        #  generalização para dados novos.
        Dropout(0.5),

        # Camada de saída:
        # - 1 neurônio (classificação binária: 0 ou 1)
        # - Função de ativação Sigmoid, que mapeia os valores para o intervalo [0, 1]
        Dense(1, activation='sigmoid')
    ])

    # Compila o modelo:
    # - Otimizador: Adam ajuda dinamicamente (ajusta os pesos para minimizar a função de perda)
    # - Função de perda: Binary Crossentropy (adequada para classificação binária)
    # - Métrica: Accuracy (taxa de acerto) avalia o desempenho do modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model  # Retorna o modelo criado
