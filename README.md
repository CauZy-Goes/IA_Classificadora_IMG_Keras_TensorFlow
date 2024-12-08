# IA Classificadora de Imagens de Pulmões com Keras e TensorFlow

Este projeto utiliza técnicas de aprendizado de máquina e redes neurais convolucionais (CNNs) para classificar imagens de pulmões. O modelo é treinado para identificar se o pulmão está saudável ou com pneumonia a partir de imagens de raio-X.

## Objetivo

O objetivo deste projeto é treinar um modelo de rede neural convolucional (CNN) utilizando o **TensorFlow** e **Keras** para classificar imagens de pulmões em duas categorias:

- **Pulmão saudável**
- **Pulmão com pneumonia**

As imagens utilizadas são provenientes de um **dataset de raio-X de pulmões**, que contém imagens de pulmões normais e com pneumonia. O modelo é treinado para aprender as características visuais que distinguem as duas classes e, em seguida, ser capaz de classificar novas imagens como pertencentes a uma dessas duas categorias.

## Tecnologias Utilizadas

- **Python 3.x**: Linguagem de programação principal do projeto.
- **TensorFlow 2.x**: Framework para construção e treinamento de modelos de aprendizado profundo.
- **Keras**: API de alto nível do TensorFlow para construção e treinamento de redes neurais.
- **NumPy**: Biblioteca fundamental para manipulação de arrays e operações numéricas.
- **Matplotlib**: Biblioteca para visualização de dados e gráficos.
- **Pillow**: Biblioteca para processamento de imagens.
- **scikit-learn**: Biblioteca para métricas de avaliação e ferramentas de aprendizado de máquina.

## Dataset

Este projeto utiliza um **dataset de imagens de pulmões** que contém duas classes:

1. **Normal**: Imagens de pulmões saudáveis.
2. **Pneumonia**: Imagens de pulmões com pneumonia.

Essas imagens são usadas para treinar o modelo a fim de classificá-las corretamente entre as duas categorias.

O dataset é um conjunto de imagens de raio-X de pulmões, onde o modelo aprende a distinguir os padrões que indicam a presença de pneumonia nos pulmões. Essas imagens são processadas e utilizadas para treinar o modelo de classificação de forma supervisionada.

## Pré-requisitos

Antes de executar o projeto, certifique-se de ter o Python instalado na sua máquina. Você pode verificar a versão do Python com o comando:

```bash
python --version
```

## Passo a Passo de como executar

1. Clone o repositorio
   
```bash
git clone https://github.com/CauZy-Goes/IA_Classificadora_IMG_Keras_TensorFlow.git
```

2. Com o terminal na pasta do projeto, inicialize o ambiente virtual python
   
```bash
python -m venv <nome_do_ambiente>
```

3. ative o ambiente virtual python

Windows 
```bash
venv\Scripts\activate
```

Mac
```bash
source venv/bin/activate
```

4. Instale as depencias do projeto usando o requirements.txt
   
```bash
pip install -r requirements.txt
```

5. Execute o projeto
   
```bash
python main.py
```



