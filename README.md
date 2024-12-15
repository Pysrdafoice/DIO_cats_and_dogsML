# **DIO_cats_and_dogsML**  
**Identificando diferentes raças de cães e gatos com Transfer Learning.**

## **Descrição**  
Este projeto utiliza a técnica de Transfer Learning com redes de Deep Learning para classificar imagens de cães e gatos de diferentes raças. Foi aplicado o modelo pré-treinado **MobileNetV2**, otimizando o treinamento e aumentando a precisão, mesmo com um conjunto de dados limitado. O modelo base foi ajustado para extração de recursos e, em seguida, refinado para maximizar a acurácia nas predições.

---

## **Instalação**  
### **Dataset**  
O dataset utilizado pode ser acessado [neste link](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip).  

### **Bibliotecas utilizadas**  
Para rodar o projeto, as seguintes bibliotecas são necessárias:  
```python
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt
```

---

## **Metodologia**  
### **1. Preparação do dataset**  
- Validação e visualização das imagens para verificar a qualidade e estrutura do conjunto de dados.  
- Separação em conjuntos de treinamento, validação e teste:
  - **Lotes de validação**: 26  
  - **Lotes de teste**: 6  
- Pré-processamento das imagens:
  - Redimensionamento para 160x160 pixels.
  - Aumento de dados (data augmentation), incluindo rotações horizontais para melhorar a generalização e evitar overfitting.  
  - Normalização dos valores de pixel de [0, 255] para [-1, 1] usando a camada `tf.keras.layers.Rescaling`.

### **2. Transfer Learning com MobileNetV2**  
- **Modelo base**: MobileNetV2, pré-treinado no dataset **ImageNet**, com mais de 1.000 classes.  
- Extração de recursos da camada de gargalo, que retém maior generalidade para novas classificações.  
- Congelamento da base convolucional para evitar alterações nos pesos pré-treinados.  

### **3. Construção do modelo completo**  
- Adição de um cabeçalho para gerar previsões, usando a camada `GlobalAveragePooling2D` e um classificador denso.  
- Configuração do modelo:
  - Função de perda: `tf.keras.losses.BinaryCrossentropy` com `from_logits=True`.  
  - Taxa de aprendizado inicial: 0.0001.

### **4. Treinamento inicial**  
- Treinamento por **10 épocas**, resultando em:  
  - **Acurácia**: 95.92%  
  - **Erro**: 0.1271  
- Gráficos de curva de aprendizado gerados com `matplotlib` para análise da evolução do modelo.

### **5. Fine-tuning (ajuste fino)**  
- Descongelamento das camadas superiores do modelo base para permitir treinamento adicional.  
- Novo treinamento com ajuste dos pesos:
  - **Acurácia**: 98.51%  
  - **Erro**: 0.0451  

---

## **Resultados**  
Após os ajustes e o refinamento, o modelo alcançou excelente performance no conjunto de validação, com uma acurácia final de **98.51%**. A técnica de Transfer Learning demonstrou ser eficiente, permitindo treinar um modelo robusto mesmo com dados limitados.

### **Resumo do processo:**  
1. Pré-processamento e aumento dos dados para garantir robustez no treinamento.  
2. Uso do modelo pré-treinado **MobileNetV2** como base para extração de recursos.  
3. Ajuste fino das camadas superiores para aumentar a precisão.  
4. Visualização dos resultados por meio de gráficos, indicando melhoria contínua ao longo do treinamento.

Este projeto comprova a eficácia do Transfer Learning para tarefas de classificação de imagens, especialmente em cenários com conjuntos de dados menores.

---
