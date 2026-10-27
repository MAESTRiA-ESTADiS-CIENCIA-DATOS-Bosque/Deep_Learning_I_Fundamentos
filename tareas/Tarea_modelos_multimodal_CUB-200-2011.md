# Tarea 2: Fusión multimodal, red siamesa y CLIP con CUB-200-2011


## Objetivo

Usando el pipeline de la tarea 1, entrenarán y compararán cuatro formas de combinar o relacionar las modalidades de CUB-200-2011: **early fusion**, **late fusion**, una **red siames** y un modelo tipo **CLIP**. La pregunta que atraviesa toda la tarea es: ¿qué gana un modelo al usar más de una modalidad, y de qué manera conviene combinarlas?

- **Framework:** PyTorch, Jax, o TensorFlow a elección cualquier módulo.
- **Punto de partida:** sus loaders de la tarea 1. 
- **Entrega:** un notebook ejecutado y su PDF **(salidas visibles!)**, con cada decisión y cada resultado explicados en celdas markdown.

| Parte | Pregunta que responde | Modalidades | Papel de la especie |
| --- | --- | --- | --- |
| 1. Modelos unimodales | ¿Cuánto rinde cada modalidad por separado? | imagen, texto y atributos, cada una sola | objetivo de clasificación |
| 2. Early fusion | ¿Mejora un único modelo que ve todo a la vez? | las tres juntas | objetivo de clasificación |
| 3. Late fusion | ¿Basta con combinar las predicciones de los modelos de la parte 1? | las tres, por separado | objetivo de clasificación |
| 4. Red siamesa | ¿Se pueden reconocer especies nuevas comparando fotos? | imagen ↔ imagen | define qué pares son positivos |
| 5. CLIP | ¿Se pueden clasificar especies nunca vistas a partir de texto? | imagen ↔ texto | solo para el split y la evaluación |


Early y late fusion predicen una de las 200 especies; la red siamesa y CLIP aprenden distancias, y por eso pueden trabajar con especies que nunca vieron.

## Datos y punto de partida

Cada batch de la tarea 1 entrega tres modalidades de entrada y la especie como etiqueta:

| Pieza | Qué es | Forma (TF / PyTorch) |
| --- | --- | --- |
| Imagen | Foto recortada por *bounding box*, 224×224, normalizada con media/std de ImageNet | `(B, 224, 224, 3)` / `(B, 3, 224, 224)` |
| Texto | Una de las 10 captions de la imagen, como ids del vocabulario de la tarea 1 | `input_ids (B, L)` + `lengths (B,)` |
| Atributos | 312 atributos binarios + 6 variables de geometría escaladas con `StandardScaler` | `(B, 318)` |
| Especie (target) | Entero de 0 a 199 | `(B,)` |


**Embeddings de imagen precalculados (muy recomendado).** Usen un backbone preentrenado en ImageNet y **congelado** (por ejemplo ResNet50, que da vectores de 2048) para calcular una sola vez. El código que genera el archivo debe estar en el notebook, aunque luego lo carguen desde disco.

**Texto.** El encoder de texto puede ser `Embedding → LSTM` (o BiLSTM), entrenado desde cero con el vocabulario de la tarea 1. No usen modelos de lenguaje preentrenados: el foco de la tarea está en las redes recurrentes.

## Parte 1: modelos unimodales

Entrenen un clasificador de 200 especies por cada modalidad. 

1. **Imagen:** una cabeza `Dense` (por ejemplo, 512 unidades + dropout + `Dense(200)`) sobre los embeddings precalculados.
2. **Texto:** `Embedding(dim 128, con máscara de padding) → LSTM(256) → Dense(200)`.

Para cada modelo reporten exactitud, F1-macro,ROC-macro en test, las curvas de entrenamiento y validación, y las **10 parejas de especies que más se confunden** (una matriz de confusión de 200×200 no se puede leer).

- ¿Qué modalidad rinde mejor por sí sola y por qué creen que ocurre?
- ¿Las especies que confunde la imagen son las mismas que confunde el texto? Muestren dos o tres ejemplos con su foto y su caption.

## Parte 2: early fusion

Un solo modelo recibe las tres modalidades, junta sus representaciones en un vector y aprende una única cabeza de clasificación, todo con **una sola pérdida** y entrenado de punta a punta.

```text
embedding de imagen (2048) ──► Dense(512) ──┐
caption ──► Embedding ──► LSTM(256) ────────┤──► concat ──► Dense ──► Dense(200)
atributos (318) ──► Dense(128) ─────────────┘
```

**Sobre el nombre.** En sentido estricto, *early fusion* concatena las entradas crudas. Una foto y una secuencia de palabras no se pueden concatenar directamente, así que aquí fusionamos al nivel de *features* (en la literatura también se llama *intermediate fusion*). 


## Parte 3: late fusion, ablación y robustez

Combinen las **predicciones** de los tres modelos de la parte 1, sin volver a entrenarlos. Cada modelo da un vector de probabilidades sobre las 200 especies, y se combinan.

**Tabla resumen.** Una sola tabla con top-1 y top-5 en test de los modelos de la parte 1, las versiones de early fusion y  late fusion.

**Ablación.** Repitan la mejor regla de late fusion con cada par de modalidades (imagen+texto, imagen+atributos, texto+atributos). ¿Qué modalidad es imprescindible?

**Robustez.** Degraden **una** modalidad en test y midan cuánto cae cada modelo, early y late:

1. Texto: reemplazar la caption por una de **otra** especie elegida al azar.
2. Imagen: ruido gaussiano en los píxeles con σ ∈ {0.1, 0.25, 0.5}, recalculando los embeddings de test. Si añaden el ruido directamente al embedding, díganlo y justifíquenlo.

## Parte 4: red siamesa de imágenes

Una red siamesa usa **un solo encoder con pesos compartidos** para proyectar dos fotos en un espacio donde la distancia significa «misma especie» o «especie distinta». Así se pueden reconocer especies que el modelo nunca vio, solo comparándolas con una foto de referencia.

**Split (protocolo estándar de *metric learning* en CUB).** Entrenen con las especies 1–100 (separando el 10 % de sus imágenes para validación) y evalúen con **todas** las imágenes de las especies 101–200, que el modelo nunca ve. Así sus resultados son comparables con la literatura.

**Encoder.** Una cabeza sobre los embeddings precalculados, por ejemplo `Dense(512, relu) → Dense(128) → normalización L2`.

**Dos pérdidas a comparar**, con d la distancia euclidiana entre embeddings y m un margen:

1. **Contrastive loss**, sobre pares (y = 1 si son de la misma especie, 0 si no):

```latex
\mathcal{L} = y\, d^2 + (1 - y)\, \max(0,\; m - d)^2
```

**Muestreo.** Armen batches de P especies × K fotos (por ejemplo, P = 16 y K = 4) y comparen negativos **aleatorios** con negativos difíciles (*batch-hard mining*: para cada ancla, el positivo más lejano y el negativo más cercano del batch). En un dataset de grano fino como CUB, esta elección cambia mucho el resultado.

**Evaluación sobre las especies 101–200:**

- **Recall@K** (K = 1, 2, 4, 8): para cada imagen, ¿hay alguna de su misma especie entre sus K vecinos más cercanos?
- Una proyección 2D (PCA o t-SNE) de 10 especies de test, coloreadas por especie.

Preguntas a responder en markdown:

- ¿Qué pérdida y qué muestreo funcionaron mejor, y por qué?
- ¿En qué se diferencia esta red de un clasificador de 200 clases? ¿Qué permite hacer que un clasificador no puede?

## Parte 5: CLIP imagen–texto con especies no vistas

Entrenen dos encoders, uno de imagen y uno de texto, **sin pesos compartidos**, para que cada foto quede cerca de sus propias captions en un espacio común. La especie no entra en la pérdida; solo se usa para definir el split y para evaluar.

**Split zero-shot.** Usen el split de Reed et al. (2016), que acompaña a las captions: 150 especies para entrenar (separen el 10 % de sus imágenes como validación) y 50 especies que solo aparecen en la evaluación. Si no encuentran las listas de clases, usen las especies 1–150 y 151–200 y indíquenlo en el notebook.

**Encoders.**

- Imagen: cabeza de proyección sobre los embeddings precalculados, `Dense → Dense(256) → normalización L2`.
- Texto: `Embedding → LSTM` o BiLSTM `→ Dense(256) → normalización L2`.
- En cada época, cada imagen se empareja con una de sus 10 captions elegida al azar.

**Pérdida.** InfoNCE simétrica sobre la matriz de similitudes del batch, con temperatura aprendible (se aprende log(1/τ), se inicializa en 1/τ = 1/0.07 y se acota a 100):

```latex
S_{ij} = \frac{z^{\text{img}}_i \cdot z^{\text{txt}}_j}{\tau}, \qquad \mathcal{L} = \frac{1}{2}\Big[\operatorname{CE}(S,\, I) + \operatorname{CE}(S^{\top},\, I)\Big]
```

I indica que el par correcto de cada fila está en la diagonal. Usen batches de tamaño fijo (descarten el último batch incompleto) y registren 1/τ en cada época.

**Evaluación sobre las 50 especies no vistas:**

- **Clasificación zero-shot.** El prototipo de cada especie es el promedio, renormalizado, de los embeddings de todas sus captions. Cada imagen se asigna a la especie del prototipo más similar. Reporten top-1 y top-5; el azar es 2 %.
- **Búsqueda texto → imagen e imagen → texto**, con precisión@10.
- **Tres consultas escritas por ustedes** (por ejemplo, *«a small bird with a red head and black wings»*), mostrando las 5 imágenes recuperadas.
- La matriz de similitudes de un batch y las curvas de pérdida y de 1/τ.


