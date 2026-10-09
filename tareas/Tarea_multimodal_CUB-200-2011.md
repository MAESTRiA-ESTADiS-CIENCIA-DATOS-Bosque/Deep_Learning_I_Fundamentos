# Tarea: Pipeline de datos multimodal con CUB-200-2011

## Objetivo

En esta tarea se construye el pipeline de datos de una base multimodal (imagen + texto + tabular) y se entregan DataLoaders de train, validación y test en **PyTorch** (Dataset + DataLoader) y en **TensorFlow** (tf.data). No se entrena ningún modelo todavía.

Este pipeline es la base de las dos tareas siguientes, así que debe quedar limpio y reutilizable:

1. **Tarea actual:** carga, preprocesamiento y DataLoaders.

2. **Tarea siguiente ….**

Al terminar, cada batch debe entregar: la imagen normalizada, el texto como ids de palabras con su longitud real, los atributos tabulares escalados y la etiqueta de especie.

## La base de datos: CUB-200-2011 + captions

Usaremos tres modalidades por imagen: la foto del ave, 10 descripciones en inglés y un vector de atributos anotados. La etiqueta es la especie (200 clases).

**CUB-200-2011** ([Caltech-UCSD Birds-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/), Wah et al., 2011) es una base de clasificación de grano fino: 11.788 imágenes de 200 especies de aves, principalmente norteamericanas. Cada imagen trae una *bounding box*, 15 ubicaciones de partes y 312 atributos binarios. El split oficial tiene 5.994 imágenes de train y 5.794 de test.

**Las captions** vienen de [Reed et al. (CVPR 2016)](https://arxiv.org/abs/1605.05395), *Learning Deep Representations of Fine-Grained Visual Descriptions*. Recolectaron 10 descripciones por imagen con Amazon Mechanical Turk, sin nombrar la especie, centradas en la apariencia (colores, pico, alas, patrones).

| Modalidad | Origen                                          | Forma final en el batch                                             |
|-----------|-------------------------------------------------|---------------------------------------------------------------------|
| Imagen    | images/ (JPG, tamaños variables)                | PyTorch (B, 3, 224, 224) · TF (B, 224, 224, 3), float32 normalizado |
| Texto     | text_c10/ (10 captions por imagen)              | input_ids (B, L) con 0 = padding, y lengths (B,), int               |
| Tabular   | image_attribute_labels.txt + bounding_boxes.txt | (B, 318), float32: 312 atributos + 6 de geometría escalados         |
| Etiqueta  | image_class_labels.txt                          | (B,), int, de 0 a 199                                               |

## Descarga y estructura de archivos

Se necesitan dos descargas: las imágenes con anotaciones (Caltech) y las captions (Reed et al.).

1. **Imágenes y anotaciones:** CUB_200_2011.tgz (~1,1 GB) desde el [registro de Caltech Data](https://data.caltech.edu/records/20098).

2. **Captions:** el archivo *birds* enlazado en el README del [repositorio de Reed](https://github.com/reedscot/cvpr2016) (Google Drive). Dentro está la carpeta text_c10/, con un .txt de 10 líneas por imagen. Alternativa: el birds.zip de metadatos del repositorio de AttnGAN, cuya carpeta text/ tiene la misma estructura.

En Colab:

```bash
# 1) Imagenes + anotaciones (descargar el .tgz desde el registro de Caltech y subirlo, o usar wget con el enlace directo del registro)
tar -xzf CUB_200_2011.tgz -C data/

# 2) Captions (Google Drive del repositorio de Reed)
pip install gdown
gdown 0B0ywwgffWnLLZW9uVHNjb2JmNlE -O birds_reed.tar.gz
tar -xzf birds_reed.tar.gz -C data/
# Buscar la carpeta text_c10 y moverla a data/text_c10
find data -type d -name text_c10
```

Estructura esperada después de descomprimir:

```text
data/
├── CUB_200_2011/
│   ├── images/
│   │   └── 001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg
│   ├── images.txt                 # <image_id> <ruta relativa>
│   ├── image_class_labels.txt     # <image_id> <class_id 1..200>
│   ├── classes.txt                # <class_id> <nombre>
│   ├── train_test_split.txt       # <image_id> <1=train, 0=test>
│   ├── bounding_boxes.txt         # <image_id> <x> <y> <w> <h>
│   └── attributes/
│       └── image_attribute_labels.txt  # <image_id> <attr_id> <is_present> <certainty> <time>
└── text_c10/
    └── 001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.txt  # 10 captions
```

La ruta de cada caption es la misma que la de su imagen, cambiando .jpg por .txt. Esa correspondencia es la que une imagen y texto.

Los enlaces de Google Drive de 2016 a veces cambian o alcanzan su límite de descargas.

## Carga común: tabla maestra y splits

La carga se hace una sola vez, sin depender del framework: se arma una tabla con una fila por imagen y todas sus modalidades, y luego se divide en train, val y test. PyTorch y TensorFlow consumen exactamente los mismos arreglos.

Los splits quedan así:

| Split | Origen                                                     | Imágenes |
|-------|------------------------------------------------------------|----------|
| train | 90% del train oficial, estratificado por especie (seed=42) | 5.394    |
| val   | 10% del train oficial, estratificado por especie           | 600      |
| test  | test oficial completo                                      | 5.794    |

## Preprocesamiento: imagen, texto y tabular

Cada modalidad tiene su propio preprocesamiento, y la regla común es que todo lo que se *aprende* de los datos (scaler, vocabulario) se ajusta solo con train.

| Modalidad | Paso                                             | Train                                                             | Val / test                                   |
|-----------|--------------------------------------------------|-------------------------------------------------------------------|----------------------------------------------|
| Imagen    | Recorte por *bounding box*                       | Sí, con 10% de margen                                             | Sí, con 10% de margen                        |
| Imagen    | Redimensionar                                    | 256 × 256                                                         | 256 × 256                                    |
| Imagen    | Recorte a 224 × 224                              | Aleatorio                                                         | Central                                      |
| Imagen    | Aumentación                                      | Flip horizontal, brillo y contraste                               | Ninguna                                      |
| Imagen    | Normalización                                    | /255 y luego media/std de ImageNet                                | Igual                                        |
| Texto     | Limpieza                                         | Minúsculas, solo palabras (sin puntuación ni números)             | Igual                                        |
| Texto     | Vocabulario                                      | Construido con las captions de train; palabras con frecuencia ≥ 2 | Mismo vocabulario; palabras nuevas → \<unk\> |
| Texto     | Ids especiales                                   | \<pad\> = 0, \<unk\> = 1                                          | Igual                                        |
| Texto     | Largo máximo max_len                             | Percentil 95 de los largos de train; truncado y padding al final  | Mismo max_len                                |
| Texto     | Longitud real                                    | Se guarda junto a cada caption (mínimo 1)                         | Igual                                        |
| Texto     | Caption usada                                    | Una de las 10 al azar en cada lectura                             | Siempre la primera                           |
| Tabular   | 312 atributos binarios                           | Sin escalar (ya están en {0, 1})                                  | Igual                                        |
| Tabular   | 6 de geometría (posición, tamaño, aspecto, área) | StandardScaler ajustado con train                                 | Mismo scaler, solo transform                 |
