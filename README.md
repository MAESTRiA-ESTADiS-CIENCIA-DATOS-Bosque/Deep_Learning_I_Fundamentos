# Deep Learning I

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)


## Descripción
Curso impartido en el Doctorado en Ciencia de Datos e Inteligencia Artificial. U. el Bosque
### Semana 1. Introducción a TensorFlow, PyTorch y Jax
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos/blob/main/clase1/01_tensores_tensorflow.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos/blob/main/clase1/01_tensores_pytorch.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos/blob/main/clase1/01_tensores_jax.ipynb)
#### Carga de datos TensorFlow,PyTorch
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos/blob/main/clase1/02_carga_datos_tensorflow.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos/blob/main/clase1/02_carga_datos_pytorch.ipynb)

### Semana 2. MLP con TensorFlow, PyTorch y Jax (modulo Keras)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos/blob/main/clase2/01_Regresion.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos/blob/main/clase2/02_Clasificacion.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos/blob/main/clase2/Pipeline__PyTorchL_KerasJAX.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos/blob/main/clase2/Pipeline_TensorFlow.ipynb)


### Semana 3. CNNs y ajuste de hiperparámetros
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos/blob/main/clase3/CNN_TensorFlow.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos/blob/main/clase3/hyperTun.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos/blob/main/clase3/Transfer.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos/blob/main/clase3/Transfer_learning_MultiGPU.ipynb)


## 🛠️ Configuración del ambiente

Este proyecto usa [`uv`](https://github.com/astral-sh/uv) para gestionar dependencias y el entorno virtual.

### Requisitos previos

- Python 3.12 o superior (ajusta según tu `pyproject.toml`)
- `uv` instalado. Si no lo tienes:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# o con pip
pip install uv

git clone https://github.com/MAESTRiA-ESTADiS-CIENCIA-DATOS-Bosque/Deep_Learning_I_Fundamentos.git
cd Deep_Learning_I_Fundamentos
uv sync

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate