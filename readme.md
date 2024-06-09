Introducción
==========================
Regresión logística con pytorch sobre el dataset Heart Failure Clinical Records (2020).

Requisitos
==========================
- Python 3.12.3 
- Pytorch
- Git

Instalación y dependencias
==========================
    $ git clone https://github.com/JuanPardos/pytorch-logistic-reg
    $ cd pytorch-logistic-reg
    $ python -m venv .venv
    Linux:
        $ source .venv/bin/activate
    Windows (cmd):
        $ .venv\Scripts\activate.bat
    $ pip install -r requirements.txt
    Pytorch:
        Nvidia: $ pip install torch
        AMD(Linux): $ pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
        CPU: $ pip install torch --index-url https://download.pytorch.org/whl/cpu
    $ python main.py

    (solo entrenamiento)

#### Notebook:
    Comandos anteriores +

    VsCode:
        $ pip install ipykernel
    Jupyter Lab:
        $ pip install jupyterlab
        $ jupyter lab
    
    (documentación, entrenamiento y predicción)

#### Google Colab (recomendado):
Subir el notebook, el archivo data.py, ejecutar y listo 🤯

Referencias
==========================
https://www.kaggle.com/datasets/aadarshvelu/heart-failure-prediction-clinical-records
