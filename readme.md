Introducción
==========================
Regresión logística con pytorch sobre el dataset Heart Failure Clinical Records (2020).

Requisitos
==========================
- Python 3.12.3 
- Git

Quick start
==========================
    **Se recomienda usar un entorno virtual de Python**

    $ git clone https://github.com/JuanPardos/pytorch-logistic-reg
    $ cd pytorch-logistic-reg
    $ python -m venv .venv
    Linux:
        $ source .venv/bin/activate
    Windows (cmd):
        $ .venv\Scripts\activate.bat
    $ pip install -r requirements.txt
    $ python train.py 

    (solo entrenamiento)

#### Notebook (Recomendado):
    VsCode:
        $ pip install ipykernel
    Jupyter Lab:
        $ pip install jupyterlab
        $ jupyter lab
    Ejecutar notebook.ipynb

Notas
==========================
- El notebook se puede usar en Google Colab o similares sin necesidad de instalar nada en local. <br>
- Para aceleración por hardware en AMD (Linux) instalar torch con el siguiente comando: <br>
    $ pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

Referencias
==========================
https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records