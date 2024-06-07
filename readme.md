Introducción
==========================
Regresión logística con pytorch sobre el dataset Heart Failure Clinical Records (2020).

Requisitos
==========================
- Python 3.12.3 
- Git

Quick start
==========================
    $ git clone https://github.com/JuanPardos/pytorch-logistic-reg
    $ cd pytorch-logistic-reg
    $ python -m venv .venv
    Linux:
        $ source .venv/bin/activate
    Windows (cmd):
        $ .venv\Scripts\activate.bat
    $ pip install -r requirements.txt
    $ python main.py 

    (solo entrenamiento)

#### Notebook:
    Comandos anteriores +

    VsCode:
        $ pip install ipykernel
    Jupyter Lab:
        $ pip install jupyterlab
        $ jupyter lab
    
    (incluye documentación, entrenamiento y predicción)

#### Google Colab (recomendado):
Subir el notebook, ejecutar y listo 🤯
        

Notas
==========================
Para aceleración por hardware en AMD (Linux) instalar torch con el siguiente comando: <br>
$ pip install torch --index-url https://download.pytorch.org/whl/rocm6.0


TODO
==========================
Script python separado para predecir

Referencias
==========================
https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records