### Install

For Python 2.x and Python 3.x respectively:

```python
pip install ratios_pca_decisiontree


```

In the end, you can import the sub-package...

```python
from ratios_pca_decisiontree import ratios_pca_decisiontree

```
```python
Este algoritmo tiene contiene las siguientes funciones:
- La funcion ratios toma un dataframe y devuelve otro que contiene los mismos ratios más un conjunto de combinaciones entre los mismos.
- La funcion entrenamiento_error recibe un dataframe y componente deseada.
Esta ultima funcion va a realizar un pca con los datos, un arbol de decision (variable a explicar-> componente de entrada, variables explicativas -> variables del dataframe de entrada) y te va a devolver las tres variables que mejor explican esa componente.

Para seleccionar el mejor arbol de decision realiza validacion cruzada para cada profundidad (de 1 a 3) y plotea el error para cada una de las profundidades en test y train. Elige la profundidad con menor error en test.



```
