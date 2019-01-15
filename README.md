# Creación de un sistema que permita el diagnóstico del Alzheimer en imágenes MRI sagitales mediante técnicas de Deep Learning

Trabajo de Fin de Máster de Alejandro Puente Castro. 

Máster de Bioinformática para Ciencias de la Salud. 

Universidad de A Coruña.

## Librerías necesarias
Para la ejecución de este proyecto son necesarias las librerías OpenCV, Numpy, Scikit-Learn, Pandas, h5py, TensorFlow y Keras. Se instalan de la siguiente manera:

```bash
pip install opencv-python numpy scikit-learn pandas h5py tensorflow keras
```
## Uso

### Mediante ficheros de configuración

Para la ejecución mediante ficheros de configuración, se introducen los parámetros deseados en el fichero [Config.py](hhttps://github.com/TheMVS/tfm_deep_learning_alzheimer/blob/master/Config.py). Se ejecuta con el comando:

```bash
python Program.py
```

### Mediante intérprete de comandos

Se incorporó un pequeño intérprete de comandos para la definición de modelos, estrategias y problemas de forma interactiva. Para la ejecución se añade la opción -i:

```bash
python Program.py -i
```

### Ejecución de los tests unitarios

Se comprobó el correcto funcionamiento de las partes más críticas del código como lo son la lectura de los datos o la generación de modelos de forma interactiva. Para ver estos tests se ejecuta el comando:

```bash
python Program.py -t
```

### Mostrar ayuda

Se puede mostrar información de ayuda con el siguiente comando:

```bash
python Program.py -h
```

## Contribuciones

Los pull requests, uso, correcciones y sugerencias son bien recibidos. Para cambios mayores, abra una incidencia en la pestaña [issues](https://github.com/TheMVS/tfm_deep_learning_alzheimer/issues) o envíe un correo a a.puentec@udc.es.
