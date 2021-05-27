# Proyecto2_Embebidos
Este repositorio está dedicado al desarrollo de un sistema embebido de reconocimiento facial de emociones utilizando el flujo de Yocto Project, la bibliotecas de OpenCV además de Machine Learning

 La carpeta meta-programapy1 contiene los archivos para obtener el modelo a implementado en el programa, llamado model.tflite. Para ello se usó el modelo ya entrenado fer.json y el archivo de los pesos fer.h5

L carpeta conf, contiene las principales configuraciones de nuestro sistema hecho a la medida. El local.conf contiene las herencias de la recepi y el bblayers.conf contiene los layer utilizados

Para establecer la conexión ssh, lo hicimos por medio de putty, en donde ingresar la IP de nuestra raspberry para establecer la conexión.

Para la interfaz del servidor usamos FileZilla en donde recibe parámetros como la IP de la raspberry, login, el cual es root y el puerto (22).

