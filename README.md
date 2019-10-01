# usize-backend
## Redes neuronales
Backend para la aplicación movil de Usize

### Setup
```
pip install tensorflow keras numpy==1.16.4 pandas matplotlib pillow scikit-image scikit-learn
```

### Uso

0. (OPCIONAL) data_augmentation.py: con este script podemos generar más imágenes para la red. De las imágenes originales se pueden obtener otras con: ruido, reflejadas horizontalmente o rotadas.

1. Usize_front_images_data_save.py: para generar pickles de las imágenes (no podemos/queremos subirlas asi que dejaremos un link de descarga de el/los pickles generados).

2. UsizeNet-CreacionModeloConvolucional.py: creación de la red neuronal.

3. UsizeNetModelTest.py: testeo de la red hecha anteriormente. Se entregan imágenes de "test_images" y se obtiene como resultado las medidas corporales para la persona en dicha imágen.

### Pickles

Descargar archivos a la carpeta datapickles: 

+ [Imagenes frontales](https://drive.google.com/file/d/12aMl37fvd3z6eTWK9H8qYttKu8RmgVsF/view?usp=sharing)
+ [Imagenes frontales con ruido (opcional)](https://drive.google.com/file/d/1yH9E5uAeWv9HVilKXZENn5phSFidwVIU/view?usp=sharing)


### Redes

Descargar modelo a la raiz y renombrar como "UsizeNetConvolutional_front.h5".

NOTA: solo descargar uno. Vienen ordenados en relacion al rendimiento que nosotros consideramos (primero es mejor).

1. [UsizeNetConvolutional_front_2000-epochs_2019-09-15 21:54:52](https://drive.google.com/file/d/1F1_Wee-C2d2fTkWkaZcA24z503aeigKn/view?usp=sharing)
2. [UsizeNetConvolutional_front_5000-epochs](https://drive.google.com/file/d/1zBSbCMZS2Jw8rIbhBDTuB63fYvTSY3Iz/view?usp=sharing)