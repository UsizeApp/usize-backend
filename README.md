# usize-backend
## Redes neuronales
Backend para la aplicación movil de Usize

### Setup
```
pip install tensorflow keras numpy==1.16.4 pandas matplotlib pillow scikit-image
```

### Uso

0. (OPCIONAL) data_augmentation.py: con este script podemos generar más imágenes para la red. De las imágenes originales se pueden obtener otras con: ruido, reflejadas horizontalmente o rotadas.

1. Usize_front_images_data_save.py: para generar pickles de las imágenes (no podemos/queremos subirlas asi que dejaremos un link de descarga de el/los pickles generados).

2. UsizeNet-CreacionModeloConvolucional.py: creacion de la red neuronal.

3. UsizeNetModelTest.py: testeo de la red hecha anteriormente. Se entregan imágenes de "test_images" y se obtiene como resultado las medidas corporales para la persona en dicha imágen.