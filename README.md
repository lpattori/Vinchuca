# Implemetación de un modelo para reconocimiento de Vinchucas con [fast.ai](https://www.fast.ai) en [Render](https://render.com)

Este repo se basa en el modelo [fast.ai](https://github.com/fastai/fastai) para Render.

Esta aplicación de prueba se encuetra en: https://vinchuca.onrender.com. ¡Pruebela con sus imágenes!

Se pueden testear los cambios localmente instalando Docker y usando el comando:

```
docker build -t vinchuca . && docker run --rm -it -p 5000:5000 vinchuca
```

La guía para la implementación en producción en Render se encuentra en https://course.fast.ai/deployment_render.html.

Por favor usar [el thread de Render en el forum de fast.ai](https://forums.fast.ai/t/deployment-platform-render/33953) para preguntas y soporte.
