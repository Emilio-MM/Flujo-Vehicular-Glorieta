# Análisis de Flujo Vehicular con Visión por Computadora (YOLOv8)
Este repositorio contiene un sistema automatizado para el conteo y análisis direccional de tráfico vehicular en intersecciones y glorietas. El núcleo del proyecto utiliza el modelo de detección de objetos YOLOv8 (entrenado/ajustado) en conjunto con OpenCV para rastrear vehículos frame por frame, calcular sus trayectorias y exportar métricas de flujo (entradas y salidas) a formatos de datos estructurados (CSV) y video renderizado.

<div align="center">
  <img src="Video-Salida-1.gif" width="400" />
</div>

## Estructura del Proyecto
El sistema está diseñado para procesar grabaciones de largo aliento (hasta 24 horas continuas) mediante la segmentación de videos. 

## La Lógica de Rastreo: Perspectiva Cenital y Coordenadas
A diferencia de un simple conteo de objetos en pantalla, este sistema requiere saber de dónde viene y hacia dónde va cada vehículo. Para lograrlo sin depender de hardware de radar, el código emplea un enfoque geométrico basado en el lienzo del video:

Imagen Base (Plano Cenital): Se extrae un frame limpio o se utiliza una representación cenital (vista desde arriba) de la glorieta. Esta imagen sirve como el mapa de fondo estático.

Regiones de Interés (ROIs) por Píxeles: Sobre esta imagen base, se mapean polígonos virtuales utilizando arreglos de coordenadas (X, Y) en píxeles. Cada polígono representa físicamente un carril de entrada o de salida de la glorieta.

Tracking por Centroides: Cuando YOLO detecta un coche, el código calcula el centro geométrico de su bounding box. Si el centroide del vehículo con el ID #24 cruza las coordenadas en píxeles de la "Entrada Norte" y segundos después cruza las coordenadas de la "Salida Sur", el sistema registra un recorrido completo y lo suma al flujo.

## Requisitos Técnicos
Para el procesamiento eficiente (especialmente para inferencia rápida del modelo de Deep Learning), se recomienda el siguiente entorno:  
-GPU: Tarjeta gráfica compatible con CUDA (ej. serie RTX) para acelerar el procesamiento de YOLOv8.  
-Software: Python 3.8 o superior.  

## Librerías de Python:
-ultralytics: Implementación oficial para la inferencia y tracking con YOLOv8.  
-opencv-python (cv2): Procesamiento de imágenes, lectura de video, dibujo de polígonos y renderizado en tiempo real.  
-numpy: Manejo de las matrices de píxeles y operaciones matemáticas de coordenadas.  
-pandas: (Opcional) Para la estructuración y exportación limpia de datos al archivo CSV.  

## Configuración y Personalización
Antes de ejecutar el script, puedes ajustar los parámetros clave en la sección de constantes del código fuente para adaptarlo a cualquier otra intersección:

CONFIDENCE_THRESHOLD: (ej. 0.5) Filtra falsos positivos. Solo toma en cuenta detecciones con una certeza mayor a este porcentaje.

POLIGONOS_ENTRADA / POLIGONOS_SALIDA: Arreglos de NumPy con las coordenadas [x, y] que definen las líneas invisibles en la calle. Deben ajustarse si la cámara cambia de perspectiva.

RUTAS_ARCHIVOS: Las variables VIDEO_PATH, MODEL_PATH, etc., utilizan rutas relativas para garantizar la portabilidad del repositorio.
