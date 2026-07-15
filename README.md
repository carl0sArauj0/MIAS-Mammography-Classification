# MIAS Breast Cancer Classification using Transfer Learning

Este proyecto implementa un pipeline completo de Deep Learning para la clasificación de mamografías digitales utilizando la base de datos de la **Mammographic Image Analysis Society (MIAS)**. El objetivo es clasificar las imágenes en tres categorías: **Normal, Benigno y Maligno**.

## Decisiones Técnicas y Bitácora de Desarrollo

### 1. Manejo de Datos y Etiquetas (`generate_full_csv.py`)
Se desarrolló un script de automatización para procesar el `Info.txt` oficial de MIAS.
Se generó un dataset completo de 322 muestras, consolidando etiquetas duplicadas (eligiendo la anomalía más severa por imagen) y mapeando las clases a niveles numéricos (0: Normal, 1: Benigno, 2: Maligno).

### 2. Procesamiento de Imágenes: OpenCV vs PyPNM
Aunque en el Moodle mencionaba la librería `PyPNM`, durante la fase de exploración se detectaron limitaciones de compatibilidad y falta de herramientas integradas para el preprocesamiento.
Por esta razón se migró a OpenCV, ya que esta también permite la lectura nativa de formtatos ".pgm".


### 3. Preprocesamiento Crítico (CLAHE)
Las mamografías originales presentan un contraste muy bajo, lo que dificulta la extracción de características para filtros convolucionales estándar.
POr esta razón se implementó **CLAHE (Contrast Limited Adaptive Histogram Equalization)**. Esta técnica resalta bordes, espículas y microcalcificaciones sin generar ruido excesivo, permitiendo que el modelo detecte texturas tumorales de manera más eficiente.

### 4. Arquitectura del Modelo y Transfer Learning
Se utilizó una **ResNet18** pre-entrenada en ImageNet.
*   **Por qué ResNet18:** Dado que 322 imágenes sigue siendo un dataset pequeño para Visión Artificial, una arquitectura más ligera que ResNet50 ayuda a reducir el riesgo de sobreajuste.
*   **Fine-tuning:** Se reemplazó la capa completamente conectada por una estructura: `Linear(512) -> ReLU -> Dropout(0.4) -> Linear(3)`.

---

## 📈 Análisis de Resultados

### Curvas de Entrenamiento
*   **Pérdida:** Se nota una reducción constante en la pérdida de entrenamiento, llegando a niveles cercanos a 0.1. Sin embargo, la pérdida de validación tiende a subir después de las primeras 10-15 épocas, un indicador claro de **Overfitting**.
*   **Accuracy:** El modelo alcanza un pico de precisión en validación cercano al **65%**. El comportamiento oscilante de la curva de precisión de validación es producto del tamaño reducido del dataset.
![alt text](results/plots/training_curves.png)

### Matriz de Confusión
*   El modelo es altamente efectivo detectando la clase **Normal**.
*   Existe una dificultad intrínseca en distinguir entre **Benigno** y **Maligno**, lo cual es un desafío clínico real debido a la similitud morfológica de ciertos nódulos.
![alt text](results/plots/confusion_matrix.png)


---

## Estructura del Repositorio

```text
├── data/
│   ├── raw/                # Archivos .pgm originales
│   ├── processed/          # CSV de partición (test_split.csv)
│   └── info_mias.csv       # Etiquetas de las 322 imágenes
├── notebooks/
│   ├── 01_eda_and_pypnm.ipynb   # Exploración y validación de OpenCV/CLAHE
│   └── 03_model_inference.ipynb # Demo interactiva de predicción
├── results/
│   ├── models/             # Pesos guardados (best_mias_model.pth)
│   └── plots/              # Gráficas de rendimiento y matriz de confusión
├── src/
│   ├── dataset.py          # Clase MIASDataset personalizada
│   ├── model.py            # Definición de arquitectura ResNet18
│   ├── train.py            # Bucle de entrenamiento y validación
│   ├── test.py             # Script de evaluación de métricas finales
│   └── generate_full_csv.py # Generador de metadata completa
└── requirements.txt        # Dependencias del proyecto
```

---

## Cómo probar el Modelo (Notebook 03)

Para realizar una prueba individual con el modelo entrenado:

1.  Tener archivo del modelo: `results/models/best_mias_model.pth`.
2.  Abre el notebook `notebooks/03_model_inference.ipynb`.
3.  Ejecuta las celdas de inicialización.
4.  Usa la función `run_inference("ID_DE_IMAGEN")`.

---
