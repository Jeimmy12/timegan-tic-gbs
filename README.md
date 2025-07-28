# 🧠 TIC-BioRetroalimentación: Rehabilitación Asistida con Aprendizaje Automático
Este proyecto fue desarrollado como parte de una tesis de ingeniería enfocada en la mejora de procesos de rehabilitación mediante Inteligencia Artificial.

Autor: Jeimmy Eche
Institución: Escuela Politécnica Nacional
Año: 2025

Este proyecto forma parte de una iniciativa de investigación en el campo de la rehabilitación motora utilizando tecnologías de interacción humano-robot (HRpI), centrada en el **uso de datos sintéticos y modelos de regresión** para predecir la energía mecánica total (Hamiltoniano, _H_) durante ejercicios terapéuticos de miembros superiores.

## 📌 Objetivo

Desarrollar una arquitectura de aprendizaje automático capaz de:
1. **Generar datos sintéticos de calidad** con características similares a las señales reales, mediante el uso de **TimeGAN**.
2. **Entrenar modelos supervisados de regresión** sobre datos reales pasivos para estimar la variable _H_.
3. **Predecir H en escenarios activos reales y sintéticos**, con el fin de evaluar la viabilidad del uso de datos generados para tareas clínicas.

---

## 🧱 Estructura del Proyecto

TIC_JE/

│

├── pasivos/ # Datos reales pasivos por paciente

│ ├── comparationModels/ # Exploración comparación de modelos de ML de regresión

│ └── predictionSyntheticTIC/ # Predicción de H en sintéticos y reales

│ └── realData/ # Datos reales pasivos

├── activos/

│ ├── realData/ # Datos reales activos

│ └── syntheticData/ # Datos sintéticos generados con TimeGAN

│ └── timeGAN/ # Arquitectura TimeGAN

## ⚙️ Tecnologías Utilizadas

- Python 3.8+
- TensorFlow / Keras
- Scikit-learn
- TimeGAN (basado en [Yoon et al., 2019](https://arxiv.org/abs/1907.03143))
- Matplotlib & Seaborn
- Pandas / NumPy

---

## 📈 Metodología

Este proyecto sigue un enfoque basado en el modelo **CRISP-DM**:

1. **Comprensión del problema:** Estudio del Síndrome de Guillain-Barré y la necesidad de retroalimentación motora cuantitativa.
2. **Comprensión de los datos:** Análisis de señales _P_ (posición) obtenidas de HRpI en modo pasivo y activo.
3. **Preparación de los datos:** Normalización, segmentación y etiquetado.
4. **Modelado:** Entrenamiento de modelos generativos (TimeGAN) y regresores (Gradient Boosting, ANN, etc.)
5. **Evaluación:** Métricas como MSE, R², similitud de trayectorias (DTW, correlación) y análisis visual (PCA).
6. **Despliegue:** Aplicación del mejor modelo para estimar _H_ en escenarios sintéticos.

---

## 📊 Resultados Clave

- Se logró una alta correlación (>0.85) entre datos reales y sintéticos en señales de posición.
- El modelo de **Gradient Boosting** fue el más efectivo para predecir _H_ con errores reducidos en ejercicios pasivos y sintéticos.
- El análisis PCA muestra una distribución coherente entre muestras reales y generadas.
---

## 🧪 Cómo Usar

1. Clonar el repositorio:
```bash
git clone https://github.com/tuusuario/TIC-BioRetroalimentacion.git
cd TIC-BioRetroalimentacion

## ---
