# Fourier Multimedia - Procesamiento Espectral

Este proyecto es una plataforma interactiva y académica construida en Python para la exploración, transformación, codificación, decodificación y análisis de señales multimedia (**texto, audio e imágenes**) en el **dominio de la frecuencia**, utilizando la **Transformada Discreta de Fourier (DFT/FFT)**.

La aplicación permite demostrar el uso de transformaciones espectrales para la alteración, cifrado y restauración de señales complejas, evaluando los resultados mediante métricas de distorsión y calidad analítica.

---

## Estructura del Proyecto

El proyecto está diseñado de forma modular y estructurada bajo las siguientes carpetas:

```text
fourier-multimedia/
│
├── pyproject.toml         # Configuración del proyecto y dependencias de uv
├── README.md              # Documentación del proyecto
├── .gitignore             # Archivos excluidos del control de versiones
├── app.py                 # Punto de entrada principal
│
├── core/                  # Núcleo matemático y utilidades
│   ├── fourier.py         # Procesamiento FFT/IFFT 1D y 2D
│   ├── coder.py           # Algoritmos de codificación espectral (Permutación y Fase)
│   ├── metrics.py         # Calculador de métricas (SNR, RMSE, MSE, PSNR, SSIM)
│   └── utils.py           # Cargador y normalizador de archivos WAV
│
├── modules/               # Capa lógica por tipo de datos
│   ├── text.py            # Conversión y cifrado de caracteres ASCII
│   ├── audio.py           # Gestión espectral de formas de onda 1D
│   └── image.py           # Operaciones en 2D y extracción de bordes
│
└── interface/             # Interfaz de usuario interactiva
    └── streamlit_app.py   # Aplicación Streamlit con gráficas Plotly
```

---

## Arquitectura del Sistema

La arquitectura está claramente separada en etapas para cada flujo de señal:
1. **Conversión**: Entrada a vector/matriz numérica (ASCII, muestras de audio o píxeles en grises).
2. **FFT**: Transformación directa a coeficientes complejos.
3. **Procesamiento Espectral**: Aplicación de modificaciones en frecuencia.
4. **IFFT**: Transformación inversa al dominio temporal o espacial.
5. **Reconstrucción**: Restauración al formato original (texto legible, archivos de audio jugables o imágenes de visualización).
6. **Métricas**: Evaluación cuantitativa antes y después del procesamiento.
7. **Visualización**: Renderizado en tiempo real.

---

## Algoritmos de Codificación (Coder)

El sistema implementa dos modos de alteración espectral en `core/coder.py`:

### Modo A: Permutation Mode (Cifrado Espectral sin pérdida)
- Genera una semilla reproducible a partir de una clave string usando **SHA-256**.
- Genera una permutación pseudoaleatoria con un generador NumPy (`default_rng(seed)`).
- Reordena los coeficientes de la FFT en el dominio de frecuencia.
- La decodificación calcula la permutación inversa restableciendo exactamente la señal al 100% de su estado original.

### Modo B: Phase Mode (Modificación y Restauración de Fase)
- Modifica la fase (\(\theta\)) de un subconjunto configurable de coeficientes espectrales.
- **Parámetros**:
  - `% de coeficientes` a alterar.
  - `Delta máximo` (desviación de fase en radianes).
  - `Banda de frecuencia` a intervenir (Bajas, Medias, Altas o todas las frecuencias).
- Utiliza la clave de cifrado de forma simétrica para calcular de forma idéntica las variaciones y restarlas en el decodificador, garantizando una **reconstrucción exacta** del contenido original.

---

## Requisitos e Instalación

El proyecto utiliza **uv** como gestor de paquetes de última generación en Python.

### 1. Clonar e Instalar dependencias
Asegúrate de tener `uv` instalado. Ejecuta el comando de sincronización de entorno en la carpeta raíz:
```bash
uv sync
```

### 2. Ejecutar la aplicación
Ejecuta la interfaz de Streamlit utilizando el wrapper principal:
```bash
uv run python app.py
```
O de forma directa:
```bash
uv run streamlit run interface/streamlit_app.py
```

---

## Pruebas Unitarias
El núcleo matemático y los algoritmos de codificación cuentan con cobertura de pruebas unitarias. Para ejecutarlas:
```bash
uv run python -m pytest
```

---

## Módulos de la Interfaz

* **Inicio**: Explicación interactiva del proyecto, fórmulas matemáticas LaTeX y conceptos de Fourier.
* **Texto**: Cifrado y descifrado de strings con visualización del espectro de magnitud, fase, señal ASCII y errores.
* **Audio**: Carga de archivos WAV locales con reproductor original, cifrado y reconstruido. Mide SNR y RMSE.
* **Imagen**: Carga de imágenes con matriz espectral 2D y un módulo pasa-altas para **extracción de contornos (bordes)** con radio de corte dinámico. Mide MSE, PSNR y SSIM.
* **Epiciclos**: Animación Plotly interactiva donde un tren de vectores circulares encadenados rotatorios dibuja figuras complejas (Círculo, Corazón, Estrella, Infinito o Cuadrado) según la DFT.
* **Experimentos**: Gráficos comparativos de error (RMSE/SNR) por cada banda espectral intervenida y una medición empírica de velocidad computacional FFT vs DFT para probar la complejidad \(O(N \log N)\).
