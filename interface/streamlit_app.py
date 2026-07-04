import os
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image

# Import core modules
from core.fourier import FourierProcessor
from core.coder import Coder
from core.metrics import Metrics
from core.utils import Utils

# Import modules
from modules.text import process_text, text_to_ascii, ascii_to_text
from modules.audio import process_audio
from modules.image import process_image, extract_edges, load_image_gray

# Create temp files directory
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Fourier Multimedia - Procesamiento Espectral",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling for Light Theme
st.markdown(r"""
<style>
    /* Main application background */
    .stApp {
        background-color: #F7F8FA !important;
        color: #1E293B !important;
    }

    /* Style for section title */
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.3rem;
    }

    /* Override Streamlit Card backgrounds */
    div[data-testid="stNotification"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 0.75rem !important;
    }
    
    /* Metrics Card styling */
    .metric-card {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        padding: 1.2rem;
        border-radius: 0.75rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h3, .metric-card h4 {
        color: #475569 !important;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .metric-card p {
        margin: 0 !important;
    }
    .metric-card hr {
        border: 0;
        border-top: 1px solid #F1F5F9;
        margin: 0.8rem 0;
    }

    /* Ensure Streamlit metrics values are dark and high-contrast */
    div[data-testid="stMetricValue"] > div {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] > div {
        color: #475569 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
    }

    /* Streamlit tabs style */
    button[data-baseweb="tab"] {
        color: #475569 !important;
        background-color: transparent !important;
        font-weight: 500 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #6366F1 !important;
        border-bottom-color: #6366F1 !important;
        font-weight: 700 !important;
    }

    /* Disabled Text Area styling - Normal pointer cursor, copyable and readable */
    div[data-testid="stTextArea"] textarea[disabled] {
        cursor: text !important;
        color: #1E293B !important;
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        -webkit-text-fill-color: #1E293B !important;
        opacity: 1 !important;
        box-shadow: none !important;
    }
    
    /* Enabled Text Area styling */
    div[data-testid="stTextArea"] textarea {
        color: #1E293B !important;
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
    }

    /* Streamlit buttons custom light theme styling */
    .stButton > button {
        background-color: #6366F1 !important;
        color: #FFFFFF !important;
        border: 1px solid #6366F1 !important;
        border-radius: 0.375rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        background-color: #4F46E5 !important;
        color: #FFFFFF !important;
        border-color: #4F46E5 !important;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.15) !important;
    }
    .stButton > button:active {
        background-color: #3730A3 !important;
        color: #FFFFFF !important;
    }
    
    /* Play/Pause hover fixes in plotly plot */
    .updatemenu-button {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #E2E8F0 !important;
        fill: #1E293B !important;
    }
    .updatemenu-button:hover {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        fill: #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate default media files if not present
def ensure_default_media():
    default_img_path = os.path.join(TEMP_DIR, "default_image.png")
    if not os.path.exists(default_img_path):
        # Generate a beautiful concentric pattern
        x = np.linspace(-10, 10, 256)
        y = np.linspace(-10, 10, 256)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(np.sqrt(xx**2 + yy**2)) * 127 + 128
        img = Image.fromarray(z.astype(np.uint8))
        img.save(default_img_path)
        
    default_audio_path = os.path.join(TEMP_DIR, "default_audio.wav")
    if not os.path.exists(default_audio_path):
        # Synthesize a multi-tone harmonic sound
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Synthesize fundamental (A4 = 440Hz) and harmonics
        sig = 0.5 * np.sin(2 * np.pi * 220 * t) + \
              0.25 * np.sin(2 * np.pi * 440 * t) + \
              0.15 * np.sin(2 * np.pi * 660 * t) + \
              0.1 * np.sin(2 * np.pi * 880 * t)
        Utils.save_wav(default_audio_path, sig, sr, 'int16')

# Initialize defaults
ensure_default_media()

# App title and description
st.markdown('<h1 class="main-title">Fourier Multimedia</h1>', unsafe_allow_html=True)
st.markdown("**Teoría de la Información y Sistemas de Comunicación** | Herramienta interactiva para la exploración y codificación espectral mediante Transformadas de Fourier.")

# Tabs configuration
tab_home, tab_text, tab_audio, tab_image, tab_epicycles, tab_experiments = st.tabs([
    "Inicio",
    "Texto",
    "Audio",
    "Imagen",
    "Epiciclos de Fourier",
    "Experimentos"
])

# ==============================================================================
# TAB 1: HOME / INICIO
# ==============================================================================
with tab_home:
    st.markdown('<div class="section-title">Análisis en el Dominio de la Frecuencia</div>', unsafe_allow_html=True)
    
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.write("""
        Esta aplicación es una plataforma académica para explorar cómo las señales multimedia (texto, audio e imágenes) pueden 
        ser representadas, transformadas, codificadas y analizadas en el **dominio de la frecuencia** utilizando la 
        **Transformada Discreta de Fourier (DFT)** mediante su algoritmo rápido **FFT (Fast Fourier Transform)**.
        
        ### ¿Cómo funciona el procesamiento espectral?
        Cualquier señal real puede descomponerse en una suma de senos y cosenos (ondas senoidales complejas) de diferentes 
        frecuencias, amplitudes y fases.
        """)
        
        # Mathematical formula block
        st.latex(r"""
        X(k) = \sum_{n=0}^{N-1} x(n) e^{-j \frac{2\pi}{N} k n} \quad \text{(FFT)}
        """)
        st.latex(r"""
        x(n) = \frac{1}{N} \sum_{k=0}^{N-1} X(k) e^{j \frac{2\pi}{N} k n} \quad \text{(IFFT)}
        """)
        
        st.write(r"""
        ### Los dos modos de codificación del proyecto:
        1. **Modo A (Permutación Espectral)**: Codificación sin pérdida que reordena de forma pseudoaleatoria los coeficientes complejos 
        de la FFT usando una clave criptográfica (SHA-256). Al aplicar la permutación inversa en el receptor, se reconstruye exactamente 
        la señal original.
        2. **Modo B (Modificación de Fase)**: Inyección controlada de ruido en la fase (\(\theta\)) de un porcentaje configurable de 
        coeficientes en una banda específica (bajas, medias, altas o todas las frecuencias). Sirve para analizar la tolerancia al ruido espectral.
        """)
        
    with col_right:
        st.info("""
        ### Estructura de las Señales
        
        * **Texto**: Se convierte cada carácter a su representación decimal ASCII. Este vector de números se trata como una señal temporal unidimensional.
        * **Audio**: El archivo WAV se carga, se normaliza y sus muestras temporales se procesan mediante FFT 1D directa.
        * **Imagen**: La matriz 2D de píxeles en escala de grises se transforma mediante FFT 2D, permitiendo operaciones espaciales como la extracción de bordes (filtros pasa-altas).
        """)
        
        # Add visual diagram
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/50/Fourier_transform_time_and_frequency_domains.gif", 
                 caption="Transformación del dominio del tiempo al de la frecuencia.", width="stretch")

# ==============================================================================
# TAB 2: TEXT MODULE
# ==============================================================================
with tab_text:
    st.markdown('<div class="section-title">Codificación Espectral de Texto</div>', unsafe_allow_html=True)
    
    # User inputs
    text_input = st.text_area("Introduce el texto a procesar:", "¡La Transformada de Fourier es una herramienta fundamental en telecomunicaciones!")
    
    col_params, col_metrics = st.columns([2, 1])
    
    with col_params:
        st.subheader("Configuración del Coder")
        coder_mode = st.radio("Modo de Codificación (Texto):", ["Modo A: Permutación", "Modo B: Modificación de Fase"], key="text_mode")
        
        params = {}
        # Key parameter is required for both modes to ensure exact recovery
        params["key"] = st.text_input("Clave de Cifrado (Texto):", "fourier123", type="password")
        
        if "Permutación" in coder_mode:
            mode_key = "permutation"
        else:
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                params["percentage"] = st.slider("% de Coeficientes a alterar (Texto)", 0.0, 100.0, 20.0, step=1.0)
            with col_b2:
                params["max_delta"] = st.slider("Delta de Fase Máximo (rad) (Texto)", 0.0, float(np.pi), float(np.pi/2), step=0.1)
            with col_b3:
                params["band"] = st.selectbox("Banda a intervenir (Texto)", ["low", "mid", "high", "all"])
            mode_key = "phase"
            
    # Process text
    if st.button("Procesar y Codificar Texto", type="primary"):
        res = process_text(text_input, mode_key, params)
        
        # Display Metrics
        with col_metrics:
            st.subheader("Métricas de Calidad")
            st.markdown(f"""
            <div class="metric-card">
                <h3>RMSE (Error Cuadrático Medio)</h3>
                <p style="font-size: 2rem; font-weight: bold; color: #4338ca;">{res['rmse']:.6f}</p>
                <hr>
                <h3>SNR (Relación Señal/Ruido)</h3>
                <p style="font-size: 2rem; font-weight: bold; color: #047857;">{res['snr']:.2f} dB</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Results area
        st.subheader("Resultados de la Reconstrucción")
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.text_area("Texto Original:", text_input, height=150, disabled=True)
        with col_res2:
            st.text_area("Texto Codificado (Dominio del Tiempo de la FFT alterada):", res["coded_text"], height=150, disabled=True)
        with col_res3:
            st.text_area("Texto Decodificado (Reconstruido):", res["decoded_text"], height=150, disabled=True)
            
        # Visualizations (Plots)
        st.subheader("Visualizaciones Espectrales")
        
        # 1. ASCII vector plot
        fig_ascii = go.Figure()
        fig_ascii.add_trace(go.Scatter(y=res["original_ascii"], name="Original ASCII", line=dict(color="#6366f1", width=2)))
        fig_ascii.add_trace(go.Scatter(y=res["decoded_ascii"], name="Decodificado ASCII", line=dict(color="#ec4899", width=1.5, dash='dash')))
        fig_ascii.update_layout(title="Señal Temporal (Valores ASCII)", xaxis_title="Índice de Carácter", yaxis_title="Código ASCII", template="plotly_white")
        st.plotly_chart(fig_ascii, width="stretch")
        
        # 2. Magnitude and Phase Spectrum
        fig_spectrum = make_subplots(rows=1, cols=2, subplot_titles=("Espectro de Magnitud", "Espectro de Fase"))
        
        mag = res["magnitude"]
        phase = res["phase"]
        
        fig_spectrum.add_trace(go.Scatter(y=mag, name="Magnitud", line=dict(color="#10b981")), row=1, col=1)
        fig_spectrum.add_trace(go.Scatter(y=phase, name="Fase (rad)", line=dict(color="#f59e0b")), row=1, col=2)
        
        fig_spectrum.update_layout(height=400, title_text="Descomposición Espectral (Original)", template="plotly_white")
        st.plotly_chart(fig_spectrum, width="stretch")
        
        # 3. Error plot
        fig_error = go.Figure()
        fig_error.add_trace(go.Scatter(y=res["error"], name="Error", line=dict(color="#ef4444"), fill='tozeroy'))
        fig_error.update_layout(title="Error de Reconstrucción (Decodificado - Original)", xaxis_title="Índice", yaxis_title="Diferencia", template="plotly_white")
        st.plotly_chart(fig_error, width="stretch")

# ==============================================================================
# TAB 3: AUDIO MODULE
# ==============================================================================
with tab_audio:
    st.markdown('<div class="section-title">Procesamiento Espectral de Audio</div>', unsafe_allow_html=True)
    
    # File upload
    audio_file = st.file_uploader("Sube un archivo WAV:", type=["wav"])
    
    # Select audio source
    if audio_file is not None:
        audio_path = os.path.join(TEMP_DIR, "uploaded_audio.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        st.success(f"Archivo cargado con éxito: {audio_file.name}")
    else:
        audio_path = os.path.join(TEMP_DIR, "default_audio.wav")
        st.info("Utilizando audio sintetizado por defecto. Puedes subir tu propio WAV arriba.")
        
    # Parameters
    col_aud_params, col_aud_play = st.columns([2, 1])
    
    with col_aud_params:
        st.subheader("Configuración del Coder (Audio)")
        coder_mode_audio = st.radio("Modo de Codificación (Audio):", ["Modo A: Permutación", "Modo B: Modificación de Fase"], key="audio_mode")
        
        params_audio = {}
        # Key parameter for both modes
        params_audio["key"] = st.text_input("Clave de Cifrado (Audio):", "audio_key_123", type="password")
        
        if "Permutación" in coder_mode_audio:
            mode_key_audio = "permutation"
        else:
            col_ab1, col_ab2, col_ab3 = st.columns(3)
            with col_ab1:
                params_audio["percentage"] = st.slider("% de Coeficientes a alterar (Audio)", 0.0, 100.0, 15.0, step=1.0)
            with col_ab2:
                params_audio["max_delta"] = st.slider("Delta de Fase Máximo (rad) (Audio)", 0.0, float(np.pi), float(np.pi/3), step=0.1)
            with col_ab3:
                params_audio["band"] = st.selectbox("Banda a intervenir (Audio)", ["low", "mid", "high", "all"])
            mode_key_audio = "phase"
            
    # Process audio
    if st.button("Procesar y Codificar Audio", type="primary"):
        with st.spinner("Procesando señal de audio en el dominio espectral..."):
            res_audio = process_audio(audio_path, mode_key_audio, params_audio)
            
            # Save temporary files to play in UI
            coded_wav_path = os.path.join(TEMP_DIR, "coded_audio.wav")
            decoded_wav_path = os.path.join(TEMP_DIR, "decoded_audio.wav")
            
            Utils.save_wav(coded_wav_path, res_audio["coded_signal"], res_audio["sample_rate"], res_audio["original_dtype"])
            Utils.save_wav(decoded_wav_path, res_audio["decoded_signal"], res_audio["sample_rate"], res_audio["original_dtype"])
            
            # Playback panel
            with col_aud_play:
                st.subheader("Reproductores de Audio")
                st.markdown("**Original:**")
                st.audio(audio_path)
                
                st.markdown("**Cifrado / Alterado (Espectral):**")
                st.audio(coded_wav_path)
                
                st.markdown("**Decodificado / Reconstruido:**")
                st.audio(decoded_wav_path)
                
                # Metrics
                st.subheader("Métricas del Canal")
                st.markdown(f"""
                <div class="metric-card">
                    <h3>SNR (Relación Señal/Ruido)</h3>
                    <p style="font-size: 1.8rem; font-weight: bold; color: #047857;">{res_audio['snr']:.2f} dB</p>
                    <hr>
                    <h3>RMSE</h3>
                    <p style="font-size: 1.8rem; font-weight: bold; color: #4338ca;">{res_audio['rmse']:.6f}</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Visualization
            st.subheader("Análisis Temporal y Espectral del Audio")
            
            # Subsample waveforms for plotting to speed up browser rendering
            orig_sig = res_audio["original_signal"]
            dec_sig = res_audio["decoded_signal"]
            
            step = max(1, len(orig_sig) // 2000)
            t_axis = np.arange(0, len(orig_sig)) / res_audio["sample_rate"]
            
            # Waveform Plot
            fig_wave = go.Figure()
            fig_wave.add_trace(go.Scatter(x=t_axis[::step], y=orig_sig[::step], name="Señal Original", line=dict(color="#4f46e5")))
            fig_wave.add_trace(go.Scatter(x=t_axis[::step], y=dec_sig[::step], name="Señal Decodificada", line=dict(color="#ec4899", dash='dash')))
            fig_wave.update_layout(title="Forma de Onda (Tiempo)", xaxis_title="Tiempo (s)", yaxis_title="Amplitud Normalizada", template="plotly_white")
            st.plotly_chart(fig_wave, width="stretch")
            
            # Spectrum Magnitude Plot
            freqs = np.fft.fftfreq(len(orig_sig), 1/res_audio["sample_rate"])
            # Plot positive frequencies only
            pos_mask = freqs >= 0
            pos_freqs = freqs[pos_mask]
            pos_mag = res_audio["magnitude"][pos_mask]
            
            step_freq = max(1, len(pos_freqs) // 2000)
            
            fig_aud_spec = go.Figure()
            fig_aud_spec.add_trace(go.Scatter(x=pos_freqs[::step_freq], y=pos_mag[::step_freq], name="Espectro de Magnitud", line=dict(color="#10b981")))
            fig_aud_spec.update_layout(title="Espectro de Magnitud (Original)", xaxis_title="Frecuencia (Hz)", yaxis_title="Magnitud", template="plotly_white")
            st.plotly_chart(fig_aud_spec, width="stretch")

# ==============================================================================
# TAB 4: IMAGE MODULE
# ==============================================================================
with tab_image:
    st.markdown('<div class="section-title">Procesamiento Espectral de Imágenes 2D</div>', unsafe_allow_html=True)
    
    # Image upload
    image_file = st.file_uploader("Sube una imagen (PNG, JPG):", type=["png", "jpg", "jpeg"])
    
    if image_file is not None:
        image_path = os.path.join(TEMP_DIR, "uploaded_image.png")
        img = Image.open(image_file).convert('L')
        img.save(image_path)
        st.success(f"Imagen cargada: {image_file.name}")
    else:
        image_path = os.path.join(TEMP_DIR, "default_image.png")
        st.info("Utilizando imagen de prueba por defecto. Puedes cargar una tuya arriba.")
        
    img_gray = load_image_gray(image_path)
    
    col_img_ctrls, col_img_metrics = st.columns([2, 1])
    
    with col_img_ctrls:
        st.subheader("Configuración Espectral")
        coder_mode_img = st.radio("Operación sobre la Imagen:", ["Modo A: Permutación", "Modo B: Modificación de Fase", "Extracción de Bordes (Pasa-Altas)"], key="image_mode")
        
        params_img = {}
        # Key parameter for both modes
        if coder_mode_img != "Extracción de Bordes (Pasa-Altas)":
            params_img["key"] = st.text_input("Clave de Cifrado (Imagen):", "image_pass_321", type="password")
            
        if "Permutación" in coder_mode_img:
            mode_key_img = "permutation"
        elif "Fase" in coder_mode_img:
            col_ib1, col_ib2, col_ib3 = st.columns(3)
            with col_ib1:
                params_img["percentage"] = st.slider("% de Coeficientes a alterar (Imagen)", 0.0, 100.0, 5.0, step=0.5)
            with col_ib2:
                params_img["max_delta"] = st.slider("Delta de Fase Máximo (rad) (Imagen)", 0.0, float(np.pi), float(np.pi), step=0.1)
            with col_ib3:
                params_img["band"] = st.selectbox("Banda a intervenir (Imagen)", ["low", "mid", "high", "all"])
            mode_key_img = "phase"
        else:
            cutoff = st.slider("Frecuencia de Corte de Borde (Radio Fraccional):", 0.01, 0.5, 0.05, step=0.01)
            mode_key_img = "edges"
            
    # Process image
    if st.button("Procesar Imagen en 2D", type="primary"):
        with st.spinner("Ejecutando transformada FFT2 y codificación espectral..."):
            
            # Row 1 columns
            col1_r1, col2_r1 = st.columns(2)
            # Row 2 columns
            col1_r2, col2_r2 = st.columns(2)
            
            if mode_key_img == "edges":
                edge_img = extract_edges(img_gray, cutoff)
                
                # Original spectrum log magnitude
                spectrum = FourierProcessor.fft2(img_gray)
                magnitude = FourierProcessor.magnitude(spectrum)
                shifted_magnitude = FourierProcessor.fft_shift(magnitude)
                log_magnitude = np.log1p(shifted_magnitude)
                fig_sp = px.imshow(log_magnitude, color_continuous_scale="Viridis", labels={"color": "Log-Mag"})
                fig_sp.update_layout(title="Espectro FFT2 Centrado (Magnitud Log)", template="plotly_white")
                
                # Filtered spectrum (High-pass)
                shifted_spectrum = FourierProcessor.fft_shift(spectrum)
                distances = Coder.get_frequency_distances(img_gray.shape)
                high_pass_mask = distances >= cutoff
                filtered_shifted = shifted_spectrum * high_pass_mask
                filtered_log_magnitude = np.log1p(FourierProcessor.magnitude(filtered_shifted))
                fig_cspec = px.imshow(filtered_log_magnitude, color_continuous_scale="Viridis")
                fig_cspec.update_layout(title="Espectro Filtrado (Pasa-Altas)", template="plotly_white")
                
                # Row 1 Display
                with col1_r1:
                    st.image(image_path, caption="Imagen Original (Grayscale)", width="stretch")
                with col2_r1:
                    st.plotly_chart(fig_sp, width="stretch")
                    
                # Row 2 Display
                with col1_r2:
                    st.plotly_chart(fig_cspec, width="stretch")
                with col2_r2:
                    st.image(edge_img, caption="Contornos Extraídos (Filtro Pasa-Altas)", width="stretch")
                    
                with col_img_metrics:
                    st.subheader("Información de Contornos")
                    st.write("Filtro Pasa-Altas aplicado:")
                    st.latex(r"H(u,v) = \begin{cases} 0 & \text{si } D(u,v) < D_0 \\ 1 & \text{si } D(u,v) \ge D_0 \end{cases}")
                    st.write(r"Radio de Corte (\(D_0\)): " + f"{cutoff:.3f}")
            else:
                res_img = process_image(img_gray, mode_key_img, params_img)
                
                # Original Spectrum plot
                fig_spec2d = px.imshow(res_img["log_magnitude"], color_continuous_scale="Viridis")
                fig_spec2d.update_layout(title="Espectro FFT2 (Magnitud Log)", template="plotly_white")
                
                # Coded Spectrum plot
                fig_cspec = px.imshow(res_img["coded_log_magnitude"], color_continuous_scale="Viridis")
                fig_cspec.update_layout(title="Espectro de la Imagen Codificada", template="plotly_white")
                
                # Row 1 Display
                with col1_r1:
                    st.image(image_path, caption="Original (Grayscale)", width="stretch")
                with col2_r1:
                    st.plotly_chart(fig_spec2d, width="stretch")
                    
                # Row 2 Display
                with col1_r2:
                    st.image(res_img["coded_image"].astype(np.uint8), caption="Imagen Codificada / Cifrada", width="stretch")
                with col2_r2:
                    st.image(res_img["decoded_image"].astype(np.uint8), caption="Imagen Reconstruida", width="stretch")
                    
                # Metrics Display
                with col_img_metrics:
                    st.subheader("Métricas de Distorsión 2D")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>MSE (Error Medio Cuadrático)</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: #b45309;">{res_img['mse']:.4f}</p>
                        <hr>
                        <h4>PSNR (Relación Señal/Ruido Pico)</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: #047857;">{res_img['psnr']:.2f} dB</p>
                        <hr>
                        <h4>SSIM (Índice de Similitud Estructural)</h4>
                        <p style="font-size: 1.5rem; font-weight: bold; color: #4338ca;">{res_img['ssim']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 5: FOURIER EPICYCLES
# ==============================================================================
with tab_epicycles:
    st.markdown('<div class="section-title">Visualización Dinámica: Epiciclos de Fourier</div>', unsafe_allow_html=True)
    st.write("La teoría de Fourier postula que cualquier curva cerrada bidimensional puede ser trazada mediante un conjunto de vectores rotatorios encadenados (epiciclos).")
    
    col_ep1, col_ep2 = st.columns([1, 3])
    
    with col_ep1:
        shape_select = st.selectbox("Selecciona la figura a dibujar:", ["Círculo", "Corazón", "Estrella", "Infinito", "Cuadrado"])
        num_vecs = st.slider("Número de vectores (Frecuencias a incluir):", 1, 100, 30)
        show_circles = st.checkbox("Mostrar círculos de órbita", True)
        
        # Build coordinates
        N_pts = 80
        t_vals = np.linspace(0, 2*np.pi, N_pts, endpoint=False)
        
        if shape_select == "Círculo":
            pts = 100 * np.exp(1j * t_vals)
        elif shape_select == "Corazón":
            x = 16 * np.sin(t_vals)**3
            y = 13 * np.cos(t_vals) - 5 * np.cos(2*t_vals) - 2 * np.cos(3*t_vals) - np.cos(4*t_vals)
            pts = 6 * (x + 1j * y)
        elif shape_select == "Estrella":
            r = np.where(np.arange(10) % 2 == 0, 100, 40)
            theta = np.linspace(0, 2*np.pi, 10, endpoint=False) - np.pi/2
            vertices = r * np.exp(1j * theta)
            v_ext = np.append(vertices, vertices[0])
            pts = np.interp(np.linspace(0, 10, N_pts), np.arange(11), np.real(v_ext)) + \
                  1j * np.interp(np.linspace(0, 10, N_pts), np.arange(11), np.imag(v_ext))
        elif shape_select == "Infinito":
            x = 100 * np.cos(t_vals) / (1 + np.sin(t_vals)**2)
            y = 100 * np.sin(t_vals) * np.cos(t_vals) / (1 + np.sin(t_vals)**2)
            pts = x + 1j * y
        else: # Cuadrado
            vertices = [100+100j, -100+100j, -100-100j, 100-100j]
            v_ext = np.append(vertices, vertices[0])
            pts = np.interp(np.linspace(0, 4, N_pts), np.arange(5), np.real(v_ext)) + \
                  1j * np.interp(np.linspace(0, 4, N_pts), np.arange(5), np.imag(v_ext))
                  
    with col_ep2:
        # Precompute epicircle chains by separating static centroid and rotating components
        c = np.fft.fft(pts)
        c_rot = c.copy()
        centroid = c[0] / N_pts
        c_rot[0] = 0.0 # Zero out static component
        
        # Sort rotating components by magnitude
        indices_rot = np.argsort(np.abs(c_rot))[::-1]
        freqs = np.fft.fftfreq(N_pts)
        
        # Tip path over time
        tip_x = []
        tip_y = []
        
        # Calculate full path traced by the tip
        for m in range(N_pts):
            phi = 2 * np.pi * m / N_pts
            val = centroid
            for idx in indices_rot[:num_vecs]:
                C_k = c_rot[idx] / N_pts
                freq = freqs[idx] * N_pts
                val += C_k * np.exp(1j * freq * phi)
            tip_x.append(np.real(val))
            tip_y.append(np.imag(val))
            
        # Create animated Plotly plot using Frame objects
        frames = []
        for m in range(N_pts):
            phi = 2 * np.pi * m / N_pts
            
            # Coordinates chain: origin -> centroid -> rotating vectors chained
            x_coords = [0.0, np.real(centroid)]
            y_coords = [0.0, np.imag(centroid)]
            curr_val = centroid
            
            for idx in indices_rot[:num_vecs]:
                C_k = c_rot[idx] / N_pts
                freq = freqs[idx] * N_pts
                vec_val = C_k * np.exp(1j * freq * phi)
                curr_val += vec_val
                x_coords.append(np.real(curr_val))
                y_coords.append(np.imag(curr_val))
                
            trace_x = tip_x[:m+1]
            trace_y = tip_y[:m+1]
            
            data_frame = []
            
            # Trace 1: Path drawn so far
            data_frame.append(go.Scatter(
                x=trace_x, y=trace_y, 
                mode='lines', name='Trazo', 
                line=dict(color='#ec4899', width=3)
            ))
            
            # Trace 2: The tip marker
            data_frame.append(go.Scatter(
                x=[x_coords[-1]], y=[y_coords[-1]], 
                mode='markers', name='Punta', 
                marker=dict(size=8, color='#ec4899'),
                showlegend=False
            ))
            
            # Trace 3: The static vector (from origin to centroid)
            data_frame.append(go.Scatter(
                x=[0.0, x_coords[1]], y=[0.0, y_coords[1]],
                mode='lines+markers', name='Centroide',
                line=dict(color='#94a3b8', width=1.5, dash='dash'),
                marker=dict(size=4, color='#94a3b8'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Draw each rotating vector and circle (exactly num_vecs items)
            for j in range(1, num_vecs + 1):
                start_x, start_y = x_coords[j], y_coords[j]
                end_x, end_y = x_coords[j+1], y_coords[j+1]
                
                idx = indices_rot[j-1]
                C_k = c_rot[idx] / N_pts
                rad = np.abs(C_k)
                
                # Dynamic style based on vector index
                factor = (j - 1) / num_vecs
                v_width = max(1.0, 4.0 * (1.0 - factor))
                v_opacity = max(0.4, 1.0 - 0.6 * factor)
                
                color_hue = int(220 + 120 * factor) % 360
                v_color = f"hsla({color_hue}, 80%, 45%, {v_opacity:.2f})"
                c_color = f"hsla({color_hue}, 80%, 45%, {(v_opacity * 0.25):.2f})"
                
                # Vector line (Radio y extremo)
                data_frame.append(go.Scatter(
                    x=[start_x, end_x], 
                    y=[start_y, end_y], 
                    mode='lines+markers', 
                    line=dict(color=v_color, width=v_width),
                    marker=dict(size=v_width * 1.8, color=v_color),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                # Orbit circle
                if show_circles:
                    theta_circle = np.linspace(0, 2*np.pi, 50)
                    cx = start_x + rad * np.cos(theta_circle)
                    cy = start_y + rad * np.sin(theta_circle)
                    data_frame.append(go.Scatter(
                        x=cx, y=cy, 
                        mode='lines', 
                        line=dict(color=c_color, width=max(0.5, v_width * 0.4)), 
                        showlegend=False, 
                        hoverinfo='skip'
                    ))
                    
            frames.append(go.Frame(data=data_frame, name=str(m)))
            
        # Initial chart layout matches the first frame to prevent glitches
        initial_data = frames[0].data
        
        fig_ep = go.Figure(
            data=initial_data,
            layout=go.Layout(
                xaxis=dict(range=[-160, 160], autorange=False, gridcolor="#E2E8F0"),
                yaxis=dict(range=[-160, 160], autorange=False, scaleanchor="x", scaleratio=1, gridcolor="#E2E8F0"),
                height=600,
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": 50, "redraw": False}, "fromcurrent": True}]
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 10},
                    "showactive": True,
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }]
            ),
            frames=frames
        )
        
        st.plotly_chart(fig_ep, width="stretch")

# ==============================================================================
# TAB 6: EXPERIMENTS / EXPERIMENTOS
# ==============================================================================
with tab_experiments:
    st.markdown('<div class="section-title">Análisis Experimental de Perturbaciones Espectrales</div>', unsafe_allow_html=True)
    st.write("Compare cómo afecta la inyección de ruido de fase a los diferentes rangos del espectro (bajas, medias y altas frecuencias).")
    
    test_text = "El procesamiento espectral nos permite comprender y codificar los datos multimedia descomponiéndolos en armónicos fundamentales."
    
    if st.button("Iniciar Simulación y Análisis de Frecuencias", type="primary"):
        with st.spinner("Ejecutando sweep experimental de ruido de fase..."):
            
            ascii_sig = text_to_ascii(test_text)
            
            percentages = np.linspace(0, 100, 11) # 0%, 10%, ..., 100%
            bands = ["low", "mid", "high", "all"]
            
            rmse_results = {b: [] for b in bands}
            snr_results = {b: [] for b in bands}
            
            spectrum = FourierProcessor.fft(ascii_sig)
            shifted = FourierProcessor.fft_shift(spectrum)
            
            for pct in percentages:
                for band in bands:
                    # Apply phase noise with key
                    modified_shifted = Coder.apply_phase_noise(shifted, pct, np.pi, band, key="experiment_key")
                    modified_spec = FourierProcessor.ifft_shift(modified_shifted)
                    
                    # Reconstruct
                    recon = np.real(FourierProcessor.ifft(modified_spec))
                    
                    # Compute metrics
                    rmse_results[band].append(Metrics.rmse(ascii_sig, recon))
                    
                    snr_val = Metrics.snr(ascii_sig, recon)
                    if snr_val == float('inf'):
                        snr_val = 80.0
                    elif snr_val == -float('inf'):
                        snr_val = -80.0
                    snr_results[band].append(snr_val)
                    
            # 1. Plot RMSE results
            fig_exp_rmse = go.Figure()
            colors = {"low": "#ef4444", "mid": "#f59e0b", "high": "#10b981", "all": "#6366f1"}
            labels = {"low": "Bajas Frecuencias (Filtro Pasa-Bajas)", "mid": "Frecuencias Medias", "high": "Altas Frecuencias (Filtro Pasa-Altas)", "all": "Espectro Completo"}
            
            for band in bands:
                fig_exp_rmse.add_trace(go.Scatter(x=percentages, y=rmse_results[band], name=labels[band], 
                                                 line=dict(color=colors[band], width=2.5), marker=dict(size=6)))
            fig_exp_rmse.update_layout(title="RMSE vs Porcentaje de Fase Modificada", 
                                       xaxis_title="Coeficientes Alterados (%)", yaxis_title="RMSE (Menor es mejor)", template="plotly_white")
            st.plotly_chart(fig_exp_rmse, width="stretch")
            
            # 2. Plot SNR results
            fig_exp_snr = go.Figure()
            for band in bands:
                fig_exp_snr.add_trace(go.Scatter(x=percentages, y=snr_results[band], name=labels[band], 
                                                 line=dict(color=colors[band], width=2.5), marker=dict(size=6)))
            fig_exp_snr.update_layout(title="SNR (dB) vs Porcentaje de Fase Modificada", 
                                      xaxis_title="Coeficientes Alterados (%)", yaxis_title="SNR (dB) (Mayor es mejor)", template="plotly_white")
            st.plotly_chart(fig_exp_snr, width="stretch")
            
            # 3. FFT Execution Time O(N log N) Demonstration
            st.subheader("Complejidad Computacional de la FFT")
            st.write(r"Medición empírica del tiempo de ejecución frente al tamaño del vector. Demuestra la complejidad matemática de \(O(N \log N)\) frente a la DFT ingenua de \(O(N^2)\).")
            
            sizes = [2**i for i in range(8, 19)] # from 256 to 262,144
            times = []
            
            for sz in sizes:
                sig_dummy = np.random.rand(sz)
                t_start = time.perf_counter()
                _ = np.fft.fft(sig_dummy)
                t_end = time.perf_counter()
                times.append((t_end - t_start) * 1000)
                
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=sizes, y=times, name="Tiempo FFT (ms)", line=dict(color="#8b5cf6", width=3), mode='lines+markers'))
            fig_time.update_layout(title="Tiempo de Cómputo de la FFT", xaxis_type="log", 
                                   xaxis_title="Tamaño de Señal (Escala Logarítmica N)", yaxis_title="Tiempo (ms)", template="plotly_white")
            st.plotly_chart(fig_time, width="stretch")
            
            st.success("Simulaciones completadas. Note cómo alterar las bajas frecuencias causa el mayor error (RMSE alto, SNR bajo) debido a que concentran la mayor energía del contenido original, mientras que las altas frecuencias son mucho más tolerantes a cambios de fase.")
