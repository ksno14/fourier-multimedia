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

# ─── Constants ────────────────────────────────────────────────────────────────
TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

# Light theme color palette
CLR_BG         = "#F7F8FA"   # Main background
CLR_CARD       = "#FFFFFF"   # Card / container background
CLR_SECONDARY  = "#F2F3F5"   # Plot canvas background
CLR_BORDER     = "#D9D9D9"   # Borders
CLR_TEXT       = "#1F2937"   # Primary text (dark charcoal)
CLR_MUTED      = "#4B5563"   # Secondary / muted text
CLR_ACCENT     = "#6366F1"   # Primary accent (Indigo)
CLR_SUCCESS    = "#047857"   # Green for positive metrics
CLR_AMBER      = "#B45309"   # Amber for warning metrics
CLR_GRID       = "#E2E8F0"   # Chart grid lines

# ─── Plotly light theme helper ────────────────────────────────────────────────
def light_fig(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    """Apply the uniform light theme to any Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=CLR_TEXT, size=16, family="sans-serif")),
        height=height,
        plot_bgcolor=CLR_SECONDARY,
        paper_bgcolor=CLR_CARD,
        font=dict(color=CLR_TEXT, family="sans-serif"),
        legend=dict(
            font=dict(color=CLR_TEXT),
            bgcolor=CLR_CARD,
            bordercolor=CLR_BORDER,
            borderwidth=1,
        ),
        xaxis=dict(
            gridcolor=CLR_GRID,
            linecolor=CLR_BORDER,
            tickcolor=CLR_BORDER,
            tickfont=dict(color=CLR_TEXT),
            title_font=dict(color=CLR_MUTED),
            zerolinecolor=CLR_GRID,
        ),
        yaxis=dict(
            gridcolor=CLR_GRID,
            linecolor=CLR_BORDER,
            tickcolor=CLR_BORDER,
            tickfont=dict(color=CLR_TEXT),
            title_font=dict(color=CLR_MUTED),
            zerolinecolor=CLR_GRID,
        ),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    # Also style any secondary axes that may exist
    for axis in ["xaxis2", "yaxis2", "xaxis3", "yaxis3"]:
        fig.update_layout(**{
            axis: dict(
                gridcolor=CLR_GRID,
                linecolor=CLR_BORDER,
                tickcolor=CLR_BORDER,
                tickfont=dict(color=CLR_TEXT),
                title_font=dict(color=CLR_MUTED),
                zerolinecolor=CLR_GRID,
            )
        })
    return fig


def _sample_open_polyline(points: np.ndarray, samples_per_segment: int) -> np.ndarray:
    """Sample an open polyline with evenly spaced points on each segment."""
    if len(points) < 2:
        return points.astype(np.float64)

    samples_per_segment = max(2, int(samples_per_segment))
    segments = []
    for start, end in zip(points[:-1], points[1:]):
        t_values = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
        segment = start + (end - start) * t_values[:, None]
        segments.append(segment)

    sampled = np.vstack(segments + [points[-1][None, :].astype(np.float64)])
    return sampled


def build_epicycle_shape(shape_select: str) -> tuple[np.ndarray, int, str | None]:
    """Return the point cloud, slider limit and an optional notice for the selected figure."""
    N_pts = 80
    notice = None

    t_vals = np.linspace(0, 2 * np.pi, N_pts, endpoint=False)

    if shape_select == "Círculo":
        pts = 100 * np.exp(1j * t_vals)
        max_vecs = len(pts)
    elif shape_select == "Corazón":
        x = 16 * np.sin(t_vals) ** 3
        y = 13 * np.cos(t_vals) - 5 * np.cos(2 * t_vals) - 2 * np.cos(3 * t_vals) - np.cos(4 * t_vals)
        pts = 6 * (x + 1j * y)
        max_vecs = len(pts)
    elif shape_select == "Estrella":
        r = np.where(np.arange(10) % 2 == 0, 100, 40)
        ang = np.linspace(0, 2 * np.pi, 10, endpoint=False) - np.pi / 2
        vx = np.append(r * np.cos(ang), (r * np.cos(ang))[0])
        vy = np.append(r * np.sin(ang), (r * np.sin(ang))[0])
        ts = np.linspace(0, 10, N_pts)
        pts = np.interp(ts, np.arange(11), vx) + 1j * np.interp(ts, np.arange(11), vy)
        max_vecs = len(pts)
    elif shape_select == "Infinito":
        x = 100 * np.cos(t_vals) / (1 + np.sin(t_vals) ** 2)
        y = 100 * np.sin(t_vals) * np.cos(t_vals) / (1 + np.sin(t_vals) ** 2)
        pts = x + 1j * y
        max_vecs = len(pts)
    elif shape_select == "Cuadrado":
        verts = [100 + 100j, -100 + 100j, -100 - 100j, 100 - 100j, 100 + 100j]
        vx = np.array([v.real for v in verts])
        vy = np.array([v.imag for v in verts])
        ts = np.linspace(0, 4, N_pts)
        pts = np.interp(ts, np.arange(5), vx) + 1j * np.interp(ts, np.arange(5), vy)
        max_vecs = len(pts)
    else:
        pts = 100 * np.exp(1j * t_vals)
        max_vecs = len(pts)

    return pts, max_vecs, notice


# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fourier Multimedia – Procesamiento Espectral",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS injection (full light theme override) ─────────────────────────
st.markdown(r"""
<style>
/* ── Root / App shell ─────────────────────────────────────── */
html, body, [data-testid="stApp"] {
    background-color: #F7F8FA !important;
    color: #1F2937 !important;
}

/* ── Top header bar ──────────────────────────────────────── */
[data-testid="stHeader"] {
    background-color: #F7F8FA !important;
    box-shadow: 0 1px 0 #D9D9D9 !important;
}
[data-testid="stHeader"] * {
    color: #1F2937 !important;
    fill: #1F2937 !important;
}
/* Hamburger / toolbar icons */
[data-testid="stToolbar"] * { color: #1F2937 !important; fill: #1F2937 !important; }

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #D9D9D9 !important;
}
[data-testid="stSidebar"] * { color: #1F2937 !important; }

/* ── Main content area ────────────────────────────────────── */
.main .block-container {
    background-color: #F7F8FA !important;
    padding-top: 1.5rem !important;
}

/* ── Titles & section headers ─────────────────────────────── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2,
.stMarkdown h3, .stMarkdown h4 {
    color: #1F2937 !important;
}
p, span, label, div { color: #1F2937 !important; }

/* Custom section-title element */
.section-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: #111827 !important;
    margin-top: 1rem;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #D9D9D9;
}

/* ── Tabs ─────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background-color: #FFFFFF !important;
    border-bottom: 1px solid #D9D9D9 !important;
}
[data-baseweb="tab"] {
    color: #4B5563 !important;
    background-color: transparent !important;
    font-weight: 500 !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #6366F1 !important;
    border-bottom: 2px solid #6366F1 !important;
    font-weight: 700 !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
.stButton > button,
button[kind="primary"],
button[kind="secondary"] {
    background-color: #6366F1 !important;
    color: #FFFFFF !important;
    border: 1px solid #6366F1 !important;
    border-radius: 0.4rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
    transition: background-color 0.18s, box-shadow 0.18s !important;
}
.stButton > button:hover {
    background-color: #4F46E5 !important;
    box-shadow: 0 4px 10px rgba(99,102,241,0.2) !important;
    color: #FFFFFF !important;
}
.stButton > button:active {
    background-color: #3730A3 !important;
    color: #FFFFFF !important;
}

/* ── Text inputs & password inputs ────────────────────────── */
div[data-testid="stTextInput"] > div,
div[data-testid="stTextInput"] input {
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
    border: 1px solid #D9D9D9 !important;
    border-radius: 0.4rem !important;
}
div[data-testid="stTextInput"] input::placeholder {
    color: #9CA3AF !important;
    opacity: 1 !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
    outline: none !important;
}

/* ── Text areas (enabled + disabled) ─────────────────────── */
div[data-testid="stTextArea"] textarea {
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
    -webkit-text-fill-color: #1F2937 !important;
    border: 1px solid #D9D9D9 !important;
    border-radius: 0.4rem !important;
    opacity: 1 !important;
}
div[data-testid="stTextArea"] textarea[disabled] {
    background-color: #F9FAFB !important;
    color: #1F2937 !important;
    -webkit-text-fill-color: #1F2937 !important;
    cursor: text !important;
    opacity: 1 !important;
}

/* ── Select boxes ─────────────────────────────────────────── */
div[data-testid="stSelectbox"] > div > div {
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
    border: 1px solid #D9D9D9 !important;
    border-radius: 0.4rem !important;
}

/* ── Sliders ──────────────────────────────────────────────── */
div[data-testid="stSlider"] > div > div > div {
    background-color: #D9D9D9 !important;
}
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] p {
    color: #1F2937 !important;
}

/* ── Radio buttons ────────────────────────────────────────── */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] p { color: #1F2937 !important; }
div[data-testid="stRadio"] > div > div > label > div > div:first-child {
    border-color: #6366F1 !important;
}

/* ── Checkboxes ───────────────────────────────────────────── */
div[data-testid="stCheckbox"] label,
div[data-testid="stCheckbox"] p { color: #1F2937 !important; }

/* ── File uploader ────────────────────────────────────────── */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 1.5px dashed #D9D9D9 !important;
    border-radius: 0.5rem !important;
    color: #1F2937 !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #4B5563 !important;
    fill: #4B5563 !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background-color: #F2F3F5 !important;
    border-color: #6366F1 !important;
}
/* Browse file button inside uploader */
[data-testid="stFileUploaderDropzone"] button {
    background-color: #F2F3F5 !important;
    color: #1F2937 !important;
    border: 1px solid #D9D9D9 !important;
    border-radius: 0.4rem !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background-color: #E5E7EB !important;
    color: #1F2937 !important;
}

/* ── Alert / info / success / warning boxes ───────────────── */
div[data-testid="stAlert"] {
    background-color: #FFFFFF !important;
    border-radius: 0.5rem !important;
    border: 1px solid #D9D9D9 !important;
}
div[data-testid="stAlert"] * { color: #1F2937 !important; }

/* ── Streamlit metric widget ──────────────────────────────── */
div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D9D9D9 !important;
    border-radius: 0.5rem !important;
    padding: 0.6rem 1rem !important;
}
div[data-testid="stMetricLabel"] > div { color: #4B5563 !important; font-weight: 500 !important; }
div[data-testid="stMetricValue"] > div { color: #111827 !important; font-weight: 700 !important; }

/* ── Spinner ──────────────────────────────────────────────── */
div[data-testid="stSpinner"] > div { color: #6366F1 !important; }

/* ── Custom metric card ───────────────────────────────────── */
.metric-card {
    background-color: #FFFFFF;
    color: #1F2937;
    padding: 1.2rem 1.4rem;
    border-radius: 0.6rem;
    border: 1px solid #D9D9D9;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    text-align: center;
    margin-bottom: 1rem;
}
.metric-card h3, .metric-card h4 {
    color: #4B5563 !important;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.metric-card p  { margin: 0 !important; }
.metric-card hr { border: 0; border-top: 1px solid #F1F5F9; margin: 0.75rem 0; }

/* ── Subheaders ───────────────────────────────────────────── */
div[data-testid="stSubheader"] > div { color: #1F2937 !important; }

/* ── Audio player ─────────────────────────────────────────── */
audio { filter: none !important; }

/* ── Plotly toolbar icons (light) ─────────────────────────── */
.modebar-btn svg { fill: #4B5563 !important; }
.modebar-btn:hover svg { fill: #1F2937 !important; }
.modebar { background-color: #FFFFFF !important; }

/* ── Stale state overlay (loading) ───────────────────────── */
div[data-testid="stStatusWidget"] > div { color: #1F2937 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Default media generation ─────────────────────────────────────────────────
def ensure_default_media() -> None:
    default_img_path = os.path.join(TEMP_DIR, "default_image.png")
    if not os.path.exists(default_img_path):
        x = np.linspace(-10, 10, 256)
        y = np.linspace(-10, 10, 256)
        xx, yy = np.meshgrid(x, y)
        z = np.sin(np.sqrt(xx ** 2 + yy ** 2)) * 127 + 128
        Image.fromarray(z.astype(np.uint8)).save(default_img_path)

    default_audio_path = os.path.join(TEMP_DIR, "default_audio.wav")
    if not os.path.exists(default_audio_path):
        sr = 16_000
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        sig = (
            0.50 * np.sin(2 * np.pi * 220 * t)
            + 0.25 * np.sin(2 * np.pi * 440 * t)
            + 0.15 * np.sin(2 * np.pi * 660 * t)
            + 0.10 * np.sin(2 * np.pi * 880 * t)
        )
        Utils.save_wav(default_audio_path, sig, sr, "int16")


ensure_default_media()

# ─── App title ────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='background:linear-gradient(135deg,#6366F1,#8B5CF6,#EC4899);"
    "-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    "font-size:2.6rem;font-weight:800;margin-bottom:0.2rem'>"
    "Fourier Multimedia</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#4B5563;font-size:1rem;margin-top:0'>"
    "<strong>Teoría de la Información y Sistemas de Comunicación</strong> &nbsp;|&nbsp; "
    "Exploración y codificación espectral mediante Transformadas de Fourier.</p>",
    unsafe_allow_html=True,
)

# ─── Tab bar ──────────────────────────────────────────────────────────────────
tab_home, tab_text, tab_audio, tab_image, tab_epicycles, tab_experiments = st.tabs([
    "Inicio",
    "Texto",
    "Audio",
    "Imagen",
    "Epiciclos",
    "Experimentos",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – INICIO
# ══════════════════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown('<div class="section-title">Análisis en el Dominio de la Frecuencia</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.write("""
        Esta plataforma académica permite explorar cómo las señales multimedia
        (texto, audio e imágenes) pueden representarse, transformarse, codificarse
        y analizarse en el **dominio de la frecuencia** usando la
        **Transformada Discreta de Fourier (DFT/FFT)**.

        ### ¿Cómo funciona el procesamiento espectral?
        Cualquier señal real puede descomponerse en una suma de ondas senoidales
        complejas de diferentes frecuencias, amplitudes y fases.
        """)

        st.latex(r"X(k) = \sum_{n=0}^{N-1} x(n)\,e^{-j\frac{2\pi}{N}kn} \quad\text{(FFT)}")
        st.latex(r"x(n) = \frac{1}{N}\sum_{k=0}^{N-1} X(k)\,e^{j\frac{2\pi}{N}kn} \quad\text{(IFFT)}")

        st.write(r"""
        ### Modos de codificación

        **Modo A – Permutación Espectral**: Reordena pseudoaleatoriamente los
        coeficientes de la FFT con una clave SHA-256. La permutación inversa
        recupera la señal exactamente.

        **Modo B – Modificación de Fase**: Inyecta ruido controlado en la fase
        \(\theta\) de un subconjunto de coeficientes en la banda elegida.
        La clave permite recuperar la señal exactamente restando los deltas.
        """)

    with col_r:
        st.info("""
        ### Estructura de las señales

        **Texto** → Vector ASCII 1D\n
        **Audio** → Muestras WAV normalizadas\n
        **Imagen** → Matriz de píxeles en escala de grises 2D
        """)
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/5/50/"
            "Fourier_transform_time_and_frequency_domains.gif",
            caption="Transformación: dominio del tiempo → frecuencia.",
            width="stretch",
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – TEXTO
# ══════════════════════════════════════════════════════════════════════════════
with tab_text:
    st.markdown('<div class="section-title">Codificación Espectral de Texto</div>', unsafe_allow_html=True)

    text_input = st.text_area(
        "Introduce el texto a procesar:",
        "¡La Transformada de Fourier es una herramienta fundamental en telecomunicaciones!",
    )

    col_params, col_metrics = st.columns([2, 1])

    with col_params:
        st.subheader("Configuración del coder")
        coder_mode = st.radio(
            "Modo de codificación:",
            ["Modo A: Permutación", "Modo B: Modificación de Fase"],
            key="text_mode",
        )

        params: dict = {}
        params["key"] = st.text_input(
            "Clave de cifrado (Texto):", "fourier123", type="password", key="text_key"
        )

        if "Permutación" in coder_mode:
            mode_key = "permutation"
        else:
            c1, c2, c3 = st.columns(3)
            params["percentage"] = c1.slider("% coeficientes alterados", 0.0, 100.0, 20.0, 1.0, key="txt_pct")
            params["max_delta"]   = c2.slider("Delta de fase (rad)",       0.0, float(np.pi), float(np.pi / 2), 0.1, key="txt_delta")
            params["band"]        = c3.selectbox("Banda", ["low", "mid", "high", "all"], key="txt_band")
            mode_key = "phase"

    if st.button("Procesar y codificar texto", type="primary", key="btn_text"):
        res = process_text(text_input, mode_key, params)

        with col_metrics:
            st.subheader("Métricas de calidad")
            st.markdown(f"""
            <div class="metric-card">
                <h3>RMSE</h3>
                <p style="font-size:2rem;font-weight:700;color:#4338CA">{res['rmse']:.6f}</p>
                <hr>
                <h3>SNR</h3>
                <p style="font-size:2rem;font-weight:700;color:#047857">{res['snr']:.2f} dB</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Resultados de la reconstrucción")
        r1, r2, r3 = st.columns(3)
        r1.text_area("Texto original:",          text_input,           height=150, disabled=True, key="ta_orig")
        r2.text_area("Texto codificado:",         res["coded_text"],    height=150, disabled=True, key="ta_coded")
        r3.text_area("Texto decodificado:",       res["decoded_text"],  height=150, disabled=True, key="ta_dec")

        st.subheader("Visualizaciones espectrales")

        # ASCII signal
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=res["original_ascii"], name="Original",    line=dict(color="#6366F1", width=2)))
        fig.add_trace(go.Scatter(y=res["decoded_ascii"],  name="Decodificado", line=dict(color="#EC4899", width=1.5, dash="dash")))
        fig.update_xaxes(title_text="Índice de carácter")
        fig.update_yaxes(title_text="Código ASCII")
        st.plotly_chart(light_fig(fig, "Señal temporal (valores ASCII)"), width="stretch")

        # Magnitude + Phase
        fig2 = make_subplots(rows=1, cols=2, subplot_titles=["Espectro de Magnitud", "Espectro de Fase"])
        fig2.add_trace(go.Scatter(y=res["magnitude"], name="Magnitud", line=dict(color="#10B981")), row=1, col=1)
        fig2.add_trace(go.Scatter(y=res["phase"],     name="Fase (rad)", line=dict(color="#F59E0B")), row=1, col=2)
        fig2.update_annotations(font_color=CLR_TEXT)
        st.plotly_chart(light_fig(fig2, "Descomposición espectral", height=380), width="stretch")

        # Error
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=res["error"], name="Error", line=dict(color="#EF4444"), fill="tozeroy", fillcolor="rgba(239,68,68,0.1)"))
        fig3.update_xaxes(title_text="Índice")
        fig3.update_yaxes(title_text="Diferencia")
        st.plotly_chart(light_fig(fig3, "Error de reconstrucción (Decodificado − Original)"), width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – AUDIO
# ══════════════════════════════════════════════════════════════════════════════
with tab_audio:
    st.markdown('<div class="section-title">Procesamiento Espectral de Audio</div>', unsafe_allow_html=True)

    audio_file = st.file_uploader("Sube un archivo WAV:", type=["wav"], key="aud_upload")

    if audio_file is not None:
        audio_path = os.path.join(TEMP_DIR, "uploaded_audio.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        st.success(f"Archivo cargado: {audio_file.name}")
    else:
        audio_path = os.path.join(TEMP_DIR, "default_audio.wav")
        st.info("Usando audio sintetizado por defecto. Sube tu propio WAV arriba.")

    col_params_a, col_play_a = st.columns([2, 1])

    with col_params_a:
        st.subheader("Configuración del coder (Audio)")
        coder_mode_audio = st.radio(
            "Modo de codificación:",
            ["Modo A: Permutación", "Modo B: Modificación de Fase"],
            key="audio_mode",
        )
        params_audio: dict = {}
        params_audio["key"] = st.text_input(
            "Clave de cifrado (Audio):", "audio_key_123", type="password", key="aud_key"
        )
        if "Permutación" in coder_mode_audio:
            mode_key_audio = "permutation"
        else:
            a1, a2, a3 = st.columns(3)
            params_audio["percentage"] = a1.slider("% coeficientes", 0.0, 100.0, 15.0, 1.0,           key="aud_pct")
            params_audio["max_delta"]  = a2.slider("Delta (rad)",    0.0, float(np.pi), float(np.pi / 3), 0.1, key="aud_delta")
            params_audio["band"]       = a3.selectbox("Banda", ["low", "mid", "high", "all"],           key="aud_band")
            mode_key_audio = "phase"

    if st.button("Procesar y codificar audio", type="primary", key="btn_audio"):
        with st.spinner("Procesando señal de audio…"):
            res_audio = process_audio(audio_path, mode_key_audio, params_audio)

            coded_wav   = os.path.join(TEMP_DIR, "coded_audio.wav")
            decoded_wav = os.path.join(TEMP_DIR, "decoded_audio.wav")
            Utils.save_wav(coded_wav,   res_audio["coded_signal"],   res_audio["sample_rate"], res_audio["original_dtype"])
            Utils.save_wav(decoded_wav, res_audio["decoded_signal"],  res_audio["sample_rate"], res_audio["original_dtype"])

        with col_play_a:
            st.subheader("Reproductores")
            st.markdown("**Original:**")
            st.audio(audio_path)
            st.markdown("**Codificado:**")
            st.audio(coded_wav)
            st.markdown("**Decodificado / Reconstruido:**")
            st.audio(decoded_wav)
            st.markdown(f"""
            <div class="metric-card">
                <h3>SNR</h3>
                <p style="font-size:1.8rem;font-weight:700;color:#047857">{res_audio['snr']:.2f} dB</p>
                <hr>
                <h3>RMSE</h3>
                <p style="font-size:1.8rem;font-weight:700;color:#4338CA">{res_audio['rmse']:.6f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Análisis temporal y espectral")

        orig = res_audio["original_signal"]
        dec  = res_audio["decoded_signal"]
        step = max(1, len(orig) // 2_000)
        t_ax = np.arange(len(orig)) / res_audio["sample_rate"]

        fw = go.Figure()
        fw.add_trace(go.Scatter(x=t_ax[::step], y=orig[::step], name="Original",    line=dict(color="#6366F1")))
        fw.add_trace(go.Scatter(x=t_ax[::step], y=dec[::step],  name="Decodificada", line=dict(color="#EC4899", dash="dash")))
        fw.update_xaxes(title_text="Tiempo (s)")
        fw.update_yaxes(title_text="Amplitud normalizada")
        st.plotly_chart(light_fig(fw, "Forma de onda (tiempo)"), width="stretch")

        freqs    = np.fft.fftfreq(len(orig), 1 / res_audio["sample_rate"])
        pos      = freqs >= 0
        step_f   = max(1, pos.sum() // 2_000)
        fs       = go.Figure()
        fs.add_trace(go.Scatter(x=freqs[pos][::step_f], y=res_audio["magnitude"][pos][::step_f],
                                name="Magnitud", line=dict(color="#10B981")))
        fs.update_xaxes(title_text="Frecuencia (Hz)")
        fs.update_yaxes(title_text="Magnitud")
        st.plotly_chart(light_fig(fs, "Espectro de magnitud (Original)"), width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – IMAGEN
# ══════════════════════════════════════════════════════════════════════════════
with tab_image:
    st.markdown('<div class="section-title">Procesamiento Espectral de Imágenes 2D</div>', unsafe_allow_html=True)

    image_file = st.file_uploader("Sube una imagen (PNG / JPG):", type=["png", "jpg", "jpeg"], key="img_upload")

    if image_file is not None:
        image_path = os.path.join(TEMP_DIR, "uploaded_image.png")
        Image.open(image_file).convert("L").save(image_path)
        st.success(f"Imagen cargada: {image_file.name}")
    else:
        image_path = os.path.join(TEMP_DIR, "default_image.png")
        st.info("Usando imagen de prueba por defecto. Puedes subir la tuya arriba.")

    img_gray = load_image_gray(image_path)

    col_ctrls, col_met = st.columns([2, 1])

    with col_ctrls:
        st.subheader("Configuración espectral")
        coder_mode_img = st.radio(
            "Operación:",
            ["Modo A: Permutación", "Modo B: Modificación de Fase", "Extracción de Bordes (Pasa-Altas)"],
            key="image_mode",
        )
        params_img: dict = {}
        if coder_mode_img != "Extracción de Bordes (Pasa-Altas)":
            params_img["key"] = st.text_input(
                "Clave de cifrado (Imagen):", "image_pass_321", type="password", key="img_key"
            )

        if "Permutación" in coder_mode_img:
            mode_key_img = "permutation"
        elif "Fase" in coder_mode_img:
            i1, i2, i3 = st.columns(3)
            params_img["percentage"] = i1.slider("% coeficientes", 0.0, 100.0, 5.0, 0.5,            key="img_pct")
            params_img["max_delta"]  = i2.slider("Delta (rad)",    0.0, float(np.pi), float(np.pi), 0.1, key="img_delta")
            params_img["band"]       = i3.selectbox("Banda", ["low", "mid", "high", "all"],          key="img_band")
            mode_key_img = "phase"
        else:
            cutoff      = st.slider("Frecuencia de corte (radio fraccional):", 0.01, 0.50, 0.05, 0.01, key="img_cutoff")
            mode_key_img = "edges"

    if st.button("Procesar imagen 2D", type="primary", key="btn_image"):
        with st.spinner("Ejecutando FFT2 y codificación espectral…"):
            r1c1, r1c2 = st.columns(2)
            r2c1, r2c2 = st.columns(2)

            if mode_key_img == "edges":
                edge_img = extract_edges(img_gray, cutoff)
                spectrum_e = FourierProcessor.fft2(img_gray)
                log_mag_e  = np.log1p(FourierProcessor.fft_shift(FourierProcessor.magnitude(spectrum_e)))

                shifted_e   = FourierProcessor.fft_shift(spectrum_e)
                hp_mask     = Coder.get_frequency_distances(img_gray.shape) >= cutoff
                filt_log    = np.log1p(FourierProcessor.magnitude(shifted_e * hp_mask))

                fig_orig_sp = px.imshow(log_mag_e, color_continuous_scale="Viridis")
                fig_filt_sp = px.imshow(filt_log,  color_continuous_scale="Viridis")

                fig_orig_sp.update_layout(title=dict(text="Espectro FFT2 Original (Log-Mag)", font=dict(color=CLR_TEXT)),
                                          paper_bgcolor=CLR_CARD, plot_bgcolor=CLR_SECONDARY,
                                          font=dict(color=CLR_TEXT), coloraxis_colorbar=dict(tickfont=dict(color=CLR_TEXT)))
                fig_filt_sp.update_layout(title=dict(text="Espectro filtrado Pasa-Altas", font=dict(color=CLR_TEXT)),
                                          paper_bgcolor=CLR_CARD, plot_bgcolor=CLR_SECONDARY,
                                          font=dict(color=CLR_TEXT), coloraxis_colorbar=dict(tickfont=dict(color=CLR_TEXT)))

                with r1c1: st.image(image_path, caption="Imagen Original (Grayscale)", width="stretch")
                with r1c2: st.plotly_chart(fig_orig_sp, width="stretch")
                with r2c1: st.plotly_chart(fig_filt_sp, width="stretch")
                with r2c2: st.image(edge_img, caption="Contornos extraídos (Pasa-Altas)", width="stretch")

                with col_met:
                    st.subheader("Filtro de bordes")
                    st.latex(r"H(u,v) = \begin{cases}0 & D(u,v)<D_0\\1 & D(u,v)\ge D_0\end{cases}")
                    st.write(r"Radio de corte (\(D_0\)): " + f"{cutoff:.3f}")

            else:
                res_img = process_image(img_gray, mode_key_img, params_img)

                def _imshow(arr, title):
                    f = px.imshow(arr, color_continuous_scale="Viridis")
                    f.update_layout(
                        title=dict(text=title, font=dict(color=CLR_TEXT)),
                        paper_bgcolor=CLR_CARD, plot_bgcolor=CLR_SECONDARY,
                        font=dict(color=CLR_TEXT),
                        coloraxis_colorbar=dict(tickfont=dict(color=CLR_TEXT)),
                    )
                    return f

                with r1c1: st.image(image_path, caption="Original (Grayscale)", width="stretch")
                with r1c2: st.plotly_chart(_imshow(res_img["log_magnitude"], "Espectro FFT2 (Log-Mag)"), width="stretch")
                with r2c1: st.image(res_img["coded_image"].astype(np.uint8),   caption="Imagen codificada",    width="stretch")
                with r2c2: st.image(res_img["decoded_image"].astype(np.uint8), caption="Imagen reconstruida",  width="stretch")

                with col_met:
                    st.subheader("Métricas 2D")
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>MSE</h4>
                        <p style="font-size:1.5rem;font-weight:700;color:#B45309">{res_img['mse']:.4f}</p>
                        <hr>
                        <h4>PSNR</h4>
                        <p style="font-size:1.5rem;font-weight:700;color:#047857">{res_img['psnr']:.2f} dB</p>
                        <hr>
                        <h4>SSIM</h4>
                        <p style="font-size:1.5rem;font-weight:700;color:#4338CA">{res_img['ssim']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – EPICICLOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_epicycles:
    st.markdown('<div class="section-title">Visualización Dinámica: Epiciclos de Fourier</div>', unsafe_allow_html=True)
    st.write("Cualquier curva cerrada puede dibujarse mediante vectores rotatorios encadenados. La DFT determina la amplitud y velocidad de cada vector.")

    col_ctrl, col_anim = st.columns([1, 3])

    with col_ctrl:
        shape_select = st.selectbox("Figura:", ["Círculo", "Corazón", "Estrella", "Infinito", "Cuadrado"])
        show_circles = st.checkbox("Mostrar órbitas", True)
        pts, max_vecs, shape_notice = build_epicycle_shape(shape_select)

        if shape_notice:
            st.info(shape_notice)

        default_vecs = min(30, max_vecs)
        num_vecs = st.slider("Vectores (frecuencias):", 1, max_vecs, default_vecs)

    with col_anim:
        N_pts = len(pts)
        c       = np.fft.fft(pts)
        c_rot   = c.copy()
        centroid = c[0] / N_pts
        c_rot[0] = 0.0
        idx_rot  = np.argsort(np.abs(c_rot))[::-1]
        freqs    = np.fft.fftfreq(N_pts)

        # Pre-compute tip path
        tip_x, tip_y = [], []
        for m in range(N_pts):
            phi = 2 * np.pi * m / N_pts
            val = centroid + sum(
                c_rot[i] / N_pts * np.exp(1j * freqs[i] * N_pts * phi)
                for i in idx_rot[:num_vecs]
            )
            tip_x.append(val.real)
            tip_y.append(val.imag)

        frames = []
        for m in range(N_pts):
            phi    = 2 * np.pi * m / N_pts
            xs     = [0.0, centroid.real]
            ys     = [0.0, centroid.imag]
            curr   = centroid
            for i in idx_rot[:num_vecs]:
                curr += c_rot[i] / N_pts * np.exp(1j * freqs[i] * N_pts * phi)
                xs.append(curr.real)
                ys.append(curr.imag)

            fd = [
                go.Scatter(x=tip_x[:m+1], y=tip_y[:m+1],
                           mode="lines", name="Trazo",
                           line=dict(color="#EC4899", width=3)),
                go.Scatter(x=[xs[-1]], y=[ys[-1]],
                           mode="markers", showlegend=False,
                           marker=dict(size=8, color="#EC4899")),
                go.Scatter(x=[0, xs[1]], y=[0, ys[1]],
                           mode="lines+markers", showlegend=False,
                           line=dict(color="#94A3B8", width=1.5, dash="dash"),
                           marker=dict(size=4, color="#94A3B8"),
                           hoverinfo="skip"),
            ]

            for j in range(num_vecs):
                fac     = j / max(num_vecs - 1, 1)
                hue     = int(230 + 100 * fac) % 360
                v_w     = max(1.0, 3.5 * (1 - fac))
                v_op    = max(0.35, 1.0 - 0.6 * fac)
                v_col   = f"hsla({hue},75%,40%,{v_op:.2f})"
                c_col   = f"hsla({hue},75%,40%,{v_op * 0.22:.2f})"

                fd.append(go.Scatter(
                    x=[xs[j+1], xs[j+2]], y=[ys[j+1], ys[j+2]],
                    mode="lines+markers",
                    line=dict(color=v_col, width=v_w),
                    marker=dict(size=v_w * 2, color=v_col),
                    showlegend=False, hoverinfo="skip",
                ))

                if show_circles:
                    rad = abs(c_rot[idx_rot[j]] / N_pts)
                    if rad > 0.3:
                        ang_c = np.linspace(0, 2 * np.pi, 60)
                        fd.append(go.Scatter(
                            x=xs[j+1] + rad * np.cos(ang_c),
                            y=ys[j+1] + rad * np.sin(ang_c),
                            mode="lines",
                            line=dict(color=c_col, width=max(0.4, v_w * 0.4)),
                            showlegend=False, hoverinfo="skip",
                        ))

            frames.append(go.Frame(data=fd, name=str(m)))

        fig_ep = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                xaxis=dict(range=[-170, 170], autorange=False,
                           gridcolor=CLR_GRID, linecolor=CLR_BORDER,
                           tickfont=dict(color=CLR_TEXT), zerolinecolor=CLR_GRID),
                yaxis=dict(range=[-170, 170], autorange=False,
                           scaleanchor="x", scaleratio=1,
                           gridcolor=CLR_GRID, linecolor=CLR_BORDER,
                           tickfont=dict(color=CLR_TEXT), zerolinecolor=CLR_GRID),
                height=620,
                plot_bgcolor=CLR_SECONDARY,
                paper_bgcolor=CLR_CARD,
                font=dict(color=CLR_TEXT),
                legend=dict(font=dict(color=CLR_TEXT), bgcolor=CLR_CARD, bordercolor=CLR_BORDER),
                updatemenus=[dict(
                    type="buttons",
                    direction="left",
                    # Style properties belong on the updatemenu container, not on each button
                    bgcolor=CLR_CARD,
                    bordercolor=CLR_BORDER,
                    borderwidth=1,
                    font=dict(color=CLR_TEXT, size=13),
                    buttons=[
                        dict(label="▶  Play", method="animate",
                             args=[None, dict(frame=dict(duration=50, redraw=False), fromcurrent=True)]),
                        dict(label="⏸  Pausa", method="animate",
                             args=[[], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                    ],
                    pad=dict(r=10, t=10), showactive=True,
                    x=0.1, xanchor="right", y=0.0, yanchor="top",
                )],
            ),
            frames=frames,
        )
        st.plotly_chart(fig_ep, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – EXPERIMENTOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_experiments:
    st.markdown('<div class="section-title">Análisis Experimental de Perturbaciones Espectrales</div>', unsafe_allow_html=True)
    st.write("Compara cómo afecta la inyección de ruido de fase a cada banda del espectro.")

    if st.button("Iniciar simulación", type="primary", key="btn_exp"):
        with st.spinner("Ejecutando barrido experimental…"):
            test_sig = text_to_ascii(
                "El procesamiento espectral permite comprender y codificar los datos "
                "multimedia descomponiéndolos en armónicos fundamentales."
            )
            spec_ref = FourierProcessor.fft(test_sig)
            shifted  = FourierProcessor.fft_shift(spec_ref)

            pcts  = np.linspace(0, 100, 11)
            bands = ["low", "mid", "high", "all"]
            rmse_r = {b: [] for b in bands}
            snr_r  = {b: [] for b in bands}

            for pct in pcts:
                for band in bands:
                    mod = Coder.apply_phase_noise(shifted, pct, np.pi, band, key="experiment_key")
                    rec = np.real(FourierProcessor.ifft(FourierProcessor.ifft_shift(mod)))
                    rmse_r[band].append(Metrics.rmse(test_sig, rec))
                    sv = Metrics.snr(test_sig, rec)
                    snr_r[band].append(80.0 if sv == float("inf") else (-80.0 if sv == -float("inf") else sv))

        colors = {"low": "#EF4444", "mid": "#F59E0B", "high": "#10B981", "all": "#6366F1"}
        labels = {
            "low":  "Bajas frecuencias",
            "mid":  "Frecuencias medias",
            "high": "Altas frecuencias",
            "all":  "Espectro completo",
        }

        fr = go.Figure()
        for b in bands:
            fr.add_trace(go.Scatter(x=pcts, y=rmse_r[b], name=labels[b],
                                    line=dict(color=colors[b], width=2.5),
                                    marker=dict(size=6)))
        fr.update_xaxes(title_text="Coeficientes alterados (%)")
        fr.update_yaxes(title_text="RMSE (menor = mejor)")
        st.plotly_chart(light_fig(fr, "RMSE vs Porcentaje de fase modificada"), width="stretch")

        fs2 = go.Figure()
        for b in bands:
            fs2.add_trace(go.Scatter(x=pcts, y=snr_r[b], name=labels[b],
                                     line=dict(color=colors[b], width=2.5),
                                     marker=dict(size=6)))
        fs2.update_xaxes(title_text="Coeficientes alterados (%)")
        fs2.update_yaxes(title_text="SNR (dB) (mayor = mejor)")
        st.plotly_chart(light_fig(fs2, "SNR vs Porcentaje de fase modificada"), width="stretch")

        # FFT complexity benchmark
        st.subheader("Complejidad computacional de la FFT")
        st.write(r"Tiempo de cómputo empírico. La FFT escala como \(O(N\log N)\) frente a \(O(N^2)\) de la DFT ingenua.")
        sizes = [2 ** i for i in range(8, 19)]
        times = []
        for sz in sizes:
            dummy = np.random.rand(sz)
            t0 = time.perf_counter()
            np.fft.fft(dummy)
            times.append((time.perf_counter() - t0) * 1_000)

        ft = go.Figure()
        ft.add_trace(go.Scatter(x=sizes, y=times, name="Tiempo FFT (ms)",
                                line=dict(color="#8B5CF6", width=2.5),
                                marker=dict(size=7, color="#8B5CF6")))
        ft.update_xaxes(title_text="Tamaño de señal N (escala log)", type="log")
        ft.update_yaxes(title_text="Tiempo (ms)")
        st.plotly_chart(light_fig(ft, "Tiempo de cómputo de la FFT"), width="stretch")

        st.success(
            "Simulación completada. Las bajas frecuencias concentran más energía, "
            "por lo que modificarlas produce el mayor error."
        )
