# app_emergencia.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO, StringIO
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from pathlib import Path
import os
from urllib.parse import urlparse

st.set_page_config(page_title="Predicción de Emergencia Agrícola AVEFA", layout="wide")

# ====================== Utilidades de E/S segura ======================
def safe_tmp_dir() -> Path:
    p = Path(os.environ.get("TMPDIR", "/tmp"))
    p.mkdir(parents=True, exist_ok=True)
    return p

def tmp_path(filename: str) -> Path:
    return safe_tmp_dir() / filename

# ====================== Descarga y carga remota ======================
def _fetch_bytes(url: str, timeout: int = 20) -> bytes:
    """Descarga bytes desde una URL (GitHub RAW). Lanza error claro si falla."""
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Streamlit ANN Loader)"})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} al descargar: {url}")
    except URLError as e:
        raise RuntimeError(f"URL inválida o no accesible: {url} · Detalle: {e.reason}")
    except Exception as e:
        raise RuntimeError(f"Error al descargar {url}: {e}")

@st.cache_data(ttl=1800)
def load_npy_from_url(url: str) -> np.ndarray:
    raw = _fetch_bytes(url)
    return np.load(BytesIO(raw), allow_pickle=True)

def normalize_base_raw(base: str) -> str:
    """
    Acepta:
      - RAW ya correcto: https://raw.githubusercontent.com/USER/REPO/BRANCH[/subpath]
      - URL de GitHub 'blob': https://github.com/USER/REPO/blob/BRANCH/subpath
    Devuelve siempre RAW base sin barra final.
    """
    base = base.strip().rstrip("/")
    if "raw.githubusercontent.com" in base:
        return base
    if "github.com" in base and "/blob/" in base:
        # convertir /github.com/u/r/blob/branch/path  -> /raw.githubusercontent.com/u/r/branch/path
        parts = base.split("github.com/", 1)[1]  # USER/REPO/blob/BRANCH/...
        user, repo, _blob, branch, *rest = parts.split("/")
        raw = "https://raw.githubusercontent.com/" + "/".join([user, repo, branch] + rest)
        return raw.rstrip("/")
    return base  # dejar tal cual si es otra CDN válida

def join_url(base: str, *segments: str) -> str:
    base = base.rstrip("/")
    segs = [s.strip("/") for s in segments if s and s.strip("/")]
    return "/".join([base] + segs)

def resolve_weight_url(base_raw: str, fname: str) -> tuple[str, str]:
    """
    Devuelve (url_encontrada, estrategia) probando:
      1) base_raw/fname
      2) base_raw/pesos/fname  (si la 1) falla)
    """
    base_raw = normalize_base_raw(base_raw)
    candidates = [
        ("raiz", join_url(base_raw, fname)),
        ("carpeta_pesos", join_url(base_raw, "pesos", fname)),
    ]
    last_err = None
    for tag, url in candidates:
        try:
            _ = _fetch_bytes(url)  # HEAD "manual" con GET corto: si responde, sirve
            return url, tag
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"No se pudo localizar {fname} en la base dada. Último error: {last_err}")

# ====================== CSV público ======================
@st.cache_data(ttl=900)
def load_public_csv(csv_pages: str, csv_raw: str):
    last_err = None
    for url in (csv_pages, csv_raw):
        try:
            df = pd.read_csv(url, parse_dates=["Fecha"])
            req = {"Fecha", "Julian_days", "TMAX", "TMIN", "Prec"}
            if not req.issubset(df.columns):
                raise ValueError(f"Faltan columnas: {sorted(list(req - set(df.columns)))}")
            return df.sort_values("Fecha").reset_index(drop=True), url
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No se pudo leer CSV público. Último error: {last_err}")

def validar_columnas_meteo(df: pd.DataFrame):
    req = {"Julian_days", "TMAX", "TMIN", "Prec"}
    faltan = req - set(df.columns)
    return (len(faltan) == 0, "" if not faltan else f"Faltan columnas: {', '.join(sorted(faltan))}")

def obtener_colores(niveles: pd.Series):
    return niveles.map({"Bajo": "green", "Medio": "orange", "Alto": "red"}).fillna("gray")

# ====================== Modelo ======================
class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out):
        self.IW = IW              # (4,H)
        self.bias_IW = bias_IW    # (H,)
        self.LW = LW              # (1,H)
        self.bias_out = float(bias_out)  # escalar
        # Orden de entrada esperado: [Julian_days, TMIN, TMAX, Prec]
        self.input_min = np.array([1, -7, 0, 0], dtype=float)
        self.input_max = np.array([300, 25.5, 41, 84], dtype=float)

    def tansig(self, x): return np.tanh(x)
    def normalize_input(self, X):
        return 2 * (X - self.input_min) / (self.input_max - self.input_min) - 1
    def desnormalize_output(self, y, ymin=-1, ymax=1): return (y - ymin) / (ymax - ymin)

    def _predict_single(self, x_norm):
        z1 = self.IW.T @ x_norm + self.bias_IW
        a1 = self.tansig(z1)
        z2 = self.LW @ a1 + self.bias_out
        return self.tansig(z2).item()

    def predict(self, X_real):
        Xn = self.normalize_input(X_real)
        y = np.array([self._predict_single(x) for x in Xn])          # [-1..1]
        y = self.desnormalize_output(y)                               # [0..1]
        ac = np.cumsum(y) / 8.05                                      # normalización de acumulado anual
        diff = np.diff(ac, prepend=0)
        def clas(v): 
            return "Bajo" if v < 0.2 else ("Medio" if v < 0.4 else "Alto")
        nivel = np.array([clas(v) for v in diff])
        return pd.DataFrame({"EMERREL(0-1)": diff, "Nivel_Emergencia_relativa": nivel})

# ====================== UI ======================
st.title("Predicción de Emergencia Agrícola AVEFA")

st.sidebar.header("Pesos .npy (GitHub)")
base_raw_in = st.sidebar.text_input(
    "Base (puede ser RAW o URL con /blob/)",
    value="https://raw.githubusercontent.com/USUARIO/REPO/main",  # SIN carpeta 'pesos' por tu caso actual
    help="Ejemplos válidos: RAW (raw.githubusercontent.com/USER/REPO/BRANCH[/carpeta]) o github.com/USER/REPO/blob/BRANCH/carpeta"
)

fname_IW   = st.sidebar.text_input("Archivo IW.npy", value="IW.npy")
fname_bIW  = st.sidebar.text_input("Archivo bias_IW.npy", value="bias_IW.npy")
fname_LW   = st.sidebar.text_input("Archivo LW.npy", value="LW.npy")
fname_bout = st.sidebar.text_input("Archivo bias_out.npy", value="bias_out.npy")

colA, colB = st.sidebar.columns(2)
with colA:
    if st.button("Probar y resolver URLs"):
        try:
            url_IW, tag_IW     = resolve_weight_url(base_raw_in, fname_IW)
            url_bIW, tag_bIW   = resolve_weight_url(base_raw_in, fname_bIW)
            url_LW, tag_LW     = resolve_weight_url(base_raw_in, fname_LW)
            url_bout, tag_bout = resolve_weight_url(base_raw_in, fname_bout)
            st.success("Encontré rutas válidas para los 4 archivos ✅")
            st.caption(f"IW: {url_IW}  ({tag_IW})")
            st.caption(f"bias_IW: {url_bIW}  ({tag_bIW})")
            st.caption(f"LW: {url_LW}  ({tag_LW})")
            st.caption(f"bias_out: {url_bout}  ({tag_bout})")
        except Exception as e:
            st.error(str(e))

st.sidebar.header("Meteo")
csv_pages = st.sidebar.text_input("CSV (Pages)", value="https://GUILLE-bit.github.io/ANN/meteo_daily.csv")
csv_raw   = st.sidebar.text_input("CSV (Raw)",   value="https://raw.githubusercontent.com/GUILLE-bit/ANN/gh-pages/meteo_daily.csv")
fuente_meteo = st.sidebar.radio("Fuente meteo", ["Automático (CSV público)", "Subir Excel meteo"])

st.sidebar.header("Configuración")
umbral_usuario = st.sidebar.number_input("Umbral de EMEAC para 100%", min_value=1.2, max_value=3.0, value=2.70, step=0.01, format="%.2f")
if st.sidebar.button("Limpiar caché"):
    st.cache_data.clear()

# --- Resolver y cargar pesos .npy desde GitHub (con autodetección raiz/pesos) ---
try:
    url_IW, tag_IW     = resolve_weight_url(base_raw_in, fname_IW)
    url_bIW, tag_bIW   = resolve_weight_url(base_raw_in, fname_bIW)
    url_LW, tag_LW     = resolve_weight_url(base_raw_in, fname_LW)
    url_bout, tag_bout = resolve_weight_url(base_raw_in, fname_bout)

    IW      = load_npy_from_url(url_IW)
    bias_IW = load_npy_from_url(url_bIW)
    LW      = load_npy_from_url(url_LW)
    bias_out = load_npy_from_url(url_bout).item()

    st.caption(f"Pesos cargados · H={IW.shape[1]} neuronas ocultas")
    st.caption(f"(Origen detectado: {tag_IW}, {tag_bIW}, {tag_LW}, {tag_bout})")
except Exception as e:
    st.error(f"No pude cargar los .npy desde GitHub: {e}")
    st.stop()

# Validaciones básicas de dimensiones
try:
    assert IW.shape[0] == 4, "IW debe ser de tamaño 4×H"
    assert bias_IW.shape[0] == IW.shape[1], "bias_IW debe tener tamaño H"
    assert LW.shape[1] == IW.shape[1], "LW debe tener tamaño 1×H"
except AssertionError as e:
    st.error(f"Dimensiones de pesos inconsistentes: {e}")
    st.stop()

modelo = PracticalANNModel(IW, bias_IW, LW, bias_out)

# --- Cargar meteo ---
dfs = []
if fuente_meteo == "Automático (CSV público)":
    try:
        df_auto, url_usada = load_public_csv(csv_pages, csv_raw)
        dfs.append(("MeteoBahia_CSV", df_auto))
        st.caption(f"CSV usado: {url_usada} · {df_auto['Fecha'].min().date()} → {df_auto['Fecha'].max().date()} · {len(df_auto)} días")
    except Exception as e:
        st.error(f"No se pudo leer el CSV público: {e}")
else:
    ups = st.file_uploader("Subí uno o más .xlsx con columnas: Julian_days, TMAX, TMIN, Prec", type=["xlsx"], accept_multiple_files=True, key="meteo_xlsx")
    if ups:
        for f in ups:
            df_up = pd.read_excel(f)
            ok, msg = validar_columnas_meteo(df_up)
            if not ok:
                st.warning(f"{f.name}: {msg}")
                continue
            if "Fecha" not in df_up.columns:
                year = pd.Timestamp.now().year
                df_up["Fecha"] = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(df_up["Julian_days"] - 1, unit="D")
            dfs.append((Path(f.name).stem, df_up))
    else:
        st.info("Esperando archivo(s) meteo...")

# ====================== Procesamiento y visualización ======================
if dfs:
    for nombre, df in dfs:
        ok, msg = validar_columnas_meteo(df)
        if not ok:
            st.warning(f"{nombre}: {msg}")
            continue

        df = df.sort_values("Julian_days").reset_index(drop=True)
        # Orden de variables esperado por los pesos: [Julian_days, TMIN, TMAX, Prec]
        X_real = df[["Julian_days", "TMIN", "TMAX", "Prec"]].to_numpy(float)
        fechas = pd.to_datetime(df["Fecha"])

        pred = modelo.predict(X_real)
        pred["Fecha"] = fechas
        pred["Julian_days"] = df["Julian_days"]
        pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()
        pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()

        # EMEAC (%) acumulado anual y rango
        pred["EMEAC (0-1) - mínimo"] = pred["EMERREL acumulado"] / 1.2
        pred["EMEAC (0-1) - máximo"] = pred["EMERREL acumulado"] / 3.0
        pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / umbral_usuario
        for col in ["EMEAC (0-1) - mínimo", "EMEAC (0-1) - máximo", "EMEAC (0-1) - ajustable"]:
            pred[col.replace("(0-1)", "(%)")] = (pred[col] * 100).clip(0, 100)

        # Rango 1/feb → 1/nov (reinicio)
        years = pred["Fecha"].dt.year.unique()
        yr = int(years[0]) if len(years) == 1 else int(st.sidebar.selectbox("Año (reinicio 1/feb → 1/nov)", sorted(years)))
        fi = pd.Timestamp(year=yr, month=2, day=1)
        ff = pd.Timestamp(year=yr, month=11, day=1)
        m = (pred["Fecha"] >= fi) & (pred["Fecha"] <= ff)
        pred_vis = pred.loc[m].copy()
        if pred_vis.empty:
            st.warning(f"Sin datos entre {fi.date()} y {ff.date()} para {nombre}.")
            continue

        pred_vis["EMERREL acumulado (reiniciado)"] = pred_vis["EMERREL(0-1)"].cumsum()
        pred_vis["EMEAC (0-1) - mínimo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / 1.2
        pred_vis["EMEAC (0-1) - máximo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / 3.0
        pred_vis["EMEAC (0-1) - ajustable (rango)"] = pred_vis["EMERREL acumulado (reiniciado)"] / umbral_usuario
        for col in ["EMEAC (0-1) - mínimo (rango)", "EMEAC (0-1) - máximo (rango)", "EMEAC (0-1) - ajustable (rango)"]:
            pred_vis[col.replace("(0-1)", "(%)")] = (pred_vis[col] * 100).clip(0, 100)

        colores_vis = obtener_colores(pred_vis["Nivel_Emergencia_relativa"])

        # --- Gráfico EMERREL ---
        st.subheader(f"EMERREL (0-1) · {nombre} · {fi.date()} → {ff.date()} (reinicio 1/feb)")
        fig_er, ax_er = plt.subplots(figsize=(14, 5), dpi=150)
        ax_er.bar(pred_vis["Fecha"], pred_vis["EMERREL(0-1)"], color=colores_vis)
        ax_er.plot(pred_vis["Fecha"], pred_vis["EMERREL_MA5"], linewidth=2.2, label="Media móvil 5 días")
        ax_er.legend(loc="upper right"); ax_er.grid(True, linestyle="--", alpha=0.5)
        ax_er.set_xlabel("Fecha"); ax_er.set_ylabel("EMERREL (0-1)")
        ax_er.set_xlim(fi, ff); ax_er.xaxis.set_major_locator(mdates.MonthLocator()); ax_er.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        st.pyplot(fig_er)

        # --- Gráfico EMEAC ---
        st.subheader(f"EMEAC (%) · {nombre} · {fi.date()} → {ff.date()} (reinicio 1/feb)")
        fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
        ax.fill_between(pred_vis["Fecha"], pred_vis["EMEAC (%) - mínimo (rango)"], pred_vis["EMEAC (%) - máximo (rango)"], alpha=0.35, label="Rango min–max")
        ax.plot(pred_vis["Fecha"], pred_vis["EMEAC (%) - ajustable (rango)"], linewidth=2.5, label="Umbral ajustable")
        ax.plot(pred_vis["Fecha"], pred_vis["EMEAC (%) - mínimo (rango)"], linestyle="--", linewidth=1.5, label="Umbral mínimo")
        ax.plot(pred_vis["Fecha"], pred_vis["EMEAC (%) - máximo (rango)"], linestyle="--", linewidth=1.5, label="Umbral máximo")
        ax.grid(True, linestyle="--", alpha=0.5); ax.set_ylim(0,100); ax.set_xlim(fi, ff)
        ax.set_xlabel("Fecha"); ax.set_ylabel("EMEAC (%)")
        ax.xaxis.set_major_locator(mdates.MonthLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.legend(loc="lower right"); st.pyplot(fig)

        # --- Tabla y descarga (en memoria) ---
        st.subheader(f"Resultados (1/feb → 1/nov) - {nombre}")
        col_emeac = "EMEAC (%) - ajustable (rango)"
        tabla = pred_vis[["Fecha","Julian_days","Nivel_Emergencia_relativa",col_emeac]].rename(
            columns={"Nivel_Emergencia_relativa":"Nivel de EMERREL", col_emeac:"EMEAC (%)"}
        )
        st.dataframe(tabla, use_container_width=True)

        csv_buf = StringIO()
        tabla.to_csv(csv_buf, index=False)
        st.download_button(
            f"Descargar resultados (rango) - {nombre}",
            data=csv_buf.getvalue(),
            file_name=f"{nombre}_resultados_rango.csv",
            mime="text/csv"
        )


