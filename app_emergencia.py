# app_emergencia.py
import streamlit as st
import numpy as np
import pandas as pd
from matplotlib.patches import Patch  # (se usa para consistencia de colores; no imprescindible)
from io import BytesIO, StringIO
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(page_title="Predicción de Emergencia Agrícola AVEFA", layout="wide")

# ====================== Config pesos (fijo a tu repo) ======================
GITHUB_BASE_URL = "https://raw.githubusercontent.com/GUILLE-bit/AVEFA/main"
FNAME_IW   = "IW.npy"
FNAME_BIW  = "bias_IW.npy"
FNAME_LW   = "LW.npy"
FNAME_BOUT = "bias_out.npy"

# ====================== Umbrales EMERREL (EDITAR AQUÍ) ======================
THR_BAJO_MEDIO = 0.020
THR_MEDIO_ALTO = 0.079
assert THR_MEDIO_ALTO > THR_BAJO_MEDIO, "THR_MEDIO_ALTO debe ser mayor que THR_BAJO_MEDIO"

# ====================== Umbrales EMEAC (EDITAR AQUÍ) ======================
EMEAC_MIN_DEN = 1.8
EMEAC_ADJ_DEN = 2.1
EMEAC_MAX_DEN = 2.5
assert 0 < EMEAC_MIN_DEN <= EMEAC_ADJ_DEN <= EMEAC_MAX_DEN, "Asegurá MIN <= ADJ <= MAX"

# ====================== Colores por nivel ======================
COLOR_MAP = {"Bajo": "#2ca02c", "Medio": "#ff7f0e", "Alto": "#d62728"}
COLOR_FALLBACK = "#808080"

# ====================== Utilidades ======================
def _fetch_bytes(url: str, timeout: int = 20) -> bytes:
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
def load_npy_from_fixed(filename: str) -> np.ndarray:
    url = f"{GITHUB_BASE_URL}/{filename}"
    raw = _fetch_bytes(url)
    return np.load(BytesIO(raw), allow_pickle=True)

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
    return niveles.map(COLOR_MAP).fillna(COLOR_FALLBACK).to_numpy()

# ====================== Modelo ======================
class PracticalANNModel:
    def __init__(self, IW, bias_IW, LW, bias_out):
        self.IW = IW
        self.bias_IW = bias_IW
        self.LW = LW
        self.bias_out = float(bias_out)
        # IMPORTANTE: el orden aquí está sincronizado con X_real abajo (Julian_days, TMIN, TMAX, Prec)
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

    def predict(self, X_real, thr_bajo_medio=THR_BAJO_MEDIO, thr_medio_alto=THR_MEDIO_ALTO):
        Xn = self.normalize_input(X_real)
        y = np.array([self._predict_single(x) for x in Xn])
        y = self.desnormalize_output(y)
        ac = np.cumsum(y) / 8.05
        diff = np.diff(ac, prepend=0)

        def clas(v):
            if v < thr_bajo_medio:
                return "Bajo"
            elif v <= thr_medio_alto:
                return "Medio"
            else:
                return "Alto"

        nivel = np.array([clas(v) for v in diff])
        return pd.DataFrame({"EMERREL(0-1)": diff, "Nivel_Emergencia_relativa": nivel})

# ====================== UI ======================
st.title("Predicción de Emergencia Agrícola AVEFA")

# with st.expander("Origen de pesos del modelo (.npy)", expanded=False):
    st.markdown(f"- **Repositorio**: {GITHUB_BASE_URL}")
    st.markdown(f"- Archivos: {FNAME_IW}, {FNAME_BIW}, {FNAME_LW}, {FNAME_BOUT}")

# with st.expander("Parámetros (editables en código)", expanded=False):
    st.markdown(f"**EMERREL**:")
    st.markdown(f"- Bajo→Medio: < {THR_BAJO_MEDIO:.3f}")
    st.markdown(f"- Medio→Alto: ≤ {THR_MEDIO_ALTO:.3f}")
    st.markdown(f"**EMEAC** (denominadores):")
    st.markdown(f"- Mínimo: {EMEAC_MIN_DEN:.2f} · Ajustable: {EMEAC_ADJ_DEN:.2f} · Máximo: {EMEAC_MAX_DEN:.2f}")

st.sidebar.header("Meteo")
csv_pages = st.sidebar.text_input("CSV (Pages)", value="https://GUILLE-bit.github.io/ANN/meteo_daily.csv")
csv_raw   = st.sidebar.text_input("CSV (Raw)",   value="https://raw.githubusercontent.com/GUILLE-bit/ANN/gh-pages/meteo_daily.csv")
fuente_meteo = st.sidebar.radio("Fuente meteo", ["Automático (CSV público)", "Subir Excel meteo"])

if st.sidebar.button("Limpiar caché"):
    st.cache_data.clear()

# --- Cargar pesos desde tu GitHub ---
try:
    IW      = load_npy_from_fixed(FNAME_IW)
    bias_IW = load_npy_from_fixed(FNAME_BIW)
    LW      = load_npy_from_fixed(FNAME_LW)
    bias_out = load_npy_from_fixed(FNAME_BOUT).item()
    st.caption(f"Pesos cargados · H={IW.shape[1]} neuronas ocultas")
except Exception as e:
    st.error(f"No pude cargar los .npy desde GitHub: {e}")
    st.stop()

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
    ups = st.file_uploader("Subí uno o más .xlsx con columnas: Julian_days, TMAX, TMIN, Prec",
                           type=["xlsx"], accept_multiple_files=True, key="meteo_xlsx")
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
        # OJO: orden de entrada debe coincidir con input_min/max del modelo
        X_real = df[["Julian_days", "TMIN", "TMAX", "Prec"]].to_numpy(float)
        fechas = pd.to_datetime(df["Fecha"])

        pred = modelo.predict(X_real, thr_bajo_medio=THR_BAJO_MEDIO, thr_medio_alto=THR_MEDIO_ALTO)
        pred["Fecha"] = fechas
        pred["Julian_days"] = df["Julian_days"]
        pred["EMERREL acumulado"] = pred["EMERREL(0-1)"].cumsum()
        pred["EMERREL_MA5"] = pred["EMERREL(0-1)"].rolling(window=5, min_periods=1).mean()

        pred["EMEAC (0-1) - mínimo"]    = pred["EMERREL acumulado"] / EMEAC_MIN_DEN
        pred["EMEAC (0-1) - máximo"]    = pred["EMERREL acumulado"] / EMEAC_MAX_DEN
        pred["EMEAC (0-1) - ajustable"] = pred["EMERREL acumulado"] / EMEAC_ADJ_DEN
        for col in ["EMEAC (0-1) - mínimo", "EMEAC (0-1) - máximo", "EMEAC (0-1) - ajustable"]:
            pred[col.replace("(0-1)", "(%)")] = (pred[col] * 100).clip(0, 100)

        years = pred["Fecha"].dt.year.unique()
        yr = int(years[0]) if len(years) == 1 else int(st.sidebar.selectbox("Año (reinicio 1/feb → 1/nov)", sorted(years), key=f"year_select_{nombre}"))
        fi = pd.Timestamp(year=yr, month=2, day=1)
        ff = pd.Timestamp(year=yr, month=11, day=1)
        m = (pred["Fecha"] >= fi) & (pred["Fecha"] <= ff)
        pred_vis = pred.loc[m].copy()
        if pred_vis.empty:
            st.warning(f"Sin datos entre {fi.date()} y {ff.date()} para {nombre}.")
            continue

        pred_vis["EMERREL acumulado (reiniciado)"] = pred_vis["EMERREL(0-1)"].cumsum()
        pred_vis["EMEAC (0-1) - mínimo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / EMEAC_MIN_DEN
        pred_vis["EMEAC (0-1) - máximo (rango)"]    = pred_vis["EMERREL acumulado (reiniciado)"] / EMEAC_MAX_DEN
        pred_vis["EMEAC (0-1) - ajustable (rango)"] = pred_vis["EMERREL acumulado (reiniciado)"] / EMEAC_ADJ_DEN
        for col in ["EMEAC (0-1) - mínimo (rango)", "EMEAC (0-1) - máximo (rango)", "EMEAC (0-1) - ajustable (rango)"]:
            pred_vis[col.replace("(0-1)", "(%)")] = (pred_vis[col] * 100).clip(0, 100)

        colores_vis = obtener_colores(pred_vis["Nivel_Emergencia_relativa"])

        # ===================== Gráfico 1: EMERGENCIA RELATIVA DIARIA (Plotly) =====================
        st.subheader("EMERGENCIA RELATIVA DIARIA - BORDENAVE")
        fig_er = go.Figure()

        # Barras por nivel (colores)
        fig_er.add_bar(
            x=pred_vis["Fecha"],
            y=pred_vis["EMERREL(0-1)"],
            marker=dict(color=colores_vis.tolist()),
            customdata=pred_vis["Nivel_Emergencia_relativa"],
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>EMERREL: %{y:.3f}<br>Nivel: %{customdata}<extra></extra>",
            name="EMERREL (0-1)"
        )

        # Línea media móvil
        fig_er.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMERREL_MA5"],
            mode="lines",
            name="Media móvil 5 días",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>MA5: %{y:.3f}<extra></extra>"
        ))

        # Área (celeste claro) bajo la media móvil
        fig_er.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMERREL_MA5"],
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(135, 206, 250, 0.3)",  # LightSkyBlue con 30% opacidad
            name="Área MA5",
            hoverinfo="skip",
            showlegend=False
        ))

        # Líneas de referencia de niveles (Bajo / Medio) + entrada de leyenda para Alto
        low_thr = float(THR_BAJO_MEDIO)
        med_thr = float(THR_MEDIO_ALTO)
        fig_er.add_trace(go.Scatter(
            x=[fi, ff], y=[low_thr, low_thr],
            mode="lines", line=dict(color=COLOR_MAP["Bajo"], dash="dot"),
            name=f"Bajo (≤ {low_thr:.3f})", hoverinfo="skip"
        ))
        fig_er.add_trace(go.Scatter(
            x=[fi, ff], y=[med_thr, med_thr],
            mode="lines", line=dict(color=COLOR_MAP["Medio"], dash="dot"),
            name=f"Medio (≤ {med_thr:.3f})", hoverinfo="skip"
        ))
        fig_er.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=COLOR_MAP["Alto"], dash="dot"),
            name=f"Alto (> {med_thr:.3f})", hoverinfo="skip"
        ))

        fig_er.update_layout(
          # title="EMERGENCIA RELATIVA DIARIA",
            xaxis_title="Fecha",
            yaxis_title="EMERREL (0-1)",
            hovermode="x unified",
            legend_title="Referencias",
            height=650
        )
        fig_er.update_xaxes(range=[fi, ff], dtick="M1", tickformat="%b")
        fig_er.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_er, use_container_width=True, theme="streamlit")

        # ===================== Gráfico 2: EMERGENCIA ACUMULADA DIARIA (Plotly) =====================
        st.subheader("EMERGENCIA ACUMULADA DIARIA - BORDENAVE")
        fig = go.Figure()

        # Banda entre mínimo y máximo
        fig.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMEAC (%) - máximo (rango)"],
            mode="lines",
            line=dict(width=0),
            name="Máximo (reiniciado)",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMEAC (%) - mínimo (rango)"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="Mínimo (reiniciado)",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"
        ))

        # Líneas de umbral (ajustable / min / max)
        fig.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMEAC (%) - ajustable (rango)"],
            mode="lines",
            line=dict(width=2.5),
            name=f"Umbral ajustable (/{EMEAC_ADJ_DEN:.2f})",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Ajustable: %{y:.1f}%<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMEAC (%) - mínimo (rango)"],
            mode="lines",
            line=dict(dash="dash", width=1.5),
            name=f"Umbral mínimo (/{EMEAC_MIN_DEN:.2f})",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Mínimo: %{y:.1f}%<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=pred_vis["Fecha"],
            y=pred_vis["EMEAC (%) - máximo (rango)"],
            mode="lines",
            line=dict(dash="dash", width=1.5),
            name=f"Umbral máximo (/{EMEAC_MAX_DEN:.2f})",
            hovertemplate="Fecha: %{x|%d-%b-%Y}<br>Máximo: %{y:.1f}%<extra></extra>"
        ))

        # Líneas horizontales 25, 50, 75, 90 %
        for nivel in [25, 50, 75, 90]:
            fig.add_hline(y=nivel, line_dash="dash", opacity=0.6, annotation_text=f"{nivel}%")

        fig.update_layout(
          # title="EMERGENCIA ACUMULADA DIARIA",
            xaxis_title="Fecha",
            yaxis_title="EMEAC (%)",
            yaxis=dict(range=[0, 100]),
            hovermode="x unified",
            legend_title="Referencias",
            height=600
        )
        fig.update_xaxes(range=[fi, ff], dtick="M1", tickformat="%b")
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        # ===================== Tabla y descarga =====================
        st.subheader(f"Resultados (1/feb → 1/nov) - {nombre}")
        col_emeac = "EMEAC (%) - ajustable (rango)"
        nivel_icono = {"Bajo": "🟢 Bajo", "Medio": "🟠 Medio", "Alto": "🔴 Alto"}
        tabla = pred_vis[["Fecha","Julian_days","Nivel_Emergencia_relativa",col_emeac]].copy()
        tabla["Nivel_Emergencia_relativa"] = tabla["Nivel_Emergencia_relativa"].map(nivel_icono)
        tabla = tabla.rename(columns={"Nivel_Emergencia_relativa":"Nivel de EMERREL", col_emeac:"EMEAC (%)"})
        st.dataframe(tabla, use_container_width=True)

        csv_buf = StringIO()
        tabla.to_csv(csv_buf, index=False)
        st.download_button(
            f"Descargar resultados (rango) - {nombre}",
            data=csv_buf.getvalue(),
            file_name=f"{nombre}_resultados_rango.csv",
            mime="text/csv"
        )
