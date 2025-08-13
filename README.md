# PREDWEEM · Carga de pesos .npy desde GitHub
Fecha: 2025-08-13 15:15

Este paquete incluye un `app_emergencia.py` que **solo** carga los pesos del modelo desde **URLs RAW de GitHub**.
- Completá en la barra lateral la **Base RAW** (p. ej., `https://raw.githubusercontent.com/USER/REPO/BRANCH/weights`)
- Especificá los nombres de archivo: `IW.npy`, `bias_IW.npy`, `LW.npy`, `bias_out.npy`.

## Ejecución
```bash
pip install streamlit pandas numpy matplotlib
streamlit run app_emergencia.py
```

## Orden y normalización
- Entradas esperadas por los pesos: `[Julian_days, TMIN, TMAX, Prec]`
- Rangos: min = `[1, -7, 0, 0]`, max = `[300, 25.5, 41, 84]`

## Meteo
- Podés usar el CSV público (valores por defecto) o subir tu propio Excel con columnas `Julian_days, TMAX, TMIN, Prec`.
