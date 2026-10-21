App · PY
"""
Radar de Autonomía Tuali · Arca Continental · Equipo 5
Niveles de autonomía digital con PCA + GMM
 
Cómo correrlo (terminal de VS Code, dentro de la carpeta del proyecto):
    python -m streamlit run app.py
 
La base de datos (Excel o CSV) se sube desde la barra lateral.
"""
 
import re
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
 
# ──────────────────────────────────────────────────────────────
# Configuración general
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Radar de Autonomía Tuali", page_icon="🏪", layout="wide")
 
 
# Las 7 variables con las que se entrenó el modelo (mismo orden que el notebook)
VARS = [
    "PROMOTIONRATE",
    "CODESEARCHRATE",
    "EVENTCOUNT",
    "DISTINCTEVENTTYPES",
    "DISTINCT_PRODUCTS",
    "AVGSESSIONDURATIONSECONDS",
    "FAVORITERATE",
]
 
NOMBRES = {
    "PROMOTIONRATE": "Uso de promociones",
    "CODESEARCHRATE": "Búsqueda por código",
    "EVENTCOUNT": "Eventos en la app",
    "DISTINCTEVENTTYPES": "Tipos de eventos",
    "DISTINCT_PRODUCTS": "Productos distintos",
    "AVGSESSIONDURATIONSECONDS": "Duración de sesión (seg)",
    "FAVORITERATE": "Uso de favoritos",
    "COUPLED": "Pedidos acoplados a visita",
}
 
# Qué hacer en la visita cuando esa variable es la mayor brecha contra el autónomo promedio
PROTOCOLOS = {
    "PROMOTIONRATE": "Mostrar la sección de promociones y cerrar un pedido con promo",
    "CODESEARCHRATE": "Capacitación en búsqueda por marca y catálogo (dejar el código SKU)",
    "EVENTCOUNT": "Activar notificaciones y acompañar un pedido completo en la app",
    "DISTINCTEVENTTYPES": "Enfocar el uso de la app en el flujo de pedido",
    "DISTINCT_PRODUCTS": "Recorrer el catálogo para ampliar el surtido",
    "AVGSESSIONDURATIONSECONDS": "Acompañar una sesión de pedido de principio a fin",
    "FAVORITERATE": "Armar su lista de favoritos para repetir pedido en un clic",
}
 
VARS_ENTERAS = {"EVENTCOUNT", "DISTINCTEVENTTYPES", "DISTINCT_PRODUCTS"}
VARS_TASA = {"PROMOTIONRATE", "CODESEARCHRATE", "FAVORITERATE"}
 
ORDEN = ["Autónomo", "Semi autónomo", "Dependiente"]
COLORES = {"Autónomo": "#3f9a63", "Semi autónomo": "#e0a32e", "Dependiente": "#d0453f"}
CIRCULO = {"Autónomo": "🟢", "Semi autónomo": "🟡", "Dependiente": "🔴"}
PLURAL = {"Autónomo": "Autónomos", "Semi autónomo": "Semi autónomos", "Dependiente": "Dependientes"}
ROJO_ARCA = "#c62828"
TINTA = "#2b2118"
 
# Nivel de una ruta según su % de dependientes
NIVEL_RUTA = {"Dependiente": "🔴 Dependiente", "Semi autónomo": "🟡 Semi autónoma", "Autónomo": "🟢 Autónoma"}
 
DIMENSIONES = {
    "Estado": "STATE",
    "CEDIS": "CEDISID",
    "Ruta": "ROUTEID",
    "Subcanal": "SUBCHANNEL",
    "Tamaño de cliente": "CUSTOMERSIZE",
    "Tipo de negocio": "BUSINESSTYPE",
}
 
 
# ──────────────────────────────────────────────────────────────
# Estilo
# ──────────────────────────────────────────────────────────────
def html(codigo: str, donde=st):
    """Escribe HTML en una sola línea (evita que Streamlit lo tome como bloque de código)."""
    donde.markdown(re.sub(r"\s*\n\s*", " ", codigo), unsafe_allow_html=True)
 
 
html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
h1, h2, h3, .stApp h1, .stApp h2, .stApp h3 {{font-family: 'Fraunces', Georgia, serif !important; color: {TINTA};}}
.banner {{background: radial-gradient(circle at 85% 20%, #e5483f 0%, {ROJO_ARCA} 35%, #8e1b1b 100%);
          padding: 26px 30px; border-radius: 18px; color: white; margin-bottom: 18px;
          display: flex; align-items: center; justify-content: space-between; gap: 20px;
          box-shadow: 0 10px 30px rgba(142,27,27,.25);}}
.banner .eyebrow {{font-family: 'Plus Jakarta Sans', sans-serif; font-size: .75rem; letter-spacing: .16em; opacity: .9;}}
.banner h1 {{font-family: 'Fraunces', serif !important; font-weight: 700; font-size: 2.3rem; margin: 4px 0 8px;
             color: white !important; padding: 0;}}
.banner p {{font-family: 'Plus Jakarta Sans', sans-serif; margin: 0; font-size: 1.02rem;}}
.banner .tiendas {{display: flex; gap: 10px; align-items: flex-end;}}
.card {{background: #ffffff; border-radius: 16px; padding: 16px 18px; min-height: 132px; margin-bottom: 10px;
        box-shadow: 0 2px 10px rgba(43,33,24,.07); color: {TINTA}; font-family: 'Plus Jakarta Sans', sans-serif;
        display: flex; justify-content: space-between; gap: 10px;}}
.card-t {{font-size: .85rem; opacity: .7; font-weight: 600;}}
.card-v {{font-family: 'Fraunces', serif; font-size: 2.1rem; font-weight: 700; margin: 4px 0; line-height: 1.1;}}
.card-s {{font-size: .8rem; opacity: .65;}}
.waffle {{display: grid; grid-template-columns: repeat(20, 1fr); gap: 5px; margin: 8px 0 10px;}}
.waffle svg {{width: 100%; height: auto;}}
.leyenda {{display: flex; gap: 18px; flex-wrap: wrap; font-family: 'Plus Jakarta Sans', sans-serif; font-size: .9rem; color: {TINTA};}}
.leyenda span {{display: inline-flex; align-items: center; gap: 6px;}}
.camino {{display: flex; align-items: stretch; justify-content: space-between; gap: 10px; flex-wrap: wrap;}}
.paso {{flex: 1 1 200px; background: #ffffff; border-radius: 16px; padding: 18px; text-align: center;
        box-shadow: 0 2px 10px rgba(43,33,24,.07); color: {TINTA}; font-family: 'Plus Jakarta Sans', sans-serif;}}
.paso-n {{font-family: 'Fraunces', serif; font-size: 2.2rem; font-weight: 700; line-height: 1;}}
.paso-t {{font-weight: 700; margin: 4px 0;}}
.paso-s {{font-size: .82rem; opacity: .7;}}
.flecha {{align-self: center; font-size: 2rem; color: #b9a99a;}}
.accion {{border-left: 5px solid {ROJO_ARCA}; background: #ffffff; border-radius: 10px; color: {TINTA};
          padding: 12px 16px; margin-bottom: 10px; box-shadow: 0 2px 8px rgba(43,33,24,.06);
          font-family: 'Plus Jakarta Sans', sans-serif;}}
.accion b {{display: block;}}
.accion span {{font-size: .82rem; opacity: .7;}}
.badge {{color: white; padding: 4px 12px; border-radius: 999px; font-size: .85rem; font-weight: 600;
         font-family: 'Plus Jakarta Sans', sans-serif;}}
.stTabs [data-baseweb="tab-list"] {{gap: 8px; flex-wrap: wrap;}}
.stTabs [data-baseweb="tab"] {{background: #f1e9df; border-radius: 999px; padding: 8px 18px; height: auto;}}
.stTabs [aria-selected="true"] {{background: {ROJO_ARCA}; color: white !important;}}
.stTabs [aria-selected="true"] p {{color: white !important;}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{display: none;}}
</style>
""")
 
 
def tienda_svg(color: str, tam: int = 40, cuerpo: str = "#ffffff") -> str:
    """Ícono de tiendita (toldo, letrero, ventanas y puerta) del color indicado."""
    festones = "".join(f"<circle cx='{x}' cy='24' r='4' fill='{color}'/>" for x in range(8, 60, 8))
    rayas = "".join(f"<rect x='{x}' y='12' width='4' height='12' fill='white' opacity='.35'/>" for x in range(14, 54, 8))
    return (
        f"<svg viewBox='0 0 64 64' width='{tam}' height='{tam}' xmlns='http://www.w3.org/2000/svg'>"
        f"<rect x='18' y='3' width='28' height='8' rx='2' fill='{color}'/>"
        f"<rect x='10' y='24' width='44' height='35' rx='2' fill='{cuerpo}' stroke='{color}' stroke-width='3'/>"
        f"<path d='M8 11 H56 L60 24 H4 Z' fill='{color}'/>{rayas}{festones}"
        f"<rect x='15' y='34' width='11' height='10' rx='1.5' fill='{color}' opacity='.35'/>"
        f"<rect x='38' y='34' width='11' height='10' rx='1.5' fill='{color}' opacity='.35'/>"
        f"<rect x='27' y='39' width='10' height='20' rx='1.5' fill='{color}'/></svg>"
    )
 
 
def tarjeta(col, titulo, valor, sub="", color=ROJO_ARCA, icono=None):
    icono_html = f"<div>{icono}</div>" if icono else ""
    html(
        f"<div class='card' style='border-top:5px solid {color}'><div><div class='card-t'>{titulo}</div>"
        f"<div class='card-v'>{valor}</div><div class='card-s'>{sub}</div></div>{icono_html}</div>",
        col,
    )
 
 
def medidor(valor: float, cortes: tuple, titulo: str = "", alto: int = 230) -> go.Figure:
    """Velocímetro del IAD (0–10) con franjas de color por nivel."""
    fig = go.Figure(go.Indicator(
        mode="gauge", value=valor,
        title={"text": titulo, "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 10], "tickwidth": 1},
            "bar": {"color": TINTA, "thickness": 0.22},
            "bgcolor": "white",
            "steps": [
                {"range": [0, cortes[0]], "color": COLORES["Dependiente"]},
                {"range": [cortes[0], cortes[1]], "color": COLORES["Semi autónomo"]},
                {"range": [cortes[1], 10], "color": COLORES["Autónomo"]},
            ],
        },
    ))
    # Número centrado debajo de la aguja
    fig.add_annotation(x=0.5, y=0.02, xref="paper", yref="paper", showarrow=False, text=f"<b>{valor:.1f}</b>",
                       font=dict(size=44 if alto >= 260 else 34, family="Fraunces, serif", color=TINTA))
    fig.update_layout(height=alto, margin=dict(t=40, b=10, l=30, r=30))
    return fig
 
 
def restablecer(valores: dict):
    """Regresa los controles a su valor inicial (se usa en los botones ↺ Restablecer)."""
    for clave, valor in valores.items():
        st.session_state[clave] = valor
 
 
def iniciar(valores: dict):
    """Pone el valor inicial de cada control solo la primera vez."""
    for clave, valor in valores.items():
        st.session_state.setdefault(clave, valor)
 
 
# ──────────────────────────────────────────────────────────────
# Datos y modelo
# ──────────────────────────────────────────────────────────────
# Nombres aceptados para las columnas de coordenadas del cliente
ALIAS_LAT = ["LATITUDE", "LATITUD", "LAT", "CUSTOMERLATITUDE", "LATITUD_CLIENTE"]
ALIAS_LON = ["LONGITUDE", "LONGITUD", "LON", "LNG", "LONG", "CUSTOMERLONGITUDE", "LONGITUD_CLIENTE"]
 
# Columnas opcionales: si la base no las trae, se rellenan para que el dashboard funcione igual
OPCIONALES = {
    "SUBCHANNEL": "Sin dato", "ROUTEID": "Sin ruta", "CEDISID": "Sin CEDIS", "STATE": "Sin estado",
    "CITY": "Sin dato", "JOINDATE": "", "CUSTOMERSIZE": "Sin dato", "BUSINESSTYPE": "Sin dato",
}
 
 
@st.cache_data
def leer_archivo(contenido: bytes, nombre: str) -> pd.DataFrame:
    """Lee cualquier Excel (.xlsx) o CSV, sin importar el nombre del archivo."""
    from io import BytesIO
    if nombre.lower().endswith(".csv"):
        for codificacion in ("utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(BytesIO(contenido), sep=None, engine="python", encoding=codificacion)
            except UnicodeDecodeError:
                continue
    return pd.read_excel(BytesIO(contenido))
 
 
def preparar_datos(crudo: pd.DataFrame):
    """Normaliza nombres de columnas y rellena las opcionales. Regresa (df, columnas_faltantes)."""
    df = crudo.copy()
    df.columns = [str(c).strip().upper().replace(" ", "_") for c in df.columns]
    faltan = [v for v in VARS if v not in df.columns]
    if faltan:
        return None, faltan
    for v in VARS:
        df[v] = pd.to_numeric(df[v], errors="coerce")
    df = df.dropna(subset=VARS).reset_index(drop=True)
    if "CUSTOMERID" not in df.columns:
        df["CUSTOMERID"] = np.arange(1, len(df) + 1)
    if "BUSINESSNAME" not in df.columns:
        df["BUSINESSNAME"] = "Cliente " + df["CUSTOMERID"].astype(str)
    if "COUPLED" not in df.columns:
        df["COUPLED"] = np.nan
    df["COUPLED"] = pd.to_numeric(df["COUPLED"], errors="coerce")
    for col, valor in OPCIONALES.items():
        if col not in df.columns:
            df[col] = valor
        df[col] = df[col].fillna(valor).astype(str)
    df["BUSINESSNAME"] = df["BUSINESSNAME"].astype(str)
    df["CLIENTE"] = df["CUSTOMERID"].astype(str) + " · " + df["BUSINESSNAME"]
    # Coordenadas reales del cliente (si vienen en la base, con cualquiera de los nombres aceptados)
    col_lat = next((c for c in ALIAS_LAT if c in df.columns), None)
    col_lon = next((c for c in ALIAS_LON if c in df.columns), None)
    if col_lat and col_lon:
        df["LAT_REAL"] = pd.to_numeric(df[col_lat].astype(str).str.replace(",", "."), errors="coerce")
        df["LON_REAL"] = pd.to_numeric(df[col_lon].astype(str).str.replace(",", "."), errors="coerce")
        fuera = ~df["LAT_REAL"].between(-90, 90) | ~df["LON_REAL"].between(-180, 180)
        df.loc[fuera, ["LAT_REAL", "LON_REAL"]] = np.nan
    else:
        df["LAT_REAL"] = np.nan
        df["LON_REAL"] = np.nan
    return df, []
 
 
@st.cache_resource
def entrenar_modelo(df: pd.DataFrame):
    """Pipeline del notebook: StandardScaler → PCA(3) → GMM(3), entrenado con la base que se subió."""
    X_raw = df[VARS].values
    scaler = StandardScaler().fit(X_raw)
    Z = scaler.transform(X_raw)
    pca = PCA(n_components=3, random_state=42).fit(Z)
    X = pca.transform(Z)
    gmm = GaussianMixture(
        n_components=3, covariance_type="full", reg_covar=0.01, n_init=10, random_state=42
    ).fit(X)
    clusters = gmm.predict(X)
 
    # Nombre de cada grupo según su comportamiento: más promociones, favoritos, productos, sesión y
    # eventos, y menos búsqueda por código = más autónomo. (Coincide 100% con el notebook.)
    zd = pd.DataFrame(Z, columns=VARS)
    puntaje = (zd[["PROMOTIONRATE", "FAVORITERATE", "DISTINCT_PRODUCTS", "AVGSESSIONDURATIONSECONDS",
                   "EVENTCOUNT"]].mean(axis=1) - zd["CODESEARCHRATE"])
    orden = puntaje.groupby(clusters).mean().sort_values().index
    mapa = {orden[0]: "Dependiente", orden[1]: "Semi autónomo", orden[2]: "Autónomo"}
 
    return {
        "scaler": scaler,
        "pca": pca,
        "gmm": gmm,
        "mapa": mapa,
        "segmentos": pd.Series(clusters).map(mapa).values,
        "prob_max": gmm.predict_proba(X).max(axis=1),
        "componentes": X,
        "var_explicada": pca.explained_variance_ratio_,
        # Peso de cada variable en el IAD = carga al cuadrado en el primer componente (suman 1)
        "pesos": pd.Series(pca.components_[0] ** 2, index=VARS),
    }
 
 
def predecir(modelo, valores: dict) -> pd.Series:
    """Probabilidad de pertenecer a cada segmento para un cliente (dict variable → valor)."""
    x = np.array([[valores[v] for v in VARS]])
    z = modelo["pca"].transform(modelo["scaler"].transform(x))
    proba = modelo["gmm"].predict_proba(z)[0]
    s = pd.Series({modelo["mapa"][i]: p for i, p in enumerate(proba)})
    return s.reindex(ORDEN)
 
 
@st.cache_data
def calcular_indices(df: pd.DataFrame, _pesos: pd.Series) -> pd.DataFrame:
    """Puntaje 0–10 por variable (percentil, orientado para que más alto = más autónomo)
    e IAD (Índice de Autonomía Digital) = promedio ponderado con los pesos del PCA."""
    df = df.copy()
    medias = df.groupby("SEGMENT")[VARS].mean()
    for v in VARS:
        signo = 1 if medias.loc["Autónomo", v] >= medias.loc["Dependiente", v] else -1
        df[f"P_{v}"] = (signo * df[v]).rank(pct=True) * 10
    df["IAD"] = sum(df[f"P_{v}"] * _pesos[v] for v in VARS)
    return df
 
 
def variables_invertidas(df: pd.DataFrame) -> set:
    medias = df.groupby("SEGMENT")[VARS].mean()
    return {v for v in VARS if medias.loc["Autónomo", v] < medias.loc["Dependiente", v]}
 
 
def normalizar(df_vals: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Escala min–max (0 = mínimo de la base, 1 = máximo) para comparar variables en el radar."""
    mn, mx = base[VARS].min(), base[VARS].max()
    return ((df_vals[VARS] - mn) / (mx - mn)).clip(0, 1)
 
 
def radar(perfiles: dict, titulo: str = "", rango=(0, 1), etiquetas=None) -> go.Figure:
    """perfiles: nombre → Serie indexada por VARS."""
    etiquetas = etiquetas or [NOMBRES[v] for v in VARS]
    fig = go.Figure()
    for nombre, serie in perfiles.items():
        valores = serie[VARS].tolist()
        base = nombre.replace(" promedio", "")
        color = COLORES.get(base, TINTA)
        es_segmento = base in COLORES
        fig.add_trace(go.Scatterpolar(
            r=valores + valores[:1], theta=etiquetas + etiquetas[:1], name=nombre,
            fill="toself" if es_segmento else "none", opacity=0.55 if es_segmento else 1,
            line=dict(color=color, width=2 if es_segmento else 3, dash=None if es_segmento else "dash"),
            hovertemplate="%{theta}: %{r:.2f}<extra>" + nombre + "</extra>",
        ))
    paso = (rango[1] - rango[0]) / 4
    fig.update_layout(
        title=titulo,
        polar=dict(radialaxis=dict(visible=True, range=list(rango), tickvals=[rango[0] + paso * i for i in range(1, 5)])),
        legend=dict(orientation="h", y=-0.12), margin=dict(t=50, b=40, l=60, r=60), height=480,
    )
    return fig
 
 
def cien_tiendas(conteo: pd.Series) -> list:
    """Reparte 100 íconos entre niveles en proporción a los clientes (método del mayor residuo)."""
    exacto = conteo / conteo.sum() * 100
    n = exacto.astype(int)
    faltan = 100 - n.sum()
    for s in (exacto - n).sort_values(ascending=False).index[:faltan]:
        n[s] += 1
    return [s for s in ORDEN for _ in range(n[s])]
 
 
# ──────────────────────────────────────────────────────────────
# Mapa y rutas
# ──────────────────────────────────────────────────────────────
# Ciudades de referencia por estado (la base no trae coordenadas)
ANCLAS = {
    "Baja California": [(32.51, -117.04), (32.62, -115.45), (31.87, -116.60), (32.45, -116.90)],
    "Coahuila": [(25.42, -101.00), (25.54, -103.41), (26.90, -101.42), (28.70, -100.52)],
    "Jalisco": [(20.67, -103.35), (20.65, -105.22), (21.36, -101.93), (19.70, -103.46)],
}
 
 
@st.cache_data
def agregar_coordenadas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega LAT/LON por cliente y CEDIS_LAT/CEDIS_LON.
    - Si la base trae latitud y longitud, se usan las coordenadas reales del cliente y el CEDIS se ubica
      en el centro de sus clientes.
    - Si no, se generan posiciones aproximadas: cada CEDIS en una ciudad de su estado y cada cliente
      alrededor del centro de su ruta."""
    df = df.copy()
 
    # Posiciones aproximadas (respaldo para clientes sin coordenadas)
    pos_cedis = {}
    cedis_estado = df.groupby("CEDISID")["STATE"].first()
    for estado, grupo in cedis_estado.groupby(cedis_estado):
        anclas = ANCLAS.get(estado, [(23.6, -102.5)])
        for i, cedis in enumerate(sorted(grupo.index)):
            rng = np.random.default_rng(int("".join(filter(str.isdigit, cedis)) or 0))
            lat, lon = anclas[i % len(anclas)]
            pos_cedis[cedis] = (lat + rng.normal(0, 0.07), lon + rng.normal(0, 0.07))
    aprox_cedis_lat = df["CEDISID"].map(lambda c: pos_cedis[c][0])
    aprox_cedis_lon = df["CEDISID"].map(lambda c: pos_cedis[c][1])
    centros = {}
    for ruta, g in df.groupby("ROUTEID"):
        rng = np.random.default_rng(int("".join(filter(str.isdigit, ruta)) or 0))
        centros[ruta] = (aprox_cedis_lat[g.index[0]] + rng.normal(0, 0.06), aprox_cedis_lon[g.index[0]] + rng.normal(0, 0.06))
    rng = np.random.default_rng(42)
    aprox_lat = df["ROUTEID"].map(lambda r: centros[r][0]) + rng.normal(0, 0.02, len(df))
    aprox_lon = df["ROUTEID"].map(lambda r: centros[r][1]) + rng.normal(0, 0.02, len(df))
 
    # Coordenadas reales cuando existen
    df["COORD_REAL"] = df["LAT_REAL"].notna() & df["LON_REAL"].notna()
    df["LAT"] = df["LAT_REAL"].where(df["COORD_REAL"], aprox_lat)
    df["LON"] = df["LON_REAL"].where(df["COORD_REAL"], aprox_lon)
 
    # CEDIS: centro de sus clientes con coordenadas reales; si no tiene ninguno, la posición aproximada
    centro_real = df[df["COORD_REAL"]].groupby("CEDISID")[["LAT", "LON"]].mean()
    df["CEDIS_LAT"] = df["CEDISID"].map(centro_real["LAT"]).fillna(aprox_cedis_lat)
    df["CEDIS_LON"] = df["CEDISID"].map(centro_real["LON"]).fillna(aprox_cedis_lon)
    return df
 
 
def zoom_para(lat: pd.Series, lon: pd.Series) -> float:
    """Zoom del mapa para que quepan todos los puntos."""
    extension = max(lat.max() - lat.min(), lon.max() - lon.min(), 0.005)
    return float(np.clip(np.log2(360 / extension) - 1.2, 3, 14))
 
 
def resumen_rutas(d: pd.DataFrame) -> pd.DataFrame:
    """Una fila por ruta con su mezcla de niveles y nivel de la ruta."""
    r = d.groupby(["ROUTEID", "CEDISID", "STATE"]).agg(
        Clientes=("CUSTOMERID", "count"),
        Autonomos=("SEGMENT", lambda s: (s == "Autónomo").mean()),
        Semi=("SEGMENT", lambda s: (s == "Semi autónomo").mean()),
        Dependientes=("SEGMENT", lambda s: (s == "Dependiente").mean()),
        IAD=("IAD", "mean"),
        Acoplamiento=("COUPLED", "mean"),
    ).reset_index()
    nivel = pd.cut(r["Dependientes"], [-0.01, 0.15, 0.30, 1.0],
                   labels=["Autónomo", "Semi autónomo", "Dependiente"]).astype(str)
    r.insert(3, "Nivel de la ruta", nivel.map(NIVEL_RUTA))
    return r.sort_values("Dependientes", ascending=False)
 
 
# ──────────────────────────────────────────────────────────────
# Lista de trabajo: riesgo, protocolo y acción por tienda
# ──────────────────────────────────────────────────────────────
def lista_trabajo(d: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    meta = base[base["SEGMENT"] == "Autónomo"][[f"P_{v}" for v in VARS]].mean()
    brechas = pd.DataFrame({v: meta[f"P_{v}"] - d[f"P_{v}"] for v in VARS})
    d["BRECHA_PRINCIPAL"] = brechas.idxmax(axis=1)
 
    no_autonomo = d["SEGMENT"] != "Autónomo"
    d["RIESGO"] = np.select(
        [
            no_autonomo & (d["COUPLED"] >= 0.40),
            no_autonomo & ((d["COUPLED"] >= 0.25) | (d["PROB_MAX"] < 0.99)),
        ],
        ["Alerta", "Vigilar"],
        default="—",
    )
    d["MOTIVO"] = np.select(
        [
            d["RIESGO"] == "Alerta",
            (d["RIESGO"] == "Vigilar") & (d["COUPLED"] >= 0.25),
            d["RIESGO"] == "Vigilar",
        ],
        [
            "40% o más de pedidos con el preventista",
            "25–40% de pedidos con el preventista",
            "El modelo duda de su nivel (puede cambiar)",
        ],
        default=np.where(d["SEGMENT"] == "Autónomo", "Ya pide solo por la app",
                         "Pocos pedidos con el preventista"),
    )
    d["PROTOCOLO"] = np.where(
        no_autonomo,
        d["BRECHA_PRINCIPAL"].map(PROTOCOLOS),
        "Auditoría de enfriador y exhibición (sin captura de pedido)",
    )
    d["ACCION_MES"] = d["SEGMENT"].map({
        "Autónomo": "Beneficio por pedir en Tuali (programa de lealtad)",
        "Semi autónomo": "Recordatorio en su día de pedido",
        "Dependiente": "Visita de acompañamiento + capacitación en Tuali",
    })
    orden_riesgo = d["RIESGO"].map({"Alerta": 0, "Vigilar": 1, "—": 2})
    orden_nivel = d["SEGMENT"].map({"Dependiente": 0, "Semi autónomo": 1, "Autónomo": 2})
    return d.assign(_r=orden_riesgo, _c=orden_nivel).sort_values(["_r", "_c", "IAD"]).drop(columns=["_r", "_c"])
 
 
# ──────────────────────────────────────────────────────────────
# Carga
# ──────────────────────────────────────────────────────────────
def banner(frase: str):
    tienditas = "".join(tienda_svg("#ffffff", t, cuerpo="rgba(255,255,255,.15)") for t in (54, 76, 54))
    html(f"""
    <div class="banner">
      <div>
        <div class="eyebrow">ARCA CONTINENTAL · DIGITAL NEST · EQUIPO 5</div>
        <h1>Radar de Autonomía Tuali</h1>
        <p>{frase}</p>
      </div>
      <div class="tiendas">{tienditas}</div>
    </div>
    """)
 
 
COLUMNAS_AYUDA = (
    "**Obligatorias** (las 7 variables del modelo): "
    + ", ".join(f"`{v}`" for v in VARS)
    + "\n\n**Opcionales** (si no vienen, el dashboard funciona igual): `CUSTOMERID`, `BUSINESSNAME`, `COUPLED`, "
    + ", ".join(f"`{c}`" for c in OPCIONALES)
    + ", `LATITUDE` / `LATITUD` y `LONGITUDE` / `LONGITUD` (ubicación real del cliente para el mapa)"
    + "\n\nLos niveles de autonomía y su probabilidad se calculan automáticamente con PCA + GMM."
)
 
# Botón para descargar la guía en PDF (el archivo va en la misma carpeta que app.py)
GUIA_PDF = Path(__file__).parent / "Guia_Radar_Autonomia_Tuali.pdf"
if GUIA_PDF.exists():
    st.sidebar.download_button("📖 Descargar guía del dashboard (PDF)", data=GUIA_PDF.read_bytes(),
                               file_name=GUIA_PDF.name, mime="application/pdf", width="stretch", type="primary")
st.sidebar.header("Datos")
subido = st.sidebar.file_uploader("Sube tu base de datos (Excel o CSV)", type=["xlsx", "csv"])
if subido is None:
    # Pantalla de bienvenida: por confidencialidad la app no guarda datos
    banner("Dashboard de segmentación de clientes por nivel de autonomía digital.")
    st.info(
        "🔒 **Por confidencialidad, esta app no guarda datos.**\n\n"
        "Sube tu base de clientes (**Excel o CSV**, con cualquier nombre) en la barra lateral (izquierda) "
        "y el dashboard se arma solo. Si no ves la barra, ábrela con la flecha **»** de arriba a la izquierda.\n\n"
        "El archivo solo se usa durante tu sesión y no se queda guardado."
    )
    with st.expander("¿Qué columnas necesita la base?"):
        st.markdown(COLUMNAS_AYUDA)
    st.stop()
 
try:
    crudo = leer_archivo(subido.getvalue(), subido.name)
except Exception:
    st.error("No pude leer el archivo. Revisa que sea un Excel (.xlsx) o un CSV válido.")
    st.stop()
 
df, faltan = preparar_datos(crudo)
if faltan:
    st.error("A tu base le faltan estas columnas: " + ", ".join(f"`{c}`" for c in faltan))
    st.markdown(COLUMNAS_AYUDA)
    st.stop()
if len(df) < 30:
    st.error(f"La base tiene {len(df)} clientes con datos completos; se necesitan al menos 30 para el modelo.")
    st.stop()
 
modelo = entrenar_modelo(df)
df["SEGMENT"] = modelo["segmentos"]
df["PROB_MAX"] = modelo["prob_max"]
df["NIVEL"] = df["SEGMENT"].map(lambda s: f"{CIRCULO[s]} {s}")
st.sidebar.success(f"📂 {subido.name} · {len(df)} clientes")
HAY_ACOPLE = df["COUPLED"].notna().any()
df = agregar_coordenadas(df)
df = calcular_indices(df, modelo["pesos"])
INVERTIDAS = variables_invertidas(df)
 
# Cortes del medidor IAD: punto medio entre los promedios de cada nivel
_m = df.groupby("SEGMENT")["IAD"].mean()
CORTES = ((_m["Dependiente"] + _m["Semi autónomo"]) / 2, (_m["Semi autónomo"] + _m["Autónomo"]) / 2)
 
# ──────────────────────────────────────────────────────────────
# Filtros
# ──────────────────────────────────────────────────────────────
st.sidebar.header("Filtros")
FILTROS_INICIALES = {f"filtro_{col}": [] for col in DIMENSIONES.values()}
st.sidebar.button("↺ Restablecer filtros", on_click=restablecer, args=(FILTROS_INICIALES,), width="stretch")
filtros = {}
for etiqueta, col in DIMENSIONES.items():
    opciones = sorted(df[col].dropna().unique())
    filtros[col] = st.sidebar.multiselect(etiqueta, opciones, placeholder="Todos", key=f"filtro_{col}")
 
dff = df.copy()
for col, seleccion in filtros.items():
    if seleccion:
        dff = dff[dff[col].isin(seleccion)]
 
st.sidebar.caption(f"Mostrando **{len(dff)}** de {len(df)} clientes.")
 
if dff.empty:
    st.warning("Ningún cliente cumple con los filtros seleccionados.")
    st.stop()
 
# ──────────────────────────────────────────────────────────────
# Encabezado y KPIs
# ──────────────────────────────────────────────────────────────
conteo = dff["SEGMENT"].value_counts().reindex(ORDEN, fill_value=0)
pct = conteo / conteo.sum() * 100
 
banner(f"{pct['Autónomo']:.1f}% de los clientes ya pide solo por Tuali; "
       f"{pct['Dependiente']:.1f}% todavía depende del preventista.")
 
acople_seg = dff.groupby("SEGMENT")["COUPLED"].mean().reindex(ORDEN)
k1, k2, k3, k4 = st.columns(4)
for col, s in zip([k1, k2, k3], ORDEN):
    tarjeta(col, PLURAL[s], f"{pct[s]:.1f}%", f"{conteo[s]} clientes", COLORES[s], tienda_svg(COLORES[s], 56))
if acople_seg.notna().all():
    tarjeta(
        k4, "Pedidos pegados a la visita",
        f"{acople_seg['Dependiente']:.0%} vs {acople_seg['Autónomo']:.0%}",
        "Dependientes vs autónomos: los niveles sí reflejan dependencia del preventista",
    )
else:
    tarjeta(k4, "Pedidos pegados a la visita", f"{dff['COUPLED'].mean():.0%}" if HAY_ACOPLE else "Sin dato",
            "Promedio de la vista" if HAY_ACOPLE else "La base no trae la columna COUPLED")
 
(tab_res, tab_mapa, tab_ficha, tab_lista, tab_radar, tab_sim, tab_esc) = st.tabs([
    "🏪 Panorama", "🗺️ Mapa y rutas", "🪪 Ficha del cliente", "📋 Lista de trabajo",
    "🕸️ Radar de perfiles", "🧪 Simulador de cliente", "🚀 Escenarios e impacto",
])
 
# ──────────────────────────────────────────────────────────────
# 1. Panorama
# ──────────────────────────────────────────────────────────────
with tab_res:
    p1, p2 = st.columns([1.7, 1])
    with p1:
        with st.container(border=True):
            st.subheader("De cada 100 tiendas…")
            iconos = "".join(tienda_svg(COLORES[s], 40) for s in cien_tiendas(conteo))
            n100 = pd.Series(cien_tiendas(conteo)).value_counts().reindex(ORDEN, fill_value=0)
            leyenda = "".join(
                f"<span>{tienda_svg(COLORES[s], 22)} <b>{n100[s]}</b> {PLURAL[s].lower()}</span>" for s in ORDEN
            )
            html(f"<div class='waffle'>{iconos}</div><div class='leyenda'>{leyenda}</div>")
    with p2:
        with st.container(border=True):
            st.subheader("Autonomía promedio")
            st.plotly_chart(medidor(dff["IAD"].mean(), CORTES, "Índice de Autonomía Digital (0 a 10)", 270),
                            width="stretch")
            st.caption("Más a la derecha = clientes que piden solos por Tuali.")
 
    st.subheader("El camino a la autonomía")
    pasos = []
    descripciones = {
        "Dependiente": "Piden cuando llega el preventista y buscan todo por código",
        "Semi autónomo": "Ya usan la app, pero todavía con poca variedad y sin promociones",
        "Autónomo": "Piden solos, aprovechan promociones, favoritos y más productos",
    }
    for s in ["Dependiente", "Semi autónomo", "Autónomo"]:
        acople_txt = f"{acople_seg[s]:.0%} de pedidos con el preventista" if pd.notna(acople_seg[s]) else ""
        pasos.append(
            f"<div class='paso' style='border-top:6px solid {COLORES[s]}'>{tienda_svg(COLORES[s], 84)}"
            f"<div class='paso-n' style='color:{COLORES[s]}'>{conteo[s]}</div>"
            f"<div class='paso-t'>{PLURAL[s]}</div><div class='paso-s'>{descripciones[s]}</div>"
            f"<div class='paso-s' style='margin-top:6px'><b>{acople_txt}</b></div></div>"
        )
    html(f"<div class='camino'>{pasos[0]}<div class='flecha'>➜</div>{pasos[1]}<div class='flecha'>➜</div>{pasos[2]}</div>")
    st.write("")
 
    with st.container(border=True):
        st.subheader("Distribución del Índice de Autonomía Digital")
        st.caption("Clientes por puntaje de 0 a 10, apilados por nivel de autonomía.")
        fig = px.histogram(dff, x="IAD", color="SEGMENT", nbins=40, color_discrete_map=COLORES,
                           category_orders={"SEGMENT": ORDEN}, labels={"SEGMENT": "Nivel", "IAD": "IAD"})
        fig.update_layout(barmode="stack", bargap=0.08, xaxis=dict(range=[0, 10]), yaxis_title="Clientes",
                          legend=dict(orientation="h", y=1.1, title=None), height=340, margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")
 
    c1, c2 = st.columns([1, 1.4])
    with c1:
        with st.container(border=True):
            st.subheader("Distribución por nivel")
            fig = px.pie(names=conteo.index, values=conteo.values, hole=0.6, color=conteo.index,
                         color_discrete_map=COLORES)
            fig.update_traces(textinfo="percent+label", sort=False, marker=dict(line=dict(color="white", width=3)))
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10), height=380,
                              annotations=[dict(text=f"<b>{len(dff)}</b><br>tiendas", showarrow=False,
                                                font=dict(size=20, family="Fraunces, serif"))])
            st.plotly_chart(fig, width="stretch")
 
    with c2:
        with st.container(border=True):
            st.subheader("Nivel de autonomía por grupo")
            dim_label = st.selectbox("Ver por", list(DIMENSIONES.keys()), key="dim_resumen")
            dim = DIMENSIONES[dim_label]
            mezcla = dff.groupby(dim)["SEGMENT"].value_counts(normalize=True).mul(100).rename("Porcentaje").reset_index()
            orden_dim = mezcla[mezcla["SEGMENT"] == "Autónomo"].sort_values("Porcentaje")[dim].tolist()
            orden_dim += [x for x in mezcla[dim].unique() if x not in orden_dim]
            fig = px.bar(mezcla, y=dim, x="Porcentaje", color="SEGMENT", orientation="h",
                         color_discrete_map=COLORES, category_orders={"SEGMENT": ORDEN, dim: orden_dim},
                         labels={dim: dim_label, "SEGMENT": "Nivel"})
            fig.update_layout(barmode="stack", xaxis_ticksuffix="%", legend=dict(orientation="h", y=1.08, title=None),
                              margin=dict(t=30, b=10), height=max(300, 22 * len(orden_dim)))
            st.plotly_chart(fig, width="stretch")
 
# ──────────────────────────────────────────────────────────────
# 2. Mapa y rutas
# ──────────────────────────────────────────────────────────────
with tab_mapa:
    with st.container(border=True):
        st.subheader("¿Dónde están las tiendas?")
        fig = px.scatter_map(
            dff, lat="LAT", lon="LON", color="SEGMENT", color_discrete_map=COLORES,
            category_orders={"SEGMENT": ORDEN}, hover_name="CLIENTE",
            hover_data={"ROUTEID": True, "CEDISID": True, "IAD": ":.1f", "LAT": False, "LON": False},
            labels={"SEGMENT": "Nivel"}, zoom=zoom_para(dff["LAT"], dff["LON"]),
            center=dict(lat=dff["LAT"].mean(), lon=dff["LON"].mean()), height=520,
            map_style="carto-positron",
        )
        fig.update_traces(marker=dict(size=10, opacity=0.85))
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), legend=dict(orientation="h", y=1.05, title=None))
        st.plotly_chart(fig, width="stretch")
        n_real = int(dff["COORD_REAL"].sum())
        if n_real == len(dff):
            st.caption("📍 Ubicación real de cada cliente (latitud y longitud de la base).")
        elif n_real > 0:
            st.caption(f"📍 {n_real} de {len(dff)} clientes con ubicación real; el resto sin coordenadas se ubica "
                       "de forma aproximada cerca de su CEDIS y su ruta.")
        else:
            st.caption("Ubicación aproximada: la base no trae latitud y longitud, así que cada cliente se ubica "
                       "cerca de su CEDIS y su ruta.")
 
    st.subheader("¿Dónde están los dependientes? · Rutas actuales")
    rutas = resumen_rutas(dff)
    s1, s2, s3 = st.columns(3)
    textos_ruta = {
        "Dependiente": "más de 30% de sus clientes son dependientes",
        "Semi autónomo": "entre 15% y 30% de dependientes",
        "Autónomo": "menos de 15% de dependientes",
    }
    for col, s in zip([s1, s2, s3], ["Dependiente", "Semi autónomo", "Autónomo"]):
        n = int((rutas["Nivel de la ruta"] == NIVEL_RUTA[s]).sum())
        tarjeta(col, f"Rutas {NIVEL_RUTA[s][2:].lower()}s", n, textos_ruta[s], COLORES[s], tienda_svg(COLORES[s], 50))
 
    color_por_nivel = {v: COLORES[k] for k, v in NIVEL_RUTA.items()}
 
    def color_fila(fila):
        return [f"background-color: {color_por_nivel[fila['Nivel de la ruta']]}33"] * len(fila)
 
    st.dataframe(
        rutas.style.apply(color_fila, axis=1).format(
            {"Autonomos": "{:.0%}", "Semi": "{:.0%}", "Dependientes": "{:.0%}", "IAD": "{:.1f}", "Acoplamiento": "{:.1%}"}
        ),
        width="stretch", hide_index=True, height=320,
    )
 
# ──────────────────────────────────────────────────────────────
# 3. Ficha del cliente
# ──────────────────────────────────────────────────────────────
trabajo = lista_trabajo(dff, df)
 
with tab_ficha:
    nivel_de = trabajo.set_index("CLIENTE")["SEGMENT"]
    cliente = st.selectbox("Buscar tienda o ID", trabajo["CLIENTE"].tolist(), key="ficha_cliente",
                           format_func=lambda c: f"{CIRCULO[nivel_de[c]]} {c} · {nivel_de[c]}")
    r = trabajo[trabajo["CLIENTE"] == cliente].iloc[0]
    proba = predecir(modelo, r[VARS].to_dict())
    nivel = r["SEGMENT"]
 
    f1, f2 = st.columns(2)
    with f1:
        with st.container(border=True):
            html(
                f"<div style='display:flex;gap:18px;align-items:center'>{tienda_svg(COLORES[nivel], 96)}"
                f"<div><h3 style='margin:0'>{r['BUSINESSNAME']}</h3>"
                f"<div style='opacity:.7;font-size:.9rem;margin:4px 0 8px'>Cliente {r['CUSTOMERID']} · "
                f"{r['BUSINESSTYPE']} · {r['SUBCHANNEL']} · {r['STATE']}</div>"
                f"<span class='badge' style='background:{COLORES[nivel]}'>{nivel}</span></div></div>"
            )
            nivel_prom = df[df["SEGMENT"] == nivel]["COUPLED"].mean()
            riesgo_txt = ("Sin riesgo" if r["RIESGO"] == "—" else r["RIESGO"]) + f" · {r['MOTIVO']}"
            datos = pd.DataFrame({
                "Dato": ["Ruta", "CEDIS", "Tamaño", "Cliente desde", "Pedidos pegados a visita", "Riesgo"],
                "Valor": [r["ROUTEID"], r["CEDISID"], r["CUSTOMERSIZE"], str(r["JOINDATE"])[:10],
                          (f"{r['COUPLED']:.0%} (promedio de su nivel: {nivel_prom:.0%})" if pd.notna(r["COUPLED"]) else "Sin dato"), riesgo_txt],
            })
            st.dataframe(datos, hide_index=True, width="stretch")
 
    with f2:
        with st.container(border=True):
            st.plotly_chart(medidor(r["IAD"], CORTES, "Índice de Autonomía Digital", 240), width="stretch")
            st.markdown("**Probabilidad de cada nivel (modelo GMM)**")
            fig = px.bar(x=proba.values, y=proba.index, orientation="h", color=proba.index,
                         color_discrete_map=COLORES, text=[f"{p:.0%}" for p in proba.values])
            fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1], tickformat=".0%", title=None),
                              yaxis=dict(title=None, autorange="reversed"), height=160, margin=dict(t=5, b=5))
            st.plotly_chart(fig, width="stretch")
 
    a1, a2 = st.columns(2)
    html(f"<div class='accion'><b>📅 {r['ACCION_MES']}</b><span>Acción del mes según su nivel</span></div>", a1)
    html(f"<div class='accion'><b>🤝 {r['PROTOCOLO']}</b><span>Protocolo en la visita del preventista</span></div>", a2)
 
    f3, f4 = st.columns(2)
    puntajes = {f"P_{v}": v for v in VARS}
    etiquetas = [NOMBRES[v] + (" (inv.)" if v in INVERTIDAS else "") for v in VARS]
    with f3:
        with st.container(border=True):
            st.markdown("**Sus variables contra los promedios**")
            st.caption("Puntaje de 0 a 10 por percentil. Más afuera = más autónomo. (inv.) = escala invertida.")
            perf = {
                "Autónomo promedio": df[df["SEGMENT"] == "Autónomo"][list(puntajes)].mean().rename(puntajes),
                "Dependiente promedio": df[df["SEGMENT"] == "Dependiente"][list(puntajes)].mean().rename(puntajes),
                "Este cliente": r[list(puntajes)].rename(puntajes).astype(float),
            }
            st.plotly_chart(radar(perf, rango=(0, 10), etiquetas=etiquetas), width="stretch")
 
    with f4:
        with st.container(border=True):
            st.markdown("**Qué le falta para subir**")
            st.caption("Las dos variables con mayor brecha contra el autónomo promedio.")
            meta = df[df["SEGMENT"] == "Autónomo"][list(puntajes)].mean()
            brechas = (meta - r[list(puntajes)].astype(float)).rename(puntajes).sort_values(ascending=False)
            brechas = brechas[brechas > 0.5].head(2)
            if brechas.empty:
                st.success("Ya está al nivel del autónomo promedio en todas sus variables.")
            for v, b in brechas.items():
                html(
                    f"<div class='accion' style='border-left-color:{COLORES['Semi autónomo']}'><b>{NOMBRES[v]}</b>"
                    f"<span>Su puntaje: {r['P_' + v]:.1f} de 10 · autónomo promedio: {meta['P_' + v]:.1f} · "
                    f"brecha: {b:.1f}<br>Qué hacer: {PROTOCOLOS[v]}</span></div>"
                )
            st.markdown("**Su autonomía frente a su ruta**")
            ruta_df = df[df["ROUTEID"] == r["ROUTEID"]].sort_values("IAD")
            fig = px.bar(ruta_df, x="IAD", y=ruta_df["BUSINESSNAME"].str.replace("Comercializadora ", ""),
                         orientation="h", color="SEGMENT", color_discrete_map=COLORES,
                         category_orders={"SEGMENT": ORDEN})
            fig.update_layout(showlegend=False, yaxis_title=None, xaxis=dict(range=[0, 10]),
                              height=max(220, 24 * len(ruta_df)), margin=dict(t=5, b=5))
            idx = ruta_df["CLIENTE"].tolist().index(cliente)
            fig.add_annotation(x=r["IAD"], y=idx, text="◀ esta tienda", showarrow=False, xanchor="left", xshift=6)
            st.plotly_chart(fig, width="stretch")
 
# ──────────────────────────────────────────────────────────────
# 4. Lista de trabajo
# ──────────────────────────────────────────────────────────────
with tab_lista:
    st.write("Herramienta para el supervisor y el preventista: qué hacer en cada tienda de la ruta y cuáles están en riesgo.")
    l1, l2, l3, l4 = st.columns(4)
    tarjeta(l1, "🚨 Alertas", int((trabajo["RIESGO"] == "Alerta").sum()),
            "Semi o dependientes con 40% o más de pedidos pegados a la visita", COLORES["Dependiente"])
    tarjeta(l2, "👀 En vigilancia", int((trabajo["RIESGO"] == "Vigilar").sum()),
            "Pegados a la visita (25–40%) o el modelo duda de su nivel", COLORES["Semi autónomo"])
    tarjeta(l3, "Visitas que se transforman", int((trabajo["SEGMENT"] == "Autónomo").sum()),
            "Tiendas autónomas: auditoría en lugar de captura", COLORES["Autónomo"], tienda_svg(COLORES["Autónomo"], 50))
    tarjeta(l4, "Tiendas en la vista", len(trabajo), "Usa la Ficha del cliente para ver el detalle")
 
    b1, b2 = st.columns([1.6, 2])
    busqueda = b1.text_input("Buscar tienda", placeholder="Nombre o ID…")
    filtro = b2.radio("Mostrar", ["Todas", "Solo riesgo", "Dependientes", "Semi autónomos", "Autónomos"], horizontal=True)
 
    vista_l = trabajo
    if busqueda:
        vista_l = vista_l[vista_l["CLIENTE"].str.contains(busqueda, case=False)]
    if filtro == "Solo riesgo":
        vista_l = vista_l[vista_l["RIESGO"] != "—"]
    elif filtro != "Todas":
        vista_l = vista_l[vista_l["SEGMENT"] == {v: k for k, v in PLURAL.items()}[filtro]]
 
    tabla_l = pd.DataFrame({
        "ID": vista_l["CUSTOMERID"],
        "Tienda": vista_l["BUSINESSNAME"],
        "Nivel de autonomía": vista_l["NIVEL"],
        "IAD": vista_l["IAD"],
        "Riesgo": vista_l["RIESGO"].map({"Alerta": "🚨 Alerta", "Vigilar": "👀 Vigilar", "—": "✅ Sin riesgo"}),
        "Motivo": vista_l["MOTIVO"],
        "Ruta": vista_l["ROUTEID"],
        "Pegados a visita": vista_l["COUPLED"],
        "Protocolo en visita": vista_l["PROTOCOLO"],
        "Acción del mes": vista_l["ACCION_MES"],
    })
    st.dataframe(
        tabla_l, hide_index=True, width="stretch", height=480,
        column_config={
            "IAD": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=10),
            "Pegados a visita": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        },
    )
    with st.expander("Ver todas las variables del modelo"):
        st.dataframe(vista_l[["CUSTOMERID", "BUSINESSNAME", "SEGMENT", "PROB_MAX"] + VARS], hide_index=True, width="stretch")
    st.download_button("Descargar lista de trabajo (CSV)", tabla_l.to_csv(index=False).encode("utf-8-sig"),
                       file_name="lista_de_trabajo.csv", mime="text/csv")
 
# ──────────────────────────────────────────────────────────────
# 5. Radar de perfiles
# ──────────────────────────────────────────────────────────────
with tab_radar:
    st.subheader("¿Cómo se comporta cada nivel en Tuali?")
    st.write("Cada eje está escalado de 0 (mínimo de toda la base) a 1 (máximo), para poder comparar variables "
             "con unidades distintas. Puedes superponer un cliente específico.")
 
    perfiles_real = dff.groupby("SEGMENT")[VARS].mean().reindex(ORDEN).dropna()
    perfiles_norm = normalizar(perfiles_real, df)
 
    col_a, col_b = st.columns([2.2, 1])
    with col_b:
        segs = st.multiselect("Niveles a mostrar", ORDEN, default=list(perfiles_norm.index))
        cliente_sel = st.selectbox("Superponer cliente (opcional)", ["—"] + dff["CLIENTE"].tolist(), key="radar_cliente")
 
    perfiles = {s: perfiles_norm.loc[s] for s in segs if s in perfiles_norm.index}
    if cliente_sel != "—":
        fila = dff[dff["CLIENTE"] == cliente_sel]
        perfiles[cliente_sel.split(" · ")[0] + " (cliente)"] = normalizar(fila, df).iloc[0]
        with col_b:
            rr = fila.iloc[0]
            html(f"<div style='display:flex;gap:12px;align-items:center'>{tienda_svg(COLORES[rr['SEGMENT']], 60)}"
                 f"<div><b>{rr['BUSINESSNAME']}</b><br>{rr['SEGMENT']} · {rr['PROB_MAX']:.1%} de probabilidad<br>"
                 f"<span style='opacity:.7'>{rr['SUBCHANNEL']} · {rr['CUSTOMERSIZE']} · {rr['STATE']}</span></div></div>")
 
    with col_a:
        with st.container(border=True):
            st.plotly_chart(radar(perfiles), width="stretch")
 
    st.markdown("**Promedios reales por nivel**")
    tabla = perfiles_real.copy()
    tabla["COUPLED"] = dff.groupby("SEGMENT")["COUPLED"].mean()
    tabla["Clientes"] = conteo
    tabla = tabla.rename(columns=NOMBRES)
    st.dataframe(
        tabla.style.format({
            NOMBRES["PROMOTIONRATE"]: "{:.1%}", NOMBRES["CODESEARCHRATE"]: "{:.1%}",
            NOMBRES["FAVORITERATE"]: "{:.1%}", NOMBRES["COUPLED"]: "{:.1%}",
            NOMBRES["EVENTCOUNT"]: "{:.0f}", NOMBRES["DISTINCTEVENTTYPES"]: "{:.1f}",
            NOMBRES["DISTINCT_PRODUCTS"]: "{:.0f}", NOMBRES["AVGSESSIONDURATIONSECONDS"]: "{:.0f}",
        }),
        width="stretch",
    )
# ──────────────────────────────────────────────────────────────
# 6. Simulador de cliente (what-if con el GMM)
# ──────────────────────────────────────────────────────────────
with tab_sim:
    st.subheader("¿Qué necesita cambiar una tienda para subir de nivel?")
    st.write("Elige un punto de partida y mueve las variables. El modelo (el mismo PCA + GMM del notebook) "
             "recalcula en vivo la probabilidad de pertenecer a cada nivel.")
 
    puntos_partida = [f"Promedio {s}" for s in ORDEN] + df["CLIENTE"].tolist()
 
    def cargar_punto_partida():
        sel = st.session_state["sim_base"]
        if sel.startswith("Promedio "):
            base = df[df["SEGMENT"] == sel.replace("Promedio ", "")][VARS].mean()
        else:
            base = df[df["CLIENTE"] == sel][VARS].iloc[0]
        for v in VARS:
            st.session_state[f"sim_{v}"] = int(round(base[v])) if v in VARS_ENTERAS else float(base[v])
 
    if "sim_base" not in st.session_state:
        st.session_state["sim_base"] = "Promedio Dependiente"
        cargar_punto_partida()
 
    sb1, sb2 = st.columns([3, 1])
    sb1.selectbox("Punto de partida", puntos_partida, key="sim_base", on_change=cargar_punto_partida)
    sb2.write("")
    sb2.button("↺ Restablecer valores", on_click=cargar_punto_partida, key="reset_simulador", width="stretch")
 
    col_s, col_r = st.columns([1, 1.3])
    valores = {}
    with col_s:
        with st.container(border=True):
            for v in VARS:
                mn, mx = df[v].min(), df[v].max()
                if v in VARS_ENTERAS:
                    valores[v] = st.slider(NOMBRES[v], int(mn), int(mx), key=f"sim_{v}")
                elif v in VARS_TASA:
                    valores[v] = st.slider(NOMBRES[v], 0.0, 1.0, step=0.01, key=f"sim_{v}", format="%.2f")
                else:
                    valores[v] = st.slider(NOMBRES[v], float(mn), float(mx), step=10.0, key=f"sim_{v}", format="%.0f")
 
    proba_sim = predecir(modelo, valores)
    nivel_sim = proba_sim.idxmax()
 
    with col_r:
        html(
            f"<div class='card' style='border-top:6px solid {COLORES[nivel_sim]};align-items:center'>"
            f"<div><div class='card-t'>Nivel estimado</div><div class='card-v'>{nivel_sim}</div>"
            f"<div class='card-s'>{proba_sim[nivel_sim]:.1%} de probabilidad</div></div>"
            f"{tienda_svg(COLORES[nivel_sim], 90)}</div>"
        )
        fig = px.bar(x=proba_sim.values, y=proba_sim.index, orientation="h", color=proba_sim.index,
                     color_discrete_map=COLORES, text=[f"{p:.1%}" for p in proba_sim.values],
                     labels={"x": "Probabilidad", "y": ""})
        fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1], tickformat=".0%"), height=180,
                          yaxis=dict(autorange="reversed"), margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
 
        if proba_sim.max() < 0.70:
            st.warning("Probabilidad máxima menor a 70%: esta tienda estaría **en transición** entre niveles.")
 
        sim_norm = normalizar(pd.DataFrame([valores]), df).iloc[0]
        perfiles_sim = {s: normalizar(df[df["SEGMENT"] == s][VARS].mean().to_frame().T, df).iloc[0] for s in ORDEN}
        perfiles_sim["Tienda simulada"] = sim_norm
        st.plotly_chart(radar(perfiles_sim), width="stretch")
 
# ──────────────────────────────────────────────────────────────
# 7. Escenarios e impacto
# ──────────────────────────────────────────────────────────────
with tab_esc:
    st.subheader("Simulador de impacto")
    st.write("Horas de preventista que se liberan al dejar de capturar pedidos en tiendas autónomas, y cuántas "
             "tiendas en transición se podrían acompañar con ese tiempo.")
 
    n_aut_v = int(conteo["Autónomo"])
    n_semi_v = int(conteo["Semi autónomo"])
    IMPACTO_INICIAL = {"imp_visitas": 16.5, "imp_minutos": 33.4, "imp_reduccion": 50, "imp_suben": 20}
    iniciar(IMPACTO_INICIAL)
    with st.container(border=True):
        i1, i2, i3 = st.columns(3)
        visitas_ano = i1.slider("Visitas al año por tienda autónoma", 4.0, 52.0, step=0.5, key="imp_visitas")
        min_visita = i2.slider("Minutos por visita", 10.0, 60.0, step=0.1, key="imp_minutos")
        reduccion = i3.slider("Reducción de visitas de captura (%)", 0, 100, step=5, key="imp_reduccion")
        i4, i5 = st.columns([3, 1])
        suben = i4.slider("Tiendas semi autónomas que suben de nivel (%)", 0, 100, step=5, key="imp_suben")
        i5.write("")
        i5.button("↺ Restablecer valores", on_click=restablecer, args=(IMPACTO_INICIAL,), key="reset_impacto",
                  width="stretch")
        st.caption("Valores iniciales tomados del análisis con datos 2025. Ajústalos para probar escenarios.")
 
    horas_hoy = n_aut_v * visitas_ano * min_visita / 60
    horas_lib = horas_hoy * reduccion / 100
    jornadas = horas_lib / 8
    fte = jornadas / 264
    tiendas_suben = round(n_semi_v * suben / 100)
 
    j1, j2, j3, j4 = st.columns(4)
    tarjeta(j1, "⏱️ Horas al año hoy en autónomos", f"{horas_hoy:,.0f} h",
            f"{n_aut_v} tiendas × {visitas_ano} visitas × {min_visita} min ÷ 60")
    tarjeta(j2, "🕒 Horas liberadas", f"+{horas_hoy * reduccion / 100:,.0f} h", f"{horas_hoy:,.0f} h × {reduccion}% de reducción",
            COLORES["Autónomo"])
    tarjeta(j3, "📆 Equivale a", f"{jornadas:,.0f} jornadas", f"{fte:.2f} preventistas de tiempo completo",
            COLORES["Autónomo"])
    tarjeta(j4, "⬆️ Tiendas que suben", f"{tiendas_suben}", f"{n_semi_v} semi autónomas × {suben}%",
            COLORES["Semi autónomo"], tienda_svg(COLORES["Semi autónomo"], 44))
 
    # Gráfica de cascada: de horas actuales a horas liberadas
    with st.container(border=True):
        st.markdown("**¿A dónde se va el tiempo del preventista en tiendas autónomas?**")
        fig = go.Figure(go.Bar(
            x=[horas_hoy - horas_lib, horas_lib], y=["Horas que se mantienen<br>(auditoría)", "Horas liberadas<br>(para acompañar semi)"],
            orientation="h", marker_color=[COLORES["Semi autónomo"], COLORES["Autónomo"]],
            text=[f"{horas_hoy - horas_lib:,.0f} h", f"{horas_lib:,.0f} h"], textposition="auto",
        ))
        fig.update_layout(height=200, margin=dict(t=10, b=10), xaxis_title="Horas al año")
        st.plotly_chart(fig, width="stretch")
 
    st.divider()
 
    st.subheader("¿Qué pasa si movemos tiendas de nivel?")
    if not HAY_ACOPLE:
        st.info("Este escenario usa la columna COUPLED (pedidos pegados a la visita) y tu base no la trae.")
        st.stop()  # es la última sección del dashboard, así que no corta nada más
    st.write("Define qué porcentaje de cada nivel logra subir uno con las estrategias de adopción. "
             "El impacto se estima con el acoplamiento promedio observado en cada nivel.")
 
    ESCENARIO_INICIAL = {"esc_dep": 30, "esc_semi": 20, "esc_pedidos": 4}
    iniciar(ESCENARIO_INICIAL)
    with st.container(border=True):
        e1, e2, e3, e4 = st.columns([1.2, 1.2, 1, 0.8])
        mig_dep = e1.slider("Dependientes que pasan a Semi autónomo (%)", 0, 100, step=5, key="esc_dep")
        mig_semi = e2.slider("Semi autónomos que pasan a Autónomo (%)", 0, 100, step=5, key="esc_semi")
        pedidos_mes = e3.number_input("Pedidos por cliente al mes (supuesto)", min_value=1, max_value=60, key="esc_pedidos")
        e4.write("")
        e4.button("↺ Restablecer valores", on_click=restablecer, args=(ESCENARIO_INICIAL,), key="reset_escenario",
                  width="stretch")
 
    acople = dff.groupby("SEGMENT")["COUPLED"].mean().reindex(ORDEN).fillna(df.groupby("SEGMENT")["COUPLED"].mean())
    actual = conteo.astype(float)
    pasan_dep = actual["Dependiente"] * mig_dep / 100
    pasan_semi = actual["Semi autónomo"] * mig_semi / 100
    nuevo = pd.Series({
        "Autónomo": actual["Autónomo"] + pasan_semi,
        "Semi autónomo": actual["Semi autónomo"] - pasan_semi + pasan_dep,
        "Dependiente": actual["Dependiente"] - pasan_dep,
    })
    acople_actual = (actual * acople).sum() / actual.sum()
    acople_nuevo = (nuevo * acople).sum() / nuevo.sum()
    pedidos_liberados = (acople_actual - acople_nuevo) * actual.sum() * pedidos_mes
 
    m1, m2, m3, m4 = st.columns(4)
    tarjeta(m1, "Tiendas que suben de nivel", f"{pasan_dep + pasan_semi:,.0f}", "Con las estrategias de adopción",
            COLORES["Autónomo"], tienda_svg(COLORES["Autónomo"], 44))
    tarjeta(m2, "Autónomos", f"{nuevo['Autónomo'] / nuevo.sum():.1%}",
            f"{(nuevo['Autónomo'] - actual['Autónomo']) / actual.sum() * 100:+.1f} pp vs. hoy", COLORES["Autónomo"])
    tarjeta(m3, "Pedidos pegados a visita", f"{acople_nuevo:.1%}",
            f"{(acople_nuevo - acople_actual) * 100:+.1f} pp vs. hoy", COLORES["Semi autónomo"])
    tarjeta(m4, "Pedidos/mes sin depender de visita", f"+{pedidos_liberados:,.0f}", f"Supuesto: {pedidos_mes} pedidos por tienda")
 
    with st.container(border=True):
        c_a, c_b = st.columns(2)
        for col, titulo, serie in [(c_a, "Hoy", actual), (c_b, "Escenario", nuevo)]:
            with col:
                st.markdown(f"**{titulo}: de cada 100 tiendas…**")
                iconos = "".join(tienda_svg(COLORES[s], 30) for s in cien_tiendas(serie.round().astype(int)))
                html(f"<div class='waffle'>{iconos}</div>")
