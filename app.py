"""
Radar de Autonomía Tuali · Arca Continental · Equipo 5
Niveles de autonomía digital con PCA + GMM

Cómo correrlo (terminal de VS Code, dentro de la carpeta del proyecto):
    pip install streamlit pandas plotly scikit-learn openpyxl scipy
    streamlit run app.py

El archivo gmm_niveles_autonomia.xlsx debe estar en la misma carpeta que app.py
(o se puede subir desde la barra lateral).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────
# Configuración general
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Radar de Autonomía Tuali", page_icon="📊", layout="wide")

ARCHIVO = Path(__file__).parent / "gmm_niveles_autonomia.xlsx"

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
COLORES = {"Autónomo": "#4f9d69", "Semi autónomo": "#d9a33b", "Dependiente": "#c94c4c"}
SEMAFORO_NIVEL = {"Autónomo": "Verde", "Semi autónomo": "Amarillo", "Dependiente": "Rojo"}
EMOJI = {"Verde": "🟢", "Amarillo": "🟡", "Rojo": "🔴"}
ROJO_ARCA = "#c62828"

DIMENSIONES = {
    "Estado": "STATE",
    "CEDIS": "CEDISID",
    "Ruta": "ROUTEID",
    "Subcanal": "SUBCHANNEL",
    "Tamaño de cliente": "CUSTOMERSIZE",
    "Tipo de negocio": "BUSINESSTYPE",
}

# ──────────────────────────────────────────────────────────────
# Estilo (encabezado rojo y tarjetas)
# ──────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    .banner {{background: linear-gradient(90deg, #8e1b1b, {ROJO_ARCA}); padding: 22px 28px;
              border-radius: 12px; color: white; margin-bottom: 18px;}}
    .banner .eyebrow {{font-size: .78rem; letter-spacing: .14em; opacity: .9;}}
    .banner h1 {{font-family: Georgia, serif; font-weight: 400; font-size: 2.1rem; margin: 4px 0 6px;
                 color: white; padding: 0;}}
    .banner p {{margin: 0; font-size: 1rem;}}
    .card {{background: rgba(127,127,127,.08); border-radius: 10px; padding: 14px 16px;
            min-height: 128px; margin-bottom: 8px;}}
    .card-t {{font-size: .85rem; opacity: .75;}}
    .card-v {{font-size: 1.9rem; font-family: Georgia, serif; margin: 4px 0;}}
    .card-s {{font-size: .8rem; opacity: .7;}}
    .accion {{border-left: 4px solid {ROJO_ARCA}; background: rgba(127,127,127,.08); border-radius: 6px;
              padding: 10px 14px; margin-bottom: 8px;}}
    .accion b {{display: block;}}
    .accion span {{font-size: .8rem; opacity: .7;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def tarjeta(col, titulo, valor, sub="", color=ROJO_ARCA):
    col.markdown(
        f"<div class='card' style='border-top:4px solid {color}'><div class='card-t'>{titulo}</div>"
        f"<div class='card-v'>{valor}</div><div class='card-s'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────
# Datos y modelo
# ──────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos(fuente) -> pd.DataFrame:
    df = pd.read_excel(fuente)
    df["CLIENTE"] = df["CUSTOMERID"].astype(str) + " · " + df["BUSINESSNAME"]
    df["SEMAFORO"] = df["SEGMENT"].map(SEMAFORO_NIVEL)
    return df


@st.cache_resource
def entrenar_modelo(df: pd.DataFrame):
    """Replica el pipeline del notebook: StandardScaler → PCA(3) → GMM(3)."""
    X_raw = df[VARS].values
    scaler = StandardScaler().fit(X_raw)
    pca = PCA(n_components=3, random_state=42).fit(scaler.transform(X_raw))
    X = pca.transform(scaler.transform(X_raw))
    gmm = GaussianMixture(
        n_components=3, covariance_type="full", reg_covar=0.01, n_init=10, random_state=42
    ).fit(X)
    clusters = gmm.predict(X)

    # A cada cluster se le asigna el nombre del segmento con el que más coincide en el Excel
    tabla = pd.crosstab(clusters, df["SEGMENT"])
    mapa = {c: tabla.loc[c].idxmax() for c in tabla.index}
    coincidencia = (pd.Series(clusters).map(mapa).values == df["SEGMENT"].values).mean()

    return {
        "scaler": scaler,
        "pca": pca,
        "gmm": gmm,
        "mapa": mapa,
        "coincidencia": coincidencia,
        "componentes": X,
        "var_explicada": pca.explained_variance_ratio_,
        "silueta": silhouette_score(X, clusters),
        # Peso de cada variable = carga al cuadrado en el primer componente (suman 1)
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
        color = COLORES.get(nombre.replace(" promedio", ""), "#264653")
        es_segmento = nombre.replace(" promedio", "") in COLORES
        fig.add_trace(
            go.Scatterpolar(
                r=valores + valores[:1],
                theta=etiquetas + etiquetas[:1],
                name=nombre,
                fill="toself" if es_segmento else "none",
                opacity=0.55 if es_segmento else 1,
                line=dict(color=color, width=2 if es_segmento else 3, dash=None if es_segmento else "dash"),
                hovertemplate="%{theta}: %{r:.2f}<extra>" + nombre + "</extra>",
            )
        )
    paso = (rango[1] - rango[0]) / 4
    fig.update_layout(
        title=titulo,
        polar=dict(radialaxis=dict(visible=True, range=list(rango),
                                   tickvals=[rango[0] + paso * i for i in range(1, 5)])),
        legend=dict(orientation="h", y=-0.12),
        margin=dict(t=50, b=40, l=60, r=60),
        height=480,
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Mapa y rutas
# ──────────────────────────────────────────────────────────────
# Ciudades de referencia por estado (la base no trae coordenadas)
ANCLAS = {
    "Baja California": [(32.51, -117.04), (32.62, -115.45), (31.87, -116.60), (32.45, -116.90)],
    "Coahuila": [(25.42, -101.00), (25.54, -103.41), (26.90, -101.42), (28.70, -100.52)],
    "Jalisco": [(20.67, -103.35), (20.65, -105.22), (21.36, -101.93), (19.70, -103.46)],
}
SEMAFORO = {"Verde": "#4f9d69", "Amarillo": "#d9a33b", "Rojo": "#c94c4c"}


@st.cache_data
def agregar_coordenadas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega LAT/LON por cliente y CEDIS_LAT/CEDIS_LON.
    Si el Excel trae LATITUDE y LONGITUDE se usan; si no, se generan posiciones aproximadas:
    cada CEDIS en una ciudad de su estado y cada cliente alrededor del centro de su ruta."""
    df = df.copy()
    pos_cedis = {}
    cedis_estado = df.groupby("CEDISID")["STATE"].first()
    for estado, grupo in cedis_estado.groupby(cedis_estado):
        anclas = ANCLAS.get(estado, [(23.6, -102.5)])
        for i, cedis in enumerate(sorted(grupo.index)):
            rng = np.random.default_rng(int("".join(filter(str.isdigit, cedis)) or 0))
            lat, lon = anclas[i % len(anclas)]
            pos_cedis[cedis] = (lat + rng.normal(0, 0.07), lon + rng.normal(0, 0.07))
    df["CEDIS_LAT"] = df["CEDISID"].map(lambda c: pos_cedis[c][0])
    df["CEDIS_LON"] = df["CEDISID"].map(lambda c: pos_cedis[c][1])

    if {"LATITUDE", "LONGITUDE"}.issubset(df.columns):
        df["LAT"], df["LON"] = df["LATITUDE"], df["LONGITUDE"]
        return df

    centros = {}
    for ruta, g in df.groupby("ROUTEID"):
        rng = np.random.default_rng(int("".join(filter(str.isdigit, ruta)) or 0))
        centros[ruta] = (g["CEDIS_LAT"].iloc[0] + rng.normal(0, 0.06), g["CEDIS_LON"].iloc[0] + rng.normal(0, 0.06))
    rng = np.random.default_rng(42)
    df["LAT"] = df["ROUTEID"].map(lambda r: centros[r][0]) + rng.normal(0, 0.02, len(df))
    df["LON"] = df["ROUTEID"].map(lambda r: centros[r][1]) + rng.normal(0, 0.02, len(df))
    return df


def resumen_rutas(d: pd.DataFrame) -> pd.DataFrame:
    """Una fila por ruta con su mezcla de niveles, semáforo y estrategia sugerida."""
    r = d.groupby(["ROUTEID", "CEDISID", "STATE"]).agg(
        Clientes=("CUSTOMERID", "count"),
        Autonomos=("SEGMENT", lambda s: (s == "Autónomo").mean()),
        Semi=("SEGMENT", lambda s: (s == "Semi autónomo").mean()),
        Dependientes=("SEGMENT", lambda s: (s == "Dependiente").mean()),
        IAD=("IAD", "mean"),
        Acoplamiento=("COUPLED", "mean"),
    ).reset_index()
    r["Semáforo"] = pd.cut(r["Dependientes"], [-0.01, 0.15, 0.30, 1.0], labels=["Verde", "Amarillo", "Rojo"]).astype(str)
    r["Estrategia"] = np.select(
        [
            (r["Semáforo"] == "Verde") & (r["Autonomos"] >= 0.5),
            r["Semáforo"] == "Rojo",
        ],
        [
            "Digital: bajar frecuencia de visita, pedido por Tuali",
            "Acompañamiento: mantener visita + capacitación en Tuali",
        ],
        default="Impulso: visitas de capacitación a semi autónomos",
    )
    return r.sort_values("Dependientes", ascending=False)


def rutas_recomendadas(d: pd.DataFrame, capacidad: int) -> pd.DataFrame:
    """Reorganiza cada CEDIS: autónomos → 1 ruta digital; semi y dependientes → rutas presenciales
    de 'capacidad' clientes, agrupados por barrido angular alrededor del CEDIS (sweep)."""
    d = d.copy()
    d["ANG"] = np.arctan2(d["LAT"] - d["CEDIS_LAT"], d["LON"] - d["CEDIS_LON"])
    d["RUTA_REC"] = ""
    for cedis, g in d.groupby("CEDISID"):
        d.loc[g.index[g["SEGMENT"] == "Autónomo"], "RUTA_REC"] = f"{cedis} · Digital"
        presenciales = g[g["SEGMENT"] != "Autónomo"].sort_values("ANG").index
        for k, i in enumerate(range(0, len(presenciales), capacidad)):
            d.loc[presenciales[i:i + capacidad], "RUTA_REC"] = f"{cedis} · P{k + 1}"
    return d


def mapa_rutas(d: pd.DataFrame, col_ruta: str) -> go.Figure:
    """Dibuja cada ruta como recorrido CEDIS → clientes → CEDIS (rutas digitales solo como puntos)."""
    d = d.copy()
    d["ANG"] = np.arctan2(d["LAT"] - d["CEDIS_LAT"], d["LON"] - d["CEDIS_LON"])
    paleta = px.colors.qualitative.Bold + px.colors.qualitative.Safe
    fig = go.Figure()
    for i, (ruta, g) in enumerate(d.groupby(col_ruta)):
        g = g.sort_values("ANG")
        color = paleta[i % len(paleta)]
        if not str(ruta).endswith("Digital"):
            clat, clon = g["CEDIS_LAT"].iloc[0], g["CEDIS_LON"].iloc[0]
            fig.add_trace(go.Scattermap(
                lat=[clat] + g["LAT"].tolist() + [clat], lon=[clon] + g["LON"].tolist() + [clon],
                mode="lines", line=dict(width=3, color=color), name=str(ruta), legendgroup=str(ruta),
                hoverinfo="skip",
            ))
        fig.add_trace(go.Scattermap(
            lat=g["LAT"], lon=g["LON"], mode="markers", legendgroup=str(ruta),
            showlegend=str(ruta).endswith("Digital"), name=str(ruta),
            marker=dict(size=13, color=g["SEGMENT"].map(COLORES)),
            text=g["CLIENTE"] + "<br>" + g["SEGMENT"] + "<br>Ruta: " + str(ruta),
            hovertemplate="%{text}<extra></extra>",
        ))
    cedis = d.drop_duplicates("CEDISID")
    fig.add_trace(go.Scattermap(
        lat=cedis["CEDIS_LAT"], lon=cedis["CEDIS_LON"], mode="markers+text", name="CEDIS",
        marker=dict(size=18, color="#222"), text=cedis["CEDISID"], textposition="top right",
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        map=dict(style="carto-positron", center=dict(lat=d["LAT"].mean(), lon=d["LON"].mean()), zoom=9.5),
        margin=dict(t=0, b=0, l=0, r=0), height=520, legend=dict(title="Rutas"),
    )
    return fig


# ──────────────────────────────────────────────────────────────
# Lista de trabajo: riesgo, protocolo y acción por tienda
# ──────────────────────────────────────────────────────────────
def lista_trabajo(d: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    meta = base[base["SEGMENT"] == "Autónomo"][[f"P_{v}" for v in VARS]].mean()
    brechas = pd.DataFrame({v: meta[f"P_{v}"] - d[f"P_{v}"] for v in VARS})
    d["BRECHA_PRINCIPAL"] = brechas.idxmax(axis=1)

    no_verde = d["SEMAFORO"] != "Verde"
    d["RIESGO"] = np.select(
        [
            no_verde & (d["COUPLED"] >= 0.40),
            no_verde & ((d["COUPLED"] >= 0.25) | (d["PROB_MAX"] < 0.99)),
        ],
        ["Alerta", "Vigilar"],
        default="—",
    )
    d["PROTOCOLO"] = np.where(
        no_verde,
        d["BRECHA_PRINCIPAL"].map(PROTOCOLOS),
        "Auditoría de enfriador y exhibición (sin captura de pedido)",
    )
    d["ACCION_MES"] = d["SEMAFORO"].map({
        "Verde": "Beneficio por pedir en Tuali (programa de lealtad)",
        "Amarillo": "Recordatorio en su día de pedido",
        "Rojo": "Visita de acompañamiento + capacitación en Tuali",
    })
    orden_riesgo = d["RIESGO"].map({"Alerta": 0, "Vigilar": 1, "—": 2})
    orden_color = d["SEMAFORO"].map({"Rojo": 0, "Amarillo": 1, "Verde": 2})
    return d.assign(_r=orden_riesgo, _c=orden_color).sort_values(["_r", "_c", "IAD"]).drop(columns=["_r", "_c"])


# ──────────────────────────────────────────────────────────────
# Carga
# ──────────────────────────────────────────────────────────────
st.sidebar.header("Datos")
subido = st.sidebar.file_uploader("Excel del modelo (opcional)", type=["xlsx"])
if subido is not None:
    df = cargar_datos(subido)
elif ARCHIVO.exists():
    df = cargar_datos(ARCHIVO)
else:
    st.error(f"No encontré **{ARCHIVO.name}**. Ponlo en la misma carpeta que app.py o súbelo en la barra lateral.")
    st.stop()

modelo = entrenar_modelo(df)
df = agregar_coordenadas(df)
df = calcular_indices(df, modelo["pesos"])
INVERTIDAS = variables_invertidas(df)

# ──────────────────────────────────────────────────────────────
# Filtros
# ──────────────────────────────────────────────────────────────
st.sidebar.header("Filtros")
filtros = {}
for etiqueta, col in DIMENSIONES.items():
    opciones = sorted(df[col].dropna().unique())
    filtros[col] = st.sidebar.multiselect(etiqueta, opciones, placeholder="Todos")

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

st.markdown(
    f"""
    <div class="banner">
      <div class="eyebrow">ARCA CONTINENTAL · DIGITAL NEST · EQUIPO 5</div>
      <h1>Radar de Autonomía Tuali</h1>
      <p>{pct['Autónomo']:.1f}% de los clientes ya pide solo por Tuali; {pct['Dependiente']:.1f}% todavía depende del preventista.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

acople_seg = dff.groupby("SEGMENT")["COUPLED"].mean().reindex(ORDEN)
k1, k2, k3, k4, k5 = st.columns(5)
tarjeta(k1, "Verdes (autónomos)", f"{pct['Autónomo']:.1f}%", f"{conteo['Autónomo']} clientes", SEMAFORO["Verde"])
tarjeta(k2, "Amarillos (en transición)", f"{pct['Semi autónomo']:.1f}%", f"{conteo['Semi autónomo']} clientes", SEMAFORO["Amarillo"])
tarjeta(k3, "Rojos (dependientes)", f"{pct['Dependiente']:.1f}%", f"{conteo['Dependiente']} clientes", SEMAFORO["Rojo"])
tarjeta(k4, "IAD promedio", f"{dff['IAD'].mean():.1f}", "Índice de Autonomía Digital, de 0 a 10")
if acople_seg.notna().all():
    tarjeta(
        k5, "Validación: pedidos pegados a visita",
        f"+{(acople_seg['Dependiente'] - acople_seg['Autónomo']) * 100:.1f} pp",
        f"Rojos {acople_seg['Dependiente']:.0%} vs verdes {acople_seg['Autónomo']:.0%}",
    )
else:
    tarjeta(k5, "Pedidos pegados a visita", f"{dff['COUPLED'].mean():.0%}", "Promedio de la vista")

(tab_res, tab_mapa, tab_ficha, tab_lista, tab_radar, tab_sim, tab_esc, tab_met) = st.tabs([
    "📊 Panorama", "🗺️ Mapa y rutas", "🪪 Ficha del cliente", "📋 Lista de trabajo",
    "🕸️ Radar de perfiles", "🧪 Simulador de cliente", "🚀 Escenarios e impacto", "📐 Método",
])

# ──────────────────────────────────────────────────────────────
# 1. Panorama
# ──────────────────────────────────────────────────────────────
with tab_res:
    st.subheader("Distribución del Índice de Autonomía Digital (IAD)")
    st.caption("Clientes por puntaje de 0 a 10, apilados por nivel del GMM.")
    fig = px.histogram(
        dff, x="IAD", color="SEGMENT", nbins=40, color_discrete_map=COLORES,
        category_orders={"SEGMENT": ORDEN}, labels={"SEGMENT": "Nivel", "IAD": "IAD"},
    )
    fig.update_layout(barmode="stack", bargap=0.05, xaxis=dict(range=[0, 10]), yaxis_title="Clientes",
                      legend=dict(orientation="h", y=1.1, title=None), height=360, margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.subheader("Distribución por nivel")
        fig = px.pie(names=conteo.index, values=conteo.values, hole=0.55, color=conteo.index,
                     color_discrete_map=COLORES)
        fig.update_traces(textinfo="percent+label", sort=False)
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10), height=360)
        st.plotly_chart(fig, width="stretch")

    with c2:
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
                          margin=dict(t=30, b=10), height=max(320, 22 * len(orden_dim)))
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Validación: dependencia del preventista")
        fig = px.box(dff, x="SEGMENT", y="COUPLED", color="SEGMENT", color_discrete_map=COLORES,
                     category_orders={"SEGMENT": ORDEN},
                     labels={"SEGMENT": "Nivel", "COUPLED": "Pedidos acoplados a visita"}, points="outliers")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%", margin=dict(t=10), height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("COUPLED no se usó para entrenar el modelo. Que crezca de Autónomo → Dependiente confirma que "
                   "los niveles reflejan distintos grados de dependencia del preventista.")

    with c4:
        st.subheader("Clientes en el espacio PCA")
        comp = pd.DataFrame(modelo["componentes"], columns=["PC1", "PC2", "PC3"], index=df.index)
        comp = comp.loc[dff.index].join(dff[["SEGMENT", "CLIENTE", "PROB_MAX"]])
        ve = modelo["var_explicada"]
        fig = px.scatter_3d(comp, x="PC1", y="PC2", z="PC3", color="SEGMENT", color_discrete_map=COLORES,
                            category_orders={"SEGMENT": ORDEN}, hover_name="CLIENTE",
                            hover_data={"PROB_MAX": ":.3f"},
                            labels={"PC1": f"PC1 ({ve[0]:.0%})", "PC2": f"PC2 ({ve[1]:.0%})",
                                    "PC3": f"PC3 ({ve[2]:.0%})", "SEGMENT": "Nivel"})
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(legend=dict(orientation="h", y=1.02, title=None), margin=dict(t=10, b=0), height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Los 3 componentes explican el {ve.sum():.1%} de la varianza.")

# ──────────────────────────────────────────────────────────────
# 2. Mapa y rutas
# ──────────────────────────────────────────────────────────────
with tab_mapa:
    st.subheader("Semáforo en mapa")
    fig = px.scatter_map(
        dff, lat="LAT", lon="LON", color="SEGMENT", color_discrete_map=COLORES,
        category_orders={"SEGMENT": ORDEN}, hover_name="CLIENTE",
        hover_data={"ROUTEID": True, "CEDISID": True, "IAD": ":.1f", "LAT": False, "LON": False},
        labels={"SEGMENT": "Nivel"}, zoom=4.3, center=dict(lat=26.5, lon=-106), height=520,
        map_style="carto-positron",
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8))
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), legend=dict(orientation="h", y=1.05, title=None))
    st.plotly_chart(fig, width="stretch")
    if not {"LATITUDE", "LONGITUDE"}.issubset(df.columns):
        st.caption("Ubicación aproximada: la base no trae coordenadas, así que cada cliente se ubica cerca de su CEDIS y su ruta.")

    st.divider()

    st.subheader("¿Dónde están los dependientes? · Rutas actuales")
    rutas = resumen_rutas(dff)
    s1, s2, s3 = st.columns(3)
    for col, color in zip([s1, s2, s3], ["Rojo", "Amarillo", "Verde"]):
        tarjeta(col, f"Rutas en {color.lower()}", int((rutas["Semáforo"] == color).sum()),
                {"Rojo": "más de 30% dependientes", "Amarillo": "15% a 30% dependientes",
                 "Verde": "menos de 15% dependientes"}[color], SEMAFORO[color])

    def color_fila(fila):
        return [f"background-color: {SEMAFORO[fila['Semáforo']]}44"] * len(fila)

    st.dataframe(
        rutas.style.apply(color_fila, axis=1).format(
            {"Autonomos": "{:.0%}", "Semi": "{:.0%}", "Dependientes": "{:.0%}", "IAD": "{:.1f}", "Acoplamiento": "{:.1%}"}
        ),
        width="stretch", hide_index=True, height=320,
    )

    st.divider()

    st.subheader("Rutas recomendadas")
    st.write(
        "Propuesta: en cada CEDIS, los **autónomos** pasan a una ruta **digital** (piden por Tuali y se visitan "
        "con menos frecuencia), y los **semi autónomos y dependientes** se reagrupan en rutas presenciales por cercanía."
    )
    p1, p2 = st.columns(2)
    capacidad = p1.slider("Clientes por ruta presencial", 4, 20, 8)
    semanas_aut = p2.slider("Autónomos: una visita cada … semanas", 1, 8, 4)

    rec = rutas_recomendadas(dff, capacidad)
    n_aut = (rec["SEGMENT"] == "Autónomo").sum()
    rutas_pres_rec = rec.loc[~rec["RUTA_REC"].str.endswith("Digital"), "RUTA_REC"].nunique()
    rutas_hoy = dff["ROUTEID"].nunique()
    visitas_actual = len(rec)  # supuesto: hoy cada cliente se visita 1 vez por semana
    visitas_rec = (len(rec) - n_aut) + n_aut / semanas_aut

    r1, r2, r3, r4 = st.columns(4)
    tarjeta(r1, "Rutas presenciales", rutas_pres_rec, f"{rutas_pres_rec - rutas_hoy:+d} vs. hoy ({rutas_hoy})")
    tarjeta(r2, "Rutas digitales", rec.loc[rec["RUTA_REC"].str.endswith("Digital"), "RUTA_REC"].nunique(),
            f"{n_aut} clientes autónomos", SEMAFORO["Verde"])
    tarjeta(r3, "Visitas por semana", f"{visitas_rec:,.0f}", f"{visitas_rec - visitas_actual:+,.0f} vs. hoy ({visitas_actual})")
    tarjeta(r4, "Ahorro de visitas", f"{1 - visitas_rec / visitas_actual:.0%}", "Supuesto: hoy 1 visita por semana", SEMAFORO["Verde"])

    st.markdown("**Ver rutas en el mapa**")
    m1, m2 = st.columns([1, 1])
    cedis_sel = m1.selectbox("CEDIS", sorted(dff["CEDISID"].unique()), key="mapa_cedis")
    vista = m2.radio("Mostrar", ["Rutas actuales", "Rutas recomendadas"], horizontal=True, key="mapa_vista")

    d_cedis = rec[rec["CEDISID"] == cedis_sel]
    col_ruta = "ROUTEID" if vista == "Rutas actuales" else "RUTA_REC"
    c_map, c_tab = st.columns([1.6, 1])
    with c_map:
        st.plotly_chart(mapa_rutas(d_cedis, col_ruta), width="stretch")
    with c_tab:
        detalle = (d_cedis.groupby(col_ruta)["SEGMENT"].value_counts().unstack(fill_value=0)
                   .reindex(columns=ORDEN, fill_value=0))
        detalle["Total"] = detalle.sum(axis=1)
        st.markdown(f"**{vista} en {cedis_sel}**")
        st.dataframe(detalle, width="stretch")
        st.caption("Los puntos se colorean por nivel; las líneas muestran el recorrido CEDIS → clientes → CEDIS.")

    st.download_button(
        "Descargar asignación de rutas recomendadas (CSV)",
        rec[["CUSTOMERID", "BUSINESSNAME", "SEGMENT", "CEDISID", "ROUTEID", "RUTA_REC"]]
        .rename(columns={"ROUTEID": "RUTA_ACTUAL", "RUTA_REC": "RUTA_RECOMENDADA"})
        .to_csv(index=False).encode("utf-8-sig"),
        file_name="rutas_recomendadas.csv", mime="text/csv",
    )

# ──────────────────────────────────────────────────────────────
# 3. Ficha del cliente
# ──────────────────────────────────────────────────────────────
trabajo = lista_trabajo(dff, df)

with tab_ficha:
    cliente = st.selectbox("Buscar tienda o ID", trabajo["CLIENTE"].tolist(), key="ficha_cliente",
                           format_func=lambda c: f"{c} · {EMOJI[trabajo.set_index('CLIENTE').loc[c, 'SEMAFORO']]}")
    r = trabajo[trabajo["CLIENTE"] == cliente].iloc[0]
    proba = predecir(modelo, r[VARS].to_dict())
    color = r["SEMAFORO"]

    f1, f2 = st.columns(2)
    with f1:
        st.markdown(f"### {r['BUSINESSNAME']}")
        st.caption(f"Cliente {r['CUSTOMERID']} · {r['BUSINESSTYPE']} · {r['SUBCHANNEL']} · {r['STATE']}")
        st.markdown(
            f"<span style='font-size:.85rem;opacity:.7'>IAD</span><br>"
            f"<span style='font-size:2.6rem;font-family:Georgia,serif'>{r['IAD']:.1f}</span> &nbsp;"
            f"<span style='background:{SEMAFORO[color]};color:white;padding:3px 10px;border-radius:12px'>{color}</span>",
            unsafe_allow_html=True,
        )
        nivel_prom = df[df["SEGMENT"] == r["SEGMENT"]]["COUPLED"].mean()
        datos = pd.DataFrame({
            "Dato": ["Nivel", "Ruta", "CEDIS", "Tamaño", "Cliente desde", "Pedidos pegados a visita", "Riesgo"],
            "Valor": [r["SEGMENT"], r["ROUTEID"], r["CEDISID"], r["CUSTOMERSIZE"], str(r["JOINDATE"])[:10],
                      f"{r['COUPLED']:.0%} (promedio de su nivel: {nivel_prom:.0%})", r["RIESGO"]],
        })
        st.dataframe(datos, hide_index=True, width="stretch")

    with f2:
        st.markdown("**Probabilidad de cada nivel (GMM)**")
        fig = px.bar(x=proba.values, y=[SEMAFORO_NIVEL[s] for s in proba.index], orientation="h",
                     color=proba.index, color_discrete_map=COLORES, text=[f"{p:.0%}" for p in proba.values])
        fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1], tickformat=".0%", title=None),
                          yaxis=dict(title=None, autorange="reversed"), height=170, margin=dict(t=5, b=5))
        st.plotly_chart(fig, width="stretch")
        st.markdown("**Acción recomendada**")
        st.markdown(f"<div class='accion'><b>{r['ACCION_MES']}</b><span>Acción del mes según su color</span></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='accion'><b>{r['PROTOCOLO']}</b><span>Protocolo en la visita del preventista</span></div>",
                    unsafe_allow_html=True)

    f3, f4 = st.columns(2)
    puntajes = {f"P_{v}": v for v in VARS}
    etiquetas = [NOMBRES[v] + (" (inv.)" if v in INVERTIDAS else "") for v in VARS]
    with f3:
        st.markdown("**Sus variables contra los promedios**")
        st.caption("Puntaje de 0 a 10 por percentil. Más afuera = más autónomo. (inv.) = escala invertida.")
        perf = {
            "Autónomo promedio": df[df["SEGMENT"] == "Autónomo"][list(puntajes)].mean().rename(puntajes),
            "Dependiente promedio": df[df["SEGMENT"] == "Dependiente"][list(puntajes)].mean().rename(puntajes),
            "Este cliente": r[list(puntajes)].rename(puntajes).astype(float),
        }
        st.plotly_chart(radar(perf, rango=(0, 10), etiquetas=etiquetas), width="stretch")

    with f4:
        st.markdown("**Qué le falta para subir**")
        st.caption("Las dos variables con mayor brecha contra el autónomo promedio.")
        meta = df[df["SEGMENT"] == "Autónomo"][list(puntajes)].mean()
        brechas = (meta - r[list(puntajes)].astype(float)).rename(puntajes).sort_values(ascending=False)
        brechas = brechas[brechas > 0.5].head(2)
        if brechas.empty:
            st.success("Ya está al nivel del autónomo promedio en todas sus variables.")
        for v, b in brechas.items():
            st.markdown(
                f"<div class='accion' style='border-left-color:{SEMAFORO['Amarillo']}'><b>{NOMBRES[v]}</b>"
                f"<span>Su puntaje: {r['P_' + v]:.1f} de 10 · autónomo promedio: {meta['P_' + v]:.1f} · "
                f"brecha: {b:.1f}<br>Qué hacer: {PROTOCOLOS[v]}</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("**Su IAD frente a su ruta**")
        ruta_df = df[df["ROUTEID"] == r["ROUTEID"]].sort_values("IAD")
        fig = px.bar(ruta_df, x="IAD", y=ruta_df["BUSINESSNAME"].str.replace("Comercializadora ", ""),
                     orientation="h", color="SEGMENT", color_discrete_map=COLORES,
                     category_orders={"SEGMENT": ORDEN})
        fig.update_traces(marker_line_width=0)
        fig.update_layout(showlegend=False, yaxis_title=None, xaxis=dict(range=[0, 10]),
                          height=max(220, 24 * len(ruta_df)), margin=dict(t=5, b=5))
        idx = ruta_df["CLIENTE"].tolist().index(cliente)
        fig.add_annotation(x=r["IAD"], y=idx, text="◀ este cliente", showarrow=False, xanchor="left", xshift=6)
        st.plotly_chart(fig, width="stretch")

# ──────────────────────────────────────────────────────────────
# 4. Lista de trabajo
# ──────────────────────────────────────────────────────────────
with tab_lista:
    st.write("Herramienta para el supervisor y el preventista: qué hacer en cada tienda de la ruta y cuáles están en riesgo.")
    l1, l2, l3, l4 = st.columns(4)
    tarjeta(l1, "Alertas", int((trabajo["RIESGO"] == "Alerta").sum()),
            "No verdes con 40% o más de pedidos pegados a la visita", SEMAFORO["Rojo"])
    tarjeta(l2, "En vigilancia", int((trabajo["RIESGO"] == "Vigilar").sum()),
            "Pegados a la visita (25–40%) o en transición entre niveles", SEMAFORO["Amarillo"])
    tarjeta(l3, "Visitas que se transforman", int((trabajo["SEMAFORO"] == "Verde").sum()),
            "Tiendas verdes: auditoría en lugar de captura", SEMAFORO["Verde"])
    tarjeta(l4, "Tiendas en la vista", len(trabajo), "Usa la Ficha del cliente para ver el detalle")

    b1, b2 = st.columns([2, 1.3])
    busqueda = b1.text_input("Buscar tienda", placeholder="Nombre o ID…")
    filtro = b2.radio("Mostrar", ["Todas", "Solo riesgo", "Rojas", "Amarillas", "Verdes"], horizontal=True)

    vista_l = trabajo
    if busqueda:
        vista_l = vista_l[vista_l["CLIENTE"].str.contains(busqueda, case=False)]
    if filtro == "Solo riesgo":
        vista_l = vista_l[vista_l["RIESGO"] != "—"]
    elif filtro != "Todas":
        vista_l = vista_l[vista_l["SEMAFORO"] == {"Rojas": "Rojo", "Amarillas": "Amarillo", "Verdes": "Verde"}[filtro]]

    tabla_l = pd.DataFrame({
        "ID": vista_l["CUSTOMERID"],
        "Tienda": vista_l["BUSINESSNAME"],
        "Color": vista_l["SEMAFORO"].map(lambda c: f"{EMOJI[c]} {c}"),
        "IAD": vista_l["IAD"],
        "Riesgo": vista_l["RIESGO"].map({"Alerta": "🔴 Alerta", "Vigilar": "🟡 Vigilar", "—": "—"}),
        "Ruta": vista_l["ROUTEID"],
        "Pegados a visita": vista_l["COUPLED"],
        "Protocolo en visita": vista_l["PROTOCOLO"],
        "Acción del mes": vista_l["ACCION_MES"],
    })
    st.dataframe(
        tabla_l, hide_index=True, width="stretch", height=480,
        column_config={
            "IAD": st.column_config.NumberColumn(format="%.1f"),
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
            st.markdown(f"**{rr['BUSINESSNAME']}**")
            st.markdown(f"Nivel: **{rr['SEGMENT']}** · probabilidad {rr['PROB_MAX']:.1%}")
            st.markdown(f"{rr['SUBCHANNEL']} · {rr['CUSTOMERSIZE']} · {rr['STATE']}")

    with col_a:
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
    st.info("**Lectura rápida:** los autónomos usan más promociones y favoritos, compran más variedad y casi no "
            "buscan por código; los dependientes buscan todo por código, tienen sesiones cortas y piden más "
            "acompañados del preventista.")

# ──────────────────────────────────────────────────────────────
# 6. Simulador de cliente (what-if con el GMM)
# ──────────────────────────────────────────────────────────────
with tab_sim:
    st.subheader("¿Qué necesita cambiar un cliente para subir de nivel?")
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

    st.selectbox("Punto de partida", puntos_partida, key="sim_base", on_change=cargar_punto_partida)

    col_s, col_r = st.columns([1, 1.3])
    valores = {}
    with col_s:
        for v in VARS:
            mn, mx = df[v].min(), df[v].max()
            if v in VARS_ENTERAS:
                valores[v] = st.slider(NOMBRES[v], int(mn), int(mx), key=f"sim_{v}")
            elif v in VARS_TASA:
                valores[v] = st.slider(NOMBRES[v], 0.0, 1.0, step=0.01, key=f"sim_{v}", format="%.2f")
            else:
                valores[v] = st.slider(NOMBRES[v], float(mn), float(mx), step=10.0, key=f"sim_{v}", format="%.0f")

    proba_sim = predecir(modelo, valores)
    nivel = proba_sim.idxmax()

    with col_r:
        st.markdown(
            f"<div style='padding:14px 18px;border-radius:10px;background:{COLORES[nivel]}22;"
            f"border-left:6px solid {COLORES[nivel]}'>"
            f"<span style='font-size:0.9rem'>Nivel estimado</span><br>"
            f"<span style='font-size:1.8rem;font-weight:700'>{nivel}</span>"
            f"<span style='font-size:1rem'> &nbsp;· {proba_sim[nivel]:.1%} de probabilidad</span></div>",
            unsafe_allow_html=True,
        )
        fig = px.bar(x=proba_sim.values, y=proba_sim.index, orientation="h", color=proba_sim.index,
                     color_discrete_map=COLORES, text=[f"{p:.1%}" for p in proba_sim.values],
                     labels={"x": "Probabilidad", "y": ""})
        fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1], tickformat=".0%"), height=200,
                          yaxis=dict(autorange="reversed"), margin=dict(t=20, b=10))
        st.plotly_chart(fig, width="stretch")

        if proba_sim.max() < 0.70:
            st.warning("Probabilidad máxima menor a 70%: este cliente estaría **en transición** entre niveles.")

        sim_norm = normalizar(pd.DataFrame([valores]), df).iloc[0]
        perfiles_sim = {s: normalizar(df[df["SEGMENT"] == s][VARS].mean().to_frame().T, df).iloc[0] for s in ORDEN}
        perfiles_sim["Cliente simulado"] = sim_norm
        st.plotly_chart(radar(perfiles_sim), width="stretch")

# ──────────────────────────────────────────────────────────────
# 7. Escenarios e impacto
# ──────────────────────────────────────────────────────────────
with tab_esc:
    # 7a) Calculadora de impacto: horas de preventista y venta adicional
    st.subheader("Simulador de impacto")
    st.write("Horas de preventista que se liberan al dejar de capturar pedidos en tiendas autónomas, y venta "
             "adicional si ese tiempo se usa para subir de nivel a tiendas en transición.")

    n_aut_v = int(conteo["Autónomo"])
    n_semi_v = int(conteo["Semi autónomo"])
    i1, i2, i3 = st.columns(3)
    visitas_ano = i1.slider("Visitas al año por tienda autónoma", 4.0, 52.0, 16.5, 0.5)
    min_visita = i2.slider("Minutos por visita", 10.0, 60.0, 33.4, 0.1)
    reduccion = i3.slider("Reducción de visitas de captura (%)", 0, 100, 50, 5)
    i4, i5, i6 = st.columns(3)
    suben = i4.slider("Tiendas semi autónomas que suben de nivel (%)", 0, 100, 20, 5)
    pedidos_ano = i5.slider("Pedidos al año por tienda semi autónoma", 10.0, 120.0, 45.5, 0.5)
    aumento_ticket = i6.slider("Aumento de ticket por pedido ($)", 0, 1500, 445, 5)
    st.caption("Valores iniciales tomados del análisis con datos 2025 (ticket autónomo $1,739 vs semi $1,294). "
               "Ajústalos para probar escenarios.")

    horas_hoy = n_aut_v * visitas_ano * min_visita / 60
    horas_lib = horas_hoy * reduccion / 100
    jornadas = horas_lib / 8
    fte = jornadas / 264
    tiendas_suben = round(n_semi_v * suben / 100)
    venta_extra = tiendas_suben * pedidos_ano * aumento_ticket

    j1, j2, j3, j4, j5 = st.columns(5)
    tarjeta(j1, "Horas al año hoy en autónomos", f"{horas_hoy:,.0f} h",
            f"{n_aut_v} tiendas × {visitas_ano} visitas × {min_visita} min ÷ 60")
    tarjeta(j2, "Horas de preventista liberadas", f"+{horas_lib:,.0f} h", f"{horas_hoy:,.0f} h × {reduccion}% de reducción",
            SEMAFORO["Verde"])
    tarjeta(j3, "Equivale a", f"{jornadas:,.0f} jornadas", f"{horas_lib:,.0f} h ÷ 8 h · {fte:.2f} preventistas de tiempo completo",
            SEMAFORO["Verde"])
    tarjeta(j4, "Tiendas semi autónomas que suben", f"{tiendas_suben} tiendas", f"{n_semi_v} semi autónomas × {suben}%",
            SEMAFORO["Amarillo"])
    tarjeta(j5, "Venta adicional estimada al año", f"${venta_extra:,.0f}",
            f"{tiendas_suben} tiendas × {pedidos_ano} pedidos × ${aumento_ticket}")

    st.markdown("**Texto para el reporte**")
    texto = (
        f"Con la frecuencia de visitas actual, las {n_aut_v} tiendas autónomas consumen cerca de {horas_hoy:,.0f} horas "
        f"de preventista al año ({visitas_ano} visitas de {min_visita} minutos por tienda). Reducir en {reduccion}% las "
        f"visitas dedicadas a capturar pedidos liberaría {horas_lib:,.0f} horas anuales, equivalentes a {jornadas:,.0f} "
        f"jornadas de trabajo o {fte:.2f} preventistas de tiempo completo, sin dejar de atender a estas tiendas: la visita "
        f"cambia de captura a auditoría de enfriador y exhibición. Si ese tiempo se reasigna a acompañar a las tiendas semi "
        f"autónomas y el {suben}% de ellas ({tiendas_suben} tiendas) alcanza el ticket promedio de una tienda autónoma "
        f"(+${aumento_ticket} por pedido en {pedidos_ano} pedidos al año), Arca capturaría una venta adicional estimada de "
        f"${venta_extra:,.0f} anuales. Esta cifra es un escenario: supone que la capacitación causa el aumento de ticket y "
        f"debe confirmarse con una prueba piloto contra un grupo de control."
    )
    st.info(texto)

    st.divider()

    # 7b) Migración entre niveles
    st.subheader("¿Qué pasa si movemos clientes de nivel?")
    st.write("Define qué porcentaje de cada nivel logra subir uno con las estrategias de adopción. "
             "El impacto se estima con el acoplamiento promedio observado en cada nivel.")

    e1, e2, e3 = st.columns(3)
    mig_dep = e1.slider("Dependientes que pasan a Semi autónomo (%)", 0, 100, 30, step=5)
    mig_semi = e2.slider("Semi autónomos que pasan a Autónomo (%)", 0, 100, 20, step=5)
    pedidos_mes = e3.number_input("Pedidos por cliente al mes (supuesto)", min_value=1, max_value=60, value=4)

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
    m1.metric("Clientes que suben de nivel", f"{pasan_dep + pasan_semi:,.0f}")
    m2.metric("Autónomos", f"{nuevo['Autónomo'] / nuevo.sum():.1%}",
              f"{(nuevo['Autónomo'] - actual['Autónomo']) / actual.sum() * 100:+.1f} pp")
    m3.metric("Pedidos acoplados a visita", f"{acople_nuevo:.1%}", f"{(acople_nuevo - acople_actual) * 100:+.1f} pp",
              delta_color="inverse")
    m4.metric("Pedidos/mes sin depender de visita", f"+{pedidos_liberados:,.0f}")

    comp_esc = pd.DataFrame({"Actual": actual, "Escenario": nuevo}).rename_axis("Nivel").reset_index()
    comp_esc = comp_esc.melt(id_vars="Nivel", var_name="Situación", value_name="Clientes")
    fig = px.bar(comp_esc, x="Situación", y="Clientes", color="Nivel", color_discrete_map=COLORES,
                 category_orders={"Nivel": ORDEN}, text=comp_esc["Clientes"].round(0).astype(int))
    fig.update_layout(barmode="stack", height=420, legend=dict(orientation="h", y=1.08, title=None))
    st.plotly_chart(fig, width="stretch")

# ──────────────────────────────────────────────────────────────
# 8. Método
# ──────────────────────────────────────────────────────────────
with tab_met:
    pesos = modelo["pesos"].sort_values()
    ve = modelo["var_explicada"]
    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Peso de cada variable en el IAD")
        st.caption("Cargas al cuadrado del primer componente principal (PCA). Suman 100%.")
        fig = px.bar(x=pesos.values, y=[NOMBRES[v] for v in pesos.index], orientation="h",
                     text=[f"{p:.0%}" for p in pesos.values], color_discrete_sequence=[ROJO_ARCA])
        fig.update_layout(xaxis=dict(tickformat=".0%", title=None), yaxis_title=None, height=380, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    with t2:
        kw = stats.kruskal(*[df.loc[df["SEGMENT"] == n, "COUPLED"] for n in ORDEN])
        st.subheader("Cómo se calcula")
        st.markdown(
            f"1. **Variables.** {', '.join(NOMBRES[v] for v in VARS)}.\n"
            f"2. **Estandarización** de las 7 variables (media 0, desviación 1).\n"
            f"3. **PCA** con 3 componentes: explican el **{ve.sum():.1%}** de la variación "
            f"(el primero solo, {ve[0]:.1%}).\n"
            f"4. **Niveles.** Gaussian Mixture Model de 3 componentes (covarianza *full*, reg_covar = 0.01, "
            f"n_init = 10). Silueta: **{modelo['silueta']:.3f}** · BIC: **{modelo['gmm'].bic(modelo['componentes']):.1f}**.\n"
            f"5. **IAD (0 a 10).** Cada variable se califica por percentil (orientada para que más alto = más autónomo) "
            f"y se pondera con las cargas al cuadrado del primer componente.\n"
            f"6. **Validación.** El acoplamiento con visitas nunca entra al modelo y crece de verde a rojo "
            f"(Kruskal-Wallis H = {kw.statistic:.1f}, p = {kw.pvalue:.1e}). El modelo reentrenado coincide en "
            f"**{modelo['coincidencia']:.1%}** con el Excel."
        )

    st.subheader("Variables que se probaron y se dejaron fuera")
    st.markdown(
        "- **COUPLED**: se reservó para validar el modelo.\n"
        "- **DISTINCTAPPVERSIONS**: casi no varía entre clientes.\n"
        "- **SEARCHCOUNT**: mide tiempo de uso más que autonomía.\n"
        "- **Minutos por tipo de visita** (seguimiento, cobranza, programada, capacitación): se resumen en la duración de visita.\n"
        "- **Duración de visita, dispersión de ubicación y pago electrónico**: valores muy parecidos en los tres niveles, "
        "no ayudan a clasificar."
    )