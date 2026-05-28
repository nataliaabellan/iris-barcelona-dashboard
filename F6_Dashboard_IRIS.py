# ===================================================================================
# Clusterització de peticions ciutadanes — Dashboard TFG Natàlia Abellan Barron
# ===================================================================================

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="IRIS Barcelona — Peticions ciutadanes",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG = "rgba(0,0,0,0)"

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #EAECEF;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h2, h3 { color: #2E75B6; margin-top: 2rem; }
h1     { color: #1F3864; }
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    border: 1px solid #E6ECF2;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.metric-val { font-size: 2.3rem; font-weight: 800; color: #1F3864; }
.metric-lab { font-size: 0.85rem; color: #777; margin-top: 4px; }
.alerta     { border-top: 3px solid #C00000 !important; }
.insight {
    background: transparent;
    padding: 0;
    margin-top: 4px;
    color: #666;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
TERRIT_DIR = DATA_DIR / "territorial"
CLUST_DIR  = DATA_DIR / "clustering"

ETIQUETES = {
    0:  "Manteniment espai urbà",
    1:  "Serveis socials",
    2:  "Cultura i oci",
    3:  "Tràmits en línia",
    4:  "Mobilitat",
    5:  "Seguretat i convivència",
    6:  "Plagues i salut pública",
    7:  "Urbanisme i habitatge",
    8:  "Atenció ciutadana",
    9:  "Neteja urbana",
    10: "Equipaments culturals",
    11: "Esports",
    12: "Educació i joventut",
    13: "Hisenda i fiscalitat",
    14: "Transports públics",
}

COLOR_SEC    = "#2E75B6"
COLOR_ACCENT = "#C00000"
FONT         = "Inter"


def fig_layout(fig, h=None):
    upd = dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT, size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    if h:
        upd["height"] = h
    fig.update_layout(**upd)
    return fig


@st.cache_data
def load_geojson():
    candidate = DATA_DIR / "districtes_barcelona.geojson"
    if candidate.exists():
        with open(candidate, encoding="utf-8") as f:
            gj = json.load(f)
        for feat in gj["features"]:
            props = feat["properties"]
            if "NOM" in props and "nom" not in props:
                props["nom"] = props["NOM"]
        return gj
    raise FileNotFoundError("GeoJSON de districtes no trobat")


@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_parquet(DATA_DIR / "iris_clustered.parquet",
                         columns=["id", "districte", "barri", "tipus",
                                  "dies_resolucio", "cluster_kmeans",
                                  "data_entrada", "area"])
    df["data_entrada"]     = pd.to_datetime(df["data_entrada"], errors="coerce")
    df["any"]              = df["data_entrada"].dt.year.astype("Int16")
    df["mes"]              = df["data_entrada"].dt.month.astype("Int8")
    df["cluster_etiqueta"] = df["cluster_kmeans"].map(ETIQUETES).astype("category")
    df["tipus"]            = df["tipus"].astype("category")
    df["area"]             = df["area"].astype("category")
    df["dies_resolucio"]   = df["dies_resolucio"].astype("float32")
    df["districte"] = df["districte"].fillna("No especificat").astype(str)
    df.loc[df["districte"].str.lower().isin(["nan", "none", ""]), "districte"] = "No especificat"
    df["districte"] = df["districte"].astype("category")
    df["barri"] = df["barri"].fillna("No especificat").astype(str)
    df.loc[df["barri"].str.lower().isin(["nan", "none", ""]), "barri"] = "No especificat"
    df["barri"] = df["barri"].astype("category")
    return df


@st.cache_data(ttl=3600)
def load_stats():
    dist  = pd.read_parquet(TERRIT_DIR / "estadistiques_districte.parquet")
    barri = pd.read_parquet(TERRIT_DIR / "estadistiques_barri.parquet")
    equit = pd.read_parquet(TERRIT_DIR / "equitat_ivu.parquet")
    perf  = pd.read_parquet(CLUST_DIR  / "cluster_profiles.parquet")
    return dist, barri, equit, perf


@st.cache_data(ttl=3600)
def load_socio():
    path = TERRIT_DIR / "indicadors_socioeconomics.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        for col in df.select_dtypes(include="float64").columns:
            df[col] = df[col].astype("float32")
        return df
    return None


df = load_data()
dist_stats, barri_stats, equitat, perfils = load_stats()
geojson  = load_geojson()
df_socio = load_socio()

DISTRICTES = sorted([d for d in df["districte"].unique() if d != "No especificat"])
CLUSTERS   = sorted(df["cluster_kmeans"].unique().tolist())
TIPUS_LIST = sorted(df["tipus"].unique().tolist())

# ── Sidebar
with st.sidebar:
    st.markdown("---")
    pagina = st.radio(
        "Navegació",
        ["Visió general", "Explora per districte",
         "Anàlisi temàtica", "Equitat territorial"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Filtres**")
    anys_disp        = sorted(df["any"].dropna().unique().astype(int).tolist())
    any_min, any_max = st.select_slider(
        "Període", options=anys_disp,
        value=(min(anys_disp), max(anys_disp)),
    )
    tipus_sel = st.multiselect(
        "Tipus de petició", options=TIPUS_LIST, default=TIPUS_LIST,
    )

_mask = df["any"].between(any_min, any_max) & df["tipus"].isin(
    tipus_sel if tipus_sel else TIPUS_LIST
)
df_f = df.loc[_mask].copy()
del _mask


# ================================================================ PÀGINA 1
if pagina == "Visió general":
    st.title("Comunicacions ciutadanes · Barcelona")
    st.caption(
        f"Sistema IRIS de l'Ajuntament de Barcelona · {any_min}–{any_max}"
    )

    n_districtes = df_f[df_f["districte"] != "No especificat"]["districte"].nunique()
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, f"{len(df_f):,}",                             "Total de peticions",         False),
        (c2, f"{n_districtes}",                            "Districtes",                 False),
        (c3, f"{df_f['cluster_kmeans'].nunique()}",        "Grups temàtics",             False),
        (c4, f"{df_f['dies_resolucio'].median():.0f} dies","Mediana de resolució",       False),
    ]
    for col, val, lab, alerta in kpis:
        cls = "metric-card alerta" if alerta else "metric-card"
        col.markdown(
            f'<div class="{cls}"><div class="metric-val">{val}</div>'
            f'<div class="metric-lab">{lab}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col_a, col_b = st.columns([4, 2])

    with col_a:
        st.subheader("Distribució territorial")
        metrica_mapa = st.selectbox(
            "Mètrica",
            ["Nombre de peticions", "Temps màxim de resolució (P95)", "% Incidències"],
            key="metrica_mapa",
        )
        col_map = {
            "Nombre de peticions":            "n",
            "Temps màxim de resolució (P95)": "dies_p95",
            "% Incidències":                  "pct_incid",
        }[metrica_mapa]

        vol_dist = (
            df_f[df_f["districte"] != "No especificat"]
            .groupby("districte")
            .agg(
                n         = ("id",            "count"),
                dies_p95  = ("dies_resolucio", lambda x: x.quantile(0.95)),
                pct_incid = ("tipus",          lambda x: (x == "INCIDENCIA").mean() * 100),
            )
            .reset_index()
        )
        fig_mapa = px.choropleth_mapbox(
            vol_dist, geojson=geojson,
            locations="districte", featureidkey="properties.nom",
            color=col_map, color_continuous_scale="Blues",
            mapbox_style="carto-positron",
            center={"lat": 41.3851, "lon": 2.1734}, zoom=10.5,
            opacity=0.75, labels={col_map: metrica_mapa},
            hover_name="districte", hover_data={col_map: ":.1f"}, height=480,
        )
        fig_mapa.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor=BG,
            coloraxis_colorbar=dict(title=metrica_mapa, thickness=12, len=0.5),
        )
        st.plotly_chart(fig_mapa, use_container_width=True)
        st.caption("La intensitat del color representa el valor de la mètrica seleccionada.")

    with col_b:
        st.subheader("Distribució temàtica")
        cl_dist = df_f["cluster_etiqueta"].value_counts().reset_index()
        cl_dist.columns = ["cluster", "n"]
        fig_donut = go.Figure(go.Pie(
            labels=cl_dist["cluster"], values=cl_dist["n"],
            hole=0.45, textposition="inside", textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value:,} peticions<br>%{percent}<extra></extra>",
            marker=dict(colors=px.colors.qualitative.Set3,
                        line=dict(color="white", width=1.5)),
        ))
        fig_donut.update_layout(
            height=480, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor=BG, showlegend=True,
            legend=dict(font=dict(size=9, family=FONT), orientation="v"),
            font=dict(family=FONT),
            annotations=[dict(
                text=f"<b>{len(df_f):,}</b><br>peticions",
                x=0.5, y=0.5, font=dict(size=11, color="#1F3864"), showarrow=False,
            )],
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.caption("Neteja urbana i Manteniment concentren prop del 50% de les peticions.")

    st.markdown("---")
    st.subheader("Especialització temàtica per districte")
    st.caption("Distribució percentual de les peticions per grup temàtic dins de cada districte.")

    df_geo_cl = df_f[df_f["districte"] != "No especificat"].copy()
    ct_pct = pd.crosstab(
        df_geo_cl["districte"], df_geo_cl["cluster_etiqueta"], normalize="index",
    ) * 100
    fig_heat = px.imshow(
        ct_pct, color_continuous_scale="YlOrRd", aspect="auto",
        text_auto=".1f",
        labels=dict(x="Grup temàtic", y="Districte", color="%"), height=380,
    )
    fig_heat.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor=BG,
        font=dict(family=FONT),
        xaxis_tickangle=-35, xaxis_tickfont=dict(size=9),
        coloraxis_colorbar=dict(title="%", thickness=10),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    st.subheader("Evolució temporal")
    st.caption("Tendència de creixement sostingut amb estacionalitat anual (pic a l'octubre, vall a l'agost).")

    df_mes = (
        df_f.groupby([pd.Grouper(key="data_entrada", freq="ME"), "tipus"])
        .size().reset_index(name="n")
    )
    fig_temp = px.line(
        df_mes, x="data_entrada", y="n", color="tipus",
        color_discrete_map={
            "INCIDENCIA": "#C00000", "CONSULTA": "#2E75B6",
            "QUEIXA": "#ED7D31", "SUGGERIMENT": "#548235",
            "PETICIO DE SERVEI": "#1F3864", "AGRAIMENT": "#7030A0",
        },
        labels={"data_entrada": "Data", "n": "Peticions", "tipus": "Tipus"},
        height=300,
    )
    fig_temp.update_traces(line=dict(width=2))
    fig_temp.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT), hovermode="x unified",
    )
    st.plotly_chart(fig_temp, use_container_width=True)


# ================================================================ PÀGINA 2
elif pagina == "Explora per districte":
    st.title("Explora per districte")
    st.caption("Anàlisi de les principals tipologies de petició, distribució i evolució temporal per districte.")

    districte_sel = st.selectbox("Districte", DISTRICTES)
    df_d = df_f[df_f["districte"] == districte_sel]

    if len(df_d) == 0:
        st.warning("No hi ha dades per a aquest districte amb els filtres actuals.")
        st.stop()

    p95_dist = df_d["dies_resolucio"].quantile(0.95)
    c1, c2, c3 = st.columns(3)
    c1.metric("Peticions",          f"{len(df_d):,}")
    c2.metric("Mediana de resolució", f"{df_d['dies_resolucio'].median():.0f} dies")
    c3.metric("P95 de resolució",   f"{p95_dist:.0f} dies",
              help="El 95% de les peticions es resolen en menys d'aquest nombre de dies.")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Tipologies principals")
        top_cl = (
            df_d.groupby("cluster_etiqueta").size()
            .reset_index(name="n").sort_values("n", ascending=True).tail(10)
        )
        fig_cl = px.bar(
            top_cl, x="n", y="cluster_etiqueta", orientation="h",
            color="n", color_continuous_scale="Blues",
            range_color=[0, top_cl["n"].max()],
            labels={"n": "Peticions", "cluster_etiqueta": ""}, height=380,
        )
        fig_cl.update_coloraxes(cmin=top_cl["n"].min() * 0.3)
        fig_cl.update_layout(
            coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT),
        )
        st.plotly_chart(fig_cl, use_container_width=True)
        st.caption("Volum de peticions per grup temàtic.")

    with col_b:
        st.subheader("Tipus de petició")
        tipus_dist = df_d["tipus"].value_counts().reset_index()
        tipus_dist.columns = ["tipus", "n"]
        fig_t = go.Figure(go.Pie(
            labels=tipus_dist["tipus"], values=tipus_dist["n"],
            hole=0.40, textposition="inside", textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{value:,} peticions<br>%{percent}<extra></extra>",
            marker=dict(colors=px.colors.qualitative.Pastel,
                        line=dict(color="white", width=1.5)),
        ))
        fig_t.update_layout(
            height=360, showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor=BG, font=dict(family=FONT),
        )
        st.plotly_chart(fig_t, use_container_width=True)
        st.caption("Distribució per tipus de comunicació ciutadana.")

    st.subheader("Evolució mensual")
    df_d_mes = (
        df_d.groupby(pd.Grouper(key="data_entrada", freq="ME"))
        .size().reset_index(name="n")
        .rename(columns={"data_entrada": "data"})
    )
    df_d_mes["MM12"] = df_d_mes["n"].rolling(12, center=True, min_periods=1).mean()

    fig_ev = go.Figure()
    fig_ev.add_trace(go.Scatter(
        x=df_d_mes["data"], y=df_d_mes["n"],
        mode="lines", name="Peticions/mes",
        line=dict(color=COLOR_SEC, width=1.2), opacity=0.4,
    ))
    fig_ev.add_trace(go.Scatter(
        x=df_d_mes["data"], y=df_d_mes["MM12"],
        mode="lines", name="Tendència (MM12)",
        line=dict(color=COLOR_SEC, width=2.5),
    ))
    fig_ev.update_layout(
        height=280, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT), hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig_ev, use_container_width=True)

    st.subheader("Barris per volum de peticions")
    top_barris = (
        df_d[df_d["barri"] != "No especificat"]
        .groupby("barri").size().reset_index(name="n")
        .sort_values("n", ascending=False).head(10)
    )
    st.dataframe(
        top_barris.rename(columns={"barri": "Barri", "n": "Peticions"}),
        use_container_width=True, hide_index=True,
    )


# ================================================================ PÀGINA 3
elif pagina == "Anàlisi temàtica":
    st.title("Anàlisi temàtica")
    st.caption("Distribució territorial, temps de resolució i evolució temporal per grup temàtic.")

    cluster_id = st.selectbox(
        "Grup temàtic",
        options=CLUSTERS,
        format_func=lambda c: f"C{c}: {ETIQUETES.get(c, str(c))}",
    )
    df_c = df_f[df_f["cluster_kmeans"] == cluster_id]

    if len(df_c) == 0:
        st.warning("No hi ha dades per a aquest grup temàtic amb els filtres actuals.")
        st.stop()

    st.subheader(f"C{cluster_id}: {ETIQUETES.get(cluster_id, '')}")

    p95_cluster = df_c["dies_resolucio"].quantile(0.95)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peticions",          f"{len(df_c):,}")
    c2.metric("Districtes",          f"{df_c[df_c['districte'] != 'No especificat']['districte'].nunique()}")
    c3.metric("Mediana de resolució", f"{df_c['dies_resolucio'].median():.0f} dies")
    c4.metric("P95 de resolució",    f"{p95_cluster:.0f} dies",
              help="El 95% de les peticions es resolen en menys d'aquest nombre de dies.")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribució per districte")
        vol_d = (
            df_c[df_c["districte"] != "No especificat"]
            .groupby("districte").size().reset_index(name="n")
            .sort_values("n", ascending=True)
        )
        fig_d = px.bar(
            vol_d, x="n", y="districte", orientation="h",
            color="n", color_continuous_scale="Blues",
            labels={"n": "Peticions", "districte": ""}, height=360,
        )
        fig_d.update_layout(
            coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT),
        )
        st.plotly_chart(fig_d, use_container_width=True)
        st.caption("Volum de peticions per districte.")

    with col_b:
        st.subheader("P95 de resolució per districte")
        res_d = (
            df_c[
                (df_c["districte"] != "No especificat") &
                df_c["dies_resolucio"].notna()
            ]
            .groupby("districte")["dies_resolucio"]
            .quantile(0.95).reset_index(name="p95").sort_values("p95")
        )
        res_d["color"] = res_d["p95"].apply(
            lambda x: "#C00000" if x > 30 else "#2E75B6"
        )
        fig_r = px.bar(
            res_d, x="p95", y="districte", orientation="h",
            color="color", color_discrete_map="identity",
            labels={"p95": "Dies (P95)", "districte": ""}, height=360,
        )
        fig_r.add_vline(
            x=30, line_dash="dash", line_color="gray", line_width=1.5,
            annotation_text="Llindar 30 dies", annotation_position="top right",
        )
        fig_r.update_layout(
            showlegend=False, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT),
        )
        st.plotly_chart(fig_r, use_container_width=True)
        st.caption("Els valors superiors a 30 dies s'indiquen en vermell.")

    st.subheader("Tendència estructural")
    df_c_ts = (
        df_c.groupby(pd.Grouper(key="data_entrada", freq="ME"))
        .size().reset_index(name="n")
        .rename(columns={"data_entrada": "data"})
    )
    df_c_ts = df_c_ts[
        (df_c_ts["data"].dt.year >= 2014) & (df_c_ts["data"].dt.year <= 2025)
    ]
    df_c_ts["MM6"] = df_c_ts["n"].rolling(6, center=True, min_periods=1).mean()

    fig_mm = go.Figure()
    fig_mm.add_trace(go.Scatter(
        x=df_c_ts["data"], y=df_c_ts["n"],
        mode="lines", name="Peticions/mes",
        line=dict(color=COLOR_ACCENT, width=1.5), opacity=0.35,
    ))
    fig_mm.add_trace(go.Scatter(
        x=df_c_ts["data"], y=df_c_ts["MM6"],
        mode="lines", name="Tendència (MM6)",
        line=dict(color=COLOR_ACCENT, width=2.5),
    ))
    fig_mm.update_layout(
        height=280, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT), hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
    )
    st.plotly_chart(fig_mm, use_container_width=True)
    st.caption("La línia suavitzada mostra la tendència estructural de la sèrie temporal.")


# ================================================================ PÀGINA 4
elif pagina == "Equitat territorial":
    st.title("Equitat territorial")
    st.markdown(
        "Anàlisi del Disparate Impact (DI) entre districtes amb alta i baixa "
        "vulnerabilitat socioeconòmica (IRFD). DI < 0.8 indica possible discriminació "
        "territorial (regla del 80%, *Griggs v. Duke Power*, 1971)."
    )

    equitat_ext = equitat.copy()
    di_path = TERRIT_DIR / "di_per_cluster.parquet"
    df_di   = pd.read_parquet(di_path) if di_path.exists() else pd.DataFrame()
    irfd_col = "irfd" if "irfd" in equitat_ext.columns else None

    if irfd_col and len(df_di) > 0:
        dist_vuln = equitat_ext[equitat_ext["irfd"] < 100]["districte"].tolist()
        dist_fav  = equitat_ext[equitat_ext["irfd"] >= 100]["districte"].tolist()
        df_eq_all = df[
            (df["dies_resolucio"].notna()) & (df["districte"] != "No especificat")
        ]
        p95_vuln_g = df_eq_all[df_eq_all["districte"].isin(dist_vuln)]["dies_resolucio"].quantile(0.95)
        p95_fav_g  = df_eq_all[df_eq_all["districte"].isin(dist_fav)]["dies_resolucio"].quantile(0.95)
        DI_global  = p95_fav_g / p95_vuln_g if p95_vuln_g > 0 else None

        k1, k2, k3 = st.columns(3)
        k1.metric("DIR global", f"{DI_global:.3f}",
                  help="Un valor entre 0,8 i 1,2 indica absència de discriminació territorial.")
        k2.metric("P95 districtes vulnerables", f"{p95_vuln_g:.0f} dies")
        k3.metric("P95 districtes favorables",  f"{p95_fav_g:.0f} dies")
        st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Disparate Impact", "Renda i resolució", "Indicadors socioeconòmics"])

    with tab1:
        st.subheader("Disparate Impact per tipologia temàtica")
        if len(df_di) > 0:
            df_di["color"] = df_di["di"].apply(
                lambda x: "Alerta (DIR < 0,8)" if x < 0.8
                else ("Favorable (DIR > 1,2)" if x > 1.2 else "Sense alerta")
            )
            fig_di = px.bar(
                df_di.sort_values("di"),
                x="di", y="etiqueta", orientation="h",
                color="color",
                color_discrete_map={
                    "Alerta (DIR < 0,8)":    "#C00000",
                    "Sense alerta":          "#2E75B6",
                    "Favorable (DIR > 1,2)": "#548235",
                },
                labels={"di": "DIR", "etiqueta": ""}, height=400,
            )
            fig_di.add_vline(x=0.8, line_dash="dash", line_color="black",
                             line_width=1.5, annotation_text="Llindar 0,8")
            fig_di.add_vline(x=1.0, line_dash="dot", line_color="gray", line_width=1)
            fig_di.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT),
            )
            st.plotly_chart(fig_di, use_container_width=True)


    with tab2:
        if irfd_col and "n_peticions" in equitat_ext.columns:
            st.subheader("Renda, resolució i volum per districte")
            bubble_df = equitat_ext.dropna(subset=["irfd", "dies_resolucio_p95"])
            fig_bubble = px.scatter(
                bubble_df, x="irfd", y="dies_resolucio_p95",
                size="n_peticions", color="irfd",
                color_continuous_scale="RdYlGn", text="districte",
                hover_name="districte",
                labels={
                    "irfd":               "IRFD (Barcelona = 100)",
                    "dies_resolucio_p95": "P95 dies resolució",
                    "n_peticions":        "Peticions",
                },
                size_max=60, height=450,
            )
            fig_bubble.add_vline(
                x=100, line_dash="dot", line_color="gray", line_width=1.5,
                annotation_text="Mitjana BCN", annotation_position="top left",
            )
            fig_bubble.update_traces(
                textposition="top center",
                marker=dict(line=dict(width=1, color="white")),
            )
            fig_bubble.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family=FONT),
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
            st.caption("La mida de cada bombolla representa el volum de peticions del districte.")

    with tab3:
        if df_socio is not None:
            col_irfd, col_gini = st.columns(2)

            with col_irfd:
                st.subheader("Evolució de l'IRFD per districte")
                st.caption("Índex de Renda Familiar Disponible (2000–2017) · Ajuntament de Barcelona · CC BY 4.0")
                df_renda = df_socio.dropna(subset=["irfd"]).copy()
                if len(df_renda) > 0:
                    fig_irfd = px.line(
                        df_renda, x="any", y="irfd", color="Territori", markers=True,
                        labels={"any": "Any", "irfd": "IRFD (Barcelona = 100)"},
                        height=360,
                    )
                    fig_irfd.add_hline(y=100, line_dash="dash", line_color="gray",
                                       annotation_text="Mitjana BCN (100)",
                                       annotation_position="right")
                    fig_irfd.update_traces(line=dict(width=2))
                    fig_irfd.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family=FONT), hovermode="x unified",
                    )
                    st.plotly_chart(fig_irfd, use_container_width=True)

            with col_gini:
                st.subheader("Evolució del Gini per districte")
                st.caption("Índex de Gini de la renda tributària (2015–2023) · Ajuntament de Barcelona · CC BY 4.0")
                df_gini = df_socio.dropna(subset=["gini"]).copy()
                if len(df_gini) > 0:
                    fig_gini = px.line(
                        df_gini, x="any", y="gini", color="Territori", markers=True,
                        labels={"any": "Any", "gini": "Índex de Gini (%)"},
                        height=360,
                    )
                    fig_gini.update_traces(line=dict(width=2))
                    fig_gini.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family=FONT), hovermode="x unified",
                    )
                    st.plotly_chart(fig_gini, use_container_width=True)

    st.markdown("---")
    st.subheader("Resum per districte")
    cols_base  = ["districte", "n_peticions", "dies_resolucio_med", "dies_resolucio_p95", "pct_geo"]
    cols_extra = [c for c in ["irfd", "gini"] if c in equitat_ext.columns]
    cols_show  = [c for c in cols_base + cols_extra if c in equitat_ext.columns]
    rename_map = {
        "districte":           "Districte",
        "n_peticions":         "Peticions",
        "dies_resolucio_med":  "Mediana (dies)",
        "dies_resolucio_p95":  "P95 (dies)",
        "pct_geo":             "Cobertura geo (%)",
        "irfd":                "IRFD (BCN=100)",
        "gini":                "Gini (%)",
    }
    taula = equitat_ext[cols_show].rename(columns=rename_map)
    sort_col = "IRFD (BCN=100)" if "IRFD (BCN=100)" in taula.columns else "Peticions"
    taula = taula.sort_values(sort_col, ascending=sort_col != "Peticions")
    st.dataframe(taula, use_container_width=True, hide_index=True)

# ── Peu de pàgina
st.markdown("---")
st.caption(
    "TFG — Clusterització de peticions ciutadanes · "
    "Natàlia Abellan Barron · UOC 2026 · "
    "Dades: Open Data BCN (CC BY 4.0)"
)