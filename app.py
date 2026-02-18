import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit.components.v1 as components
import statistics
import re
import base64

# ==========================================
# 1. CONFIGURACIÓN Y ESTILOS GLOBALES (UI/UX)
# ==========================================
st.set_page_config(page_title="Tablero Predictivo de Procesos", layout="wide")

st.markdown("""
    <style>
        header[data-testid="stHeader"] { display: none !important; }
        div[data-testid="stToolbar"] { display: none !important; }
        * { font-family: 'Arial', sans-serif !important; }
        .fixed-header {
            position: fixed; top: 0; left: 0; width: 100%;
            background-color: #f0f2f6; padding: 15px 30px;
            z-index: 999999; border-bottom: 2px solid #d1d5db;
        }
        .header-title { margin: 0; font-size: 14px; font-weight: bold; color: #1f2937; font-family: 'Arial', sans-serif !important;}
        .block-container { margin-top: 80px; }
        
        .tabla-arial {
            width: 100%; border-collapse: collapse; font-family: 'Arial', sans-serif !important;
            font-size: 14px; color: #333; margin-bottom: 2rem;
        }
        .tabla-arial th {
            background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;
            padding: 12px 15px; text-align: center; font-weight: bold;
        }
        .tabla-arial td { 
            border-bottom: 1px solid #dee2e6; padding: 10px 15px; 
            text-align: center; 
        }
        .tabla-arial tr:hover { background-color: #f1f3f5; }
    </style>
    <div class="fixed-header"><h1 class="header-title">Tablero Predictivo de Procesos</h1></div>
""", unsafe_allow_html=True)

def formato_latino(numero, decimales=1):
    if pd.isna(numero): return "0"
    if decimales == 0:
        formateado = f"{int(numero):,}"
    else:
        formateado = f"{numero:,.{decimales}f}"
    return formateado.replace(',', 'X').replace('.', ',').replace('X', '.')

# ==========================================
# RENDERIZADOR AVANZADO MERMAID
# ==========================================
def render_mermaid(code: str):
    b64_code = base64.b64encode(code.encode('utf-8')).decode('utf-8')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ margin: 0; padding: 0; display: flex; justify-content: center; font-family: Arial, sans-serif; position: relative; }}
            #graphDiv {{ width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; padding-top: 20px; }}
            
            .mermaidTooltip {{
                position: absolute !important;
                text-align: left !important; 
                min-width: 150px !important;
                padding: 10px 15px !important;
                font-family: Arial, sans-serif !important;
                font-size: 13px !important;
                background-color: #1f2937 !important;
                color: #ffffff !important;
                border-radius: 6px !important;
                pointer-events: none !important;
                z-index: 999999 !important;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.3) !important;
                transition: opacity 0.1s ease !important;
                line-height: 1.6 !important;
                white-space: nowrap !important;
            }}
        </style>
    </head>
    <body>
        <div id="graphDiv">Generando mapa de proceso...</div>
        <script type="module">
            window.noAction = function() {{ return false; }};
            
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            
            mermaid.initialize({{ 
                startOnLoad: false, 
                theme: 'default', 
                fontFamily: 'Arial', 
                securityLevel: 'loose' 
            }});
            
            try {{
                const b64 = "{b64_code}";
                const graphDefinition = decodeURIComponent(escape(window.atob(b64)));
                
                mermaid.render('mermaid-svg', graphDefinition).then((result) => {{
                    document.getElementById('graphDiv').innerHTML = result.svg;
                    if(result.bindFunctions) {{
                        result.bindFunctions(document.getElementById('graphDiv'));
                    }}
                }}).catch((error) => {{
                    document.getElementById('graphDiv').innerHTML = "<div style='color:red;'><b>Error de renderizado:</b><br>" + error.message + "</div>";
                }});
            }} catch (e) {{
                document.getElementById('graphDiv').innerHTML = "<div style='color:red;'>Error decodificando el gráfico.</div>";
            }}
            
            setInterval(function() {{
                var tooltips = document.getElementsByClassName('mermaidTooltip');
                for(var i = 0; i < tooltips.length; i++) {{
                    var tt = tooltips[i];
                    if(tt.textContent.includes('<br>')) {{
                        tt.innerHTML = tt.textContent.split('<br>').join('<br>');
                    }}
                }}
            }}, 50);
            
        </script>
    </body>
    </html>
    """
    components.html(html_content, height=750, scrolling=True)

def mostrar_tabla_html(styler):
    html = styler.to_html()
    html = html.replace('<table', '<table class="tabla-arial"')
    st.markdown(html, unsafe_allow_html=True)

# ==========================================
# 2. VARIABLES DE SESIÓN
# ==========================================
if 'datos_procesados' not in st.session_state: st.session_state.datos_procesados = False
if 'df_transiciones' not in st.session_state: st.session_state.df_transiciones = None
if 'df_variantes' not in st.session_state: st.session_state.df_variantes = None
if 'dict_orden' not in st.session_state: st.session_state.dict_orden = {}

# ==========================================
# 3. PANTALLA 1: CARGA DE DATOS
# ==========================================
if not st.session_state.datos_procesados:
    st.info("Por favor, sube tus archivos CSV para comenzar.")
    col1, col2 = st.columns(2)
    with col1: archivo_log = st.file_uploader("1. Sube tu log principal", type=['csv'])
    with col2: archivo_est = st.file_uploader("2. Sube tu maestro de estados", type=['csv'])

    if archivo_log and archivo_est:
        try:
            with st.spinner("Procesando datos y modelando procesos..."):
                df_log = pd.read_csv(archivo_log, sep=None, engine='python', on_bad_lines='skip', encoding='utf-8-sig')
                df_est = pd.read_csv(archivo_est, sep=None, engine='python', on_bad_lines='skip', encoding='utf-8-sig')
                
                # --- EXTRACCIÓN INTELIGENTE DE ORDEN ---
                dict_orden = {'Inicio proceso': -9999, 'Fin proceso': 9999}
                if 'ESTADO' in df_est.columns and 'EST_ORDEN' in df_est.columns:
                    for _, r in df_est.dropna(subset=['ESTADO', 'EST_ORDEN']).iterrows():
                        val_str = str(r['EST_ORDEN']).strip()
                        try:
                            orden_val = float(val_str)
                        except ValueError:
                            # Extrae solo los números si el Excel tiene texto mezclado (ej: "1. Etapa" -> 1.0)
                            match = re.search(r'\d+', val_str)
                            orden_val = float(match.group()) if match else 9999
                        dict_orden[str(r['ESTADO']).strip()] = orden_val
                st.session_state.dict_orden = dict_orden
                # ---------------------------------------

                df_log['FECHA_ESTADO'] = pd.to_datetime(df_log['FECHA_ESTADO'], format='%d-%m-%Y', errors='coerce')
                df = df_log.merge(df_est[['ESTADO', 'EST_ORDEN']], on='ESTADO', how='left')
                df = df.sort_values(['ID', 'FECHA_ESTADO'])
                
                transiciones = []
                for case_id, group in df.groupby('ID'):
                    estados = ['Inicio proceso'] + group['ESTADO'].tolist() + ['Fin proceso']
                    fechas = [group['FECHA_ESTADO'].min()] + group['FECHA_ESTADO'].tolist() + [group['FECHA_ESTADO'].max()]
                    tipo_rec = group['TIPO RECURSO'].iloc[0] if 'TIPO RECURSO' in group.columns else 'Desconocido'
                    
                    for i in range(len(estados)-1):
                        duracion = (fechas[i+1] - fechas[i]).days if pd.notnull(fechas[i+1]) and pd.notnull(fechas[i]) else 0
                        transiciones.append({
                            'ID': case_id, 'Origen': estados[i], 'Destino': estados[i+1], 
                            'Duracion': duracion, 'Tipo_Recurso': tipo_rec
                        })
                
                df_trans = pd.DataFrame(transiciones)
                
                df_var = df_trans.groupby('ID').agg(
                    Ruta=('Destino', lambda x: ' -> '.join([s for s in x if s != 'Fin proceso'])),
                    Duracion_Total=('Duracion', 'sum'),
                    Tipo_Recurso=('Tipo_Recurso', 'first')
                ).reset_index()

                frecuencias = df_var['Ruta'].value_counts().reset_index()
                frecuencias.columns = ['Ruta', 'Frecuencia']
                mapeo_variantes = {row['Ruta']: f"Var {i+1}" for i, row in frecuencias.iterrows()}
                
                df_var['Nombre_Variante'] = df_var['Ruta'].map(mapeo_variantes)
                df_var['Ruta_Tooltip'] = df_var['Ruta'].apply(lambda x: x.replace(' -> ', '<br>&#8627; '))

                df_trans = df_trans.merge(df_var[['ID', 'Nombre_Variante', 'Ruta']], on='ID', how='left')

                st.session_state.df_transiciones = df_trans
                st.session_state.df_variantes = df_var
                st.session_state.datos_procesados = True
                st.rerun()
        except Exception as e:
            st.error(f"Error al procesar: {e}")

# ==========================================
# 4. PANTALLA 2: PESTAÑAS
# ==========================================
if st.session_state.datos_procesados:
    df_trans = st.session_state.df_transiciones
    df_var = st.session_state.df_variantes
    dict_orden = st.session_state.dict_orden
    
    if st.sidebar.button("Cargar nuevos archivos"):
        st.session_state.datos_procesados = False
        st.rerun()

    tab1, tab2 = st.tabs(["Mapa de Proceso", "Predicciones"])

    # ------------------------------------------
    # PESTAÑA 1: MAPA DE PROCESO
    # ------------------------------------------
    with tab1:
        col_grafo, col_panel = st.columns([7, 3])
        
        with col_panel:
            with st.container(height=850):
                st.subheader("Panel de Control")
                metrica_grafo = st.radio("Métrica en las flechas:", ["Frecuencia (Casos)", "Tiempo promedio (Días)"])
                
                resaltar_cuellos = st.checkbox("Resaltar Cuellos de Botella", value=False)
                st.markdown("---")
                
                var_counts = df_var.groupby(['Nombre_Variante', 'Ruta_Tooltip', 'Ruta']).size().reset_index(name='Frecuencia')
                var_counts['Orden'] = var_counts['Nombre_Variante'].str.replace('Var ', '').astype(int)
                var_counts = var_counts.sort_values('Orden', ascending=False)

                total_casos_proceso = var_counts['Frecuencia'].sum()
                var_counts['Porcentaje'] = (var_counts['Frecuencia'] / total_casos_proceso) * 100
                var_counts['Porcentaje_Txt'] = var_counts['Porcentaje'].apply(lambda x: formato_latino(x, 1) + "%")

                fig = px.bar(
                    var_counts.tail(15), 
                    x='Frecuencia', y='Nombre_Variante', orientation='h', 
                    title="Top Variantes (Clic para ver)",
                    text='Porcentaje_Txt', 
                    custom_data=['Nombre_Variante', 'Ruta_Tooltip', 'Ruta', 'Porcentaje_Txt']
                )
                
                fig.update_traces(
                    textposition='outside',
                    hovertemplate="<b>%{y}</b><br>Casos: %{x} (%{customdata[3]})<br><br><b>Ruta:</b><br>%{customdata[1]}<extra></extra>",
                    cliponaxis=False
                )
                
                fig.update_layout(
                    height=750, font=dict(family="Arial"), 
                    margin=dict(l=0, r=50, t=30, b=80), 
                    hoverlabel=dict(align="left", font_family="Arial", bgcolor="white", font_size=13),
                    yaxis_title=None
                )
                
                seleccion = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)
                
                variante_seleccionada = None
                if seleccion and seleccion.get("selection") and seleccion["selection"].get("points"):
                    variante_seleccionada = seleccion["selection"]["points"][0]["customdata"][0]
                    st.success(f"Viendo: {variante_seleccionada}")
                    if st.button("Ver Proceso Completo (Quitar filtro)"): st.rerun()

        with col_grafo:
            st.subheader("Flujo del Proceso")
            
            if variante_seleccionada:
                df_grafo = df_trans[df_trans['Nombre_Variante'] == variante_seleccionada]
            else:
                df_grafo = df_trans
            
            edges_stats = df_grafo.groupby(['Origen', 'Destino']).agg(
                Frecuencia=('ID', 'count'), Tiempo_Promedio=('Duracion', 'mean')
            ).reset_index()
            
            if edges_stats.empty:
                st.warning("No hay suficientes datos para dibujar el mapa con esta selección.")
            else:
                node_stats = df_grafo.groupby('Origen').agg(
                    Casos=('ID', 'count'), 
                    Tiempo_Promedio=('Duracion', 'mean'),
                    Mediana=('Duracion', 'median')
                ).fillna(0).to_dict('index')

                p20 = p40 = p60 = p80 = 0
                tiempos = []
                
                if resaltar_cuellos:
                    tiempos = [v['Tiempo_Promedio'] for v in node_stats.values() if v['Tiempo_Promedio'] > 0]
                    if tiempos:
                        p20 = np.percentile(tiempos, 20)
                        p40 = np.percentile(tiempos, 40)
                        p60 = np.percentile(tiempos, 60)
                        p80 = np.percentile(tiempos, 80)
                
                nodos_unicos = list(set(edges_stats['Origen'].tolist() + edges_stats['Destino'].tolist()))
                mapa_nodos = {nodo: f"N{i}" for i, nodo in enumerate(nodos_unicos)}
                
                mermaid_code = "flowchart TD\n"
                
                for nombre_real, nodo_id in mapa_nodos.items():
                    nombre_limpio = re.sub(r'[^a-zA-Z0-9 áéíóúÁÉÍÓÚñÑ.,_-]', ' ', str(nombre_real)).strip()
                    if not nombre_limpio: nombre_limpio = "Etapa_Desconocida"
                    
                    mermaid_code += f'    {nodo_id}(["{nombre_limpio}"])\n'
                    
                    color_fondo, color_texto, color_borde, ancho_borde = "#e5e7eb", "#000", "#9ca3af", "1px"
                    
                    if nombre_real == "Inicio proceso":
                        color_fondo, color_borde, ancho_borde = "transparent", "#22c55e", "2px"
                    elif nombre_real == "Fin proceso":
                        color_fondo, color_borde, ancho_borde = "transparent", "#f43f5e", "2px"
                    elif resaltar_cuellos and nombre_real in node_stats and len(tiempos) > 0:
                        t_prom = node_stats[nombre_real]['Tiempo_Promedio']
                        if t_prom > 0:
                            if t_prom <= p20: color_fondo = "#e1e5f2"
                            elif t_prom <= p40: color_fondo = "#e9aecb"
                            elif t_prom <= p60: color_fondo = "#f078a3"
                            elif t_prom <= p80: color_fondo, color_texto = "#f8417c", "#fff"
                            else: color_fondo, color_texto = "#ff0a54", "#fff"
                                    
                    mermaid_code += f'    style {nodo_id} fill:{color_fondo},stroke:{color_borde},stroke-width:{ancho_borde},color:{color_texto}\n'
                    
                    if nombre_real in node_stats and nombre_real not in ["Inicio proceso", "Fin proceso"]:
                        datos = node_stats[nombre_real]
                        texto_tooltip = f"Casos: {int(datos['Casos'])}<br>Promedio: {formato_latino(datos['Tiempo_Promedio'])} días<br>Mediana: {formato_latino(datos['Mediana'])} días"
                        mermaid_code += f'    click {nodo_id} call noAction() "{texto_tooltip}"\n'
                    elif nombre_real == "Fin proceso":
                        mermaid_code += f'    click {nodo_id} call noAction() "Fin del flujo"\n'
                    elif nombre_real == "Inicio proceso":
                        mermaid_code += f'    click {nodo_id} call noAction() "Inicio del flujo"\n'
                    
                max_frecuencia = edges_stats['Frecuencia'].max() if not edges_stats.empty else 1
                estilos_flechas = ""
                
                for idx, (_, row) in enumerate(edges_stats.iterrows()):
                    origen = row['Origen']
                    destino = row['Destino']
                    origen_id = mapa_nodos[origen]
                    destino_id = mapa_nodos[destino]
                    freq = row['Frecuencia']
                    tiempo = row['Tiempo_Promedio']
                    
                    is_rework = False
                    if origen == destino:
                        is_rework = True
                    else:
                        o_order = dict_orden.get(str(origen).strip())
                        d_order = dict_orden.get(str(destino).strip())
                        if o_order is not None and d_order is not None:
                            if d_order < o_order:
                                is_rework = True
                        else:
                            freq_bwd = edges_stats[(edges_stats['Origen'] == destino) & (edges_stats['Destino'] == origen)]['Frecuencia'].sum()
                            if freq_bwd > freq:
                                is_rework = True
                                
                    color_linea = "#00D2FF" if is_rework else "slategray"

                    label = f"{formato_latino(tiempo)} días" if "Tiempo" in metrica_grafo else f"{formato_latino(freq, 0)} casos"
                    
                    mermaid_code += f'    {origen_id} -->|"{label}"| {destino_id}\n'
                    
                    grosor = int(round(2.0 + (freq / max_frecuencia) * 4.0))
                    estilos_flechas += f'    linkStyle {idx} stroke-width:{grosor}px,stroke:{color_linea}\n'
                
                mermaid_code += estilos_flechas
                render_mermaid(mermaid_code)

    # ------------------------------------------
    # PESTAÑA 2: PREDICCIONES
    # ------------------------------------------
    with tab2:
        col_tit, col_conf = st.columns([3, 1])
        with col_tit:
            st.markdown("### Cálculos de Probabilidad")
        with col_conf:
            confianza_input = st.number_input("Nivel de Confianza (%)", min_value=50, max_value=99, value=95, step=1)
            
        z_score = statistics.NormalDist().inv_cdf((1 + confianza_input / 100) / 2)
        conf_text = f"{confianza_input}"

        def calcular_ci(df_agrupado, col_agrupacion, col_valor, rename_col):
            stats = df_agrupado.groupby(col_agrupacion)[col_valor].agg(
                Promedio='mean', Desviacion='std', Registros='count'
            ).reset_index()
            stats['Desviacion'] = stats['Desviacion'].fillna(0)
            
            stats[f'Límite_Inferior ({conf_text}%)'] = (stats['Promedio'] - (z_score * stats['Desviacion'])).clip(lower=0)
            stats[f'Límite_Superior ({conf_text}%)'] = stats['Promedio'] + (z_score * stats['Desviacion'])
            return stats.rename(columns={col_agrupacion: rename_col})
        
        formato_columnas = {
            "Promedio": lambda x: formato_latino(x) + " días",
            "Desviacion": lambda x: formato_latino(x),
            f"Límite_Inferior ({conf_text}%)": lambda x: formato_latino(x) + " días",
            f"Límite_Superior ({conf_text}%)": lambda x: formato_latino(x) + " días",
            "Registros": lambda x: formato_latino(x, 0)
        }

        st.subheader("1. Predicción General por Tipo de Recurso")
        stats_totales = calcular_ci(df_var, 'Tipo_Recurso', 'Duracion_Total', 'Tipo de Recurso')
        estilo_totales = stats_totales.style.hide(axis="index").format(formato_columnas)
        mostrar_tabla_html(estilo_totales)

        st.subheader("2. Predicción de Tiempos por Etapa")
        df_etapas = df_trans[df_trans['Origen'] != 'Inicio proceso']
        stats_etapas = calcular_ci(df_etapas, 'Destino', 'Duracion', 'Etapa Alcanzada')
        estilo_etapas = stats_etapas.style.hide(axis="index").format(formato_columnas)
        mostrar_tabla_html(estilo_etapas)

        st.subheader("3. Predicción de Tiempo por Variante (Top 5)")
        stats_var = calcular_ci(df_var, 'Nombre_Variante', 'Duracion_Total', 'Variante')
        stats_var = stats_var.sort_values(by='Registros', ascending=False).head(5)
        
        diccionario_rutas = df_var.set_index('Nombre_Variante')['Ruta'].to_dict()
        stats_var['Variante'] = stats_var['Variante'].apply(
            lambda v: f'<span title="{diccionario_rutas.get(v, "")}" style="cursor: help; border-bottom: 1px dotted #888;">{v}</span>'
        )
        
        estilo_var = stats_var.style.hide(axis="index").format(formato_columnas)
        mostrar_tabla_html(estilo_var)