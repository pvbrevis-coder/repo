import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import statistics
import re
import base64

# ==========================================
# 1. CONFIGURACIÓN Y ESTILOS GLOBALES (UI/UX)
# ==========================================
st.set_page_config(page_title="Monitor de Procesos", layout="wide")

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
        .header-title { margin: 0; font-size: 16px !important; font-weight: bold; color: #1f2937; font-family: 'Arial', sans-serif !important;}
        .block-container { margin-top: 40px; }
        
        a.header-anchor,
        [data-testid="stMarkdownContainer"] h1 a, 
        [data-testid="stMarkdownContainer"] h2 a, 
        [data-testid="stMarkdownContainer"] h3 a,
        [data-testid="stMarkdownContainer"] h4 a,
        .st-emotion-cache-16twljr a { 
            display: none !important; 
        }
        
        .tabla-arial {
            width: 100%; border-collapse: collapse; font-family: 'Arial', sans-serif !important;
            font-size: 14px; color: #333; margin-bottom: 0.5rem; 
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
        
        .nota-outliers { font-size: 12px; color: #666; font-style: italic; margin-bottom: 2rem; text-align: right;}
    </style>
    <div class="fixed-header"><div class="header-title">Monitor de Procesos — Análisis de Tiempos y Variantes</div></div>
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
                securityLevel: 'loose',
                flowchart: {{ arrowMarkerAbsolute: true }}
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
                
                var paths = document.querySelectorAll('path[marker-end]');
                paths.forEach(function(path) {{
                    var strokeColor = path.style.stroke || path.getAttribute('stroke');
                    if (strokeColor && strokeColor !== 'none') {{
                        var markerId = path.getAttribute('marker-end');
                        if (markerId && !markerId.includes('_custom_')) {{
                            var id = markerId.replace('url(', '').replace(')', '').replace(/["']/g, '');
                            if (id.includes('#')) id = id.substring(id.indexOf('#'));
                            var marker = document.querySelector(id);
                            if (marker) {{
                                var colorSafe = strokeColor.replace(/[^a-zA-Z0-9]/g, '');
                                var newId = id.substring(1) + '_custom_' + colorSafe;
                                var existingNewMarker = document.getElementById(newId);
                                
                                if (!existingNewMarker) {{
                                    var newMarker = marker.cloneNode(true);
                                    newMarker.id = newId;
                                    var markerPaths = newMarker.querySelectorAll('path');
                                    markerPaths.forEach(function(mp) {{
                                        mp.style.fill = strokeColor;
                                        mp.style.stroke = 'none';
                                    }});
                                    marker.parentNode.appendChild(newMarker);
                                }}
                                path.setAttribute('marker-end', 'url(#' + newId + ')');
                            }}
                        }}
                    }}
                }});
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

def mostrar_nota_outliers():
    st.markdown('<div class="nota-outliers">* Nota: Las celdas en rojo y subrayadas presentan datos atípicos (outliers). Pase el ratón por encima para ver detalles.</div>', unsafe_allow_html=True)

# ==========================================
# 2. VARIABLES DE SESIÓN
# ==========================================
if 'datos_procesados' not in st.session_state: st.session_state.datos_procesados = False
if 'df_transiciones' not in st.session_state: st.session_state.df_transiciones = None
if 'df_variantes' not in st.session_state: st.session_state.df_variantes = None
if 'dict_orden' not in st.session_state: st.session_state.dict_orden = {}
if 'periodo_fechas' not in st.session_state: st.session_state.periodo_fechas = ""

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
                
                col_responsable = 'RECURSO' if 'RECURSO' in df_log.columns else ('RESPONSABLE' if 'RESPONSABLE' in df_log.columns else None)

                dict_orden = {'Inicio proceso': -9999, 'Fin proceso': 9999}
                if 'ESTADO' in df_est.columns and 'EST_ORDEN' in df_est.columns:
                    for _, r in df_est.dropna(subset=['ESTADO', 'EST_ORDEN']).iterrows():
                        val_str = str(r['EST_ORDEN']).strip()
                        try:
                            orden_val = float(val_str)
                        except ValueError:
                            match = re.search(r'\d+', val_str)
                            orden_val = float(match.group()) if match else 9999
                        dict_orden[str(r['ESTADO']).strip()] = orden_val
                st.session_state.dict_orden = dict_orden

                df_log['FECHA_ESTADO'] = pd.to_datetime(df_log['FECHA_ESTADO'], format='mixed', dayfirst=True, errors='coerce')
                
                fechas_validas = df_log['FECHA_ESTADO'].dropna()
                if not fechas_validas.empty:
                    fecha_min = fechas_validas.min().strftime('%d-%m-%Y')
                    fecha_max = fechas_validas.max().strftime('%d-%m-%Y')
                    st.session_state.periodo_fechas = f"Período {fecha_min} - {fecha_max}"
                else:
                    st.session_state.periodo_fechas = "Período no disponible"

                df = df_log.merge(df_est[['ESTADO', 'EST_ORDEN']], on='ESTADO', how='left')
                df = df.sort_values(['ID', 'FECHA_ESTADO'])
                
                transiciones = []
                for case_id, group in df.groupby('ID'):
                    estados = ['Inicio proceso'] + group['ESTADO'].tolist() + ['Fin proceso']
                    fechas = [group['FECHA_ESTADO'].min()] + group['FECHA_ESTADO'].tolist() + [group['FECHA_ESTADO'].max()]
                    
                    if col_responsable:
                        recursos_lista = group[col_responsable].tolist()
                    else:
                        recursos_lista = ['Desconocido'] * len(group)
                    
                    recursos = ['Sistema'] + recursos_lista + ['Sistema']
                    
                    for i in range(len(estados)-1):
                        duracion = (fechas[i+1] - fechas[i]).days if pd.notnull(fechas[i+1]) and pd.notnull(fechas[i]) else 0
                        transiciones.append({
                            'ID': case_id, 'Origen': estados[i], 'Destino': estados[i+1], 
                            'Duracion': duracion, 'Recurso_Origen': recursos[i]
                        })
                
                df_trans = pd.DataFrame(transiciones)
                
                df_var = df_trans.groupby('ID').agg(
                    Ruta=('Destino', lambda x: ' -> '.join([s for s in x if s != 'Fin proceso'])),
                    Duracion_Total=('Duracion', 'sum')
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
    periodo_fechas = st.session_state.periodo_fechas
    
    if st.sidebar.button("Cargar nuevos archivos"):
        st.session_state.datos_procesados = False
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["Mapa de Proceso", "Predicciones", "Resumen Ejecutivo"])

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
                    st.success(f"Viendo: {variante_seleccionada} - Quita la selección haciendo clic en un área en blanco del gráfico para ver todo.")

        with col_grafo:
            st.subheader("Mapa de proceso")
            st.caption(f"**{periodo_fechas}**")
            
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

                min_t = max_t = rango_t = 0
                tiempos_validos = []
                
                if resaltar_cuellos:
                    tiempos_validos = [v['Tiempo_Promedio'] for k, v in node_stats.items() if v['Tiempo_Promedio'] > 0 and k not in ["Inicio proceso", "Fin proceso"]]
                    if tiempos_validos:
                        min_t = min(tiempos_validos)
                        max_t = max(tiempos_validos)
                        rango_t = max_t - min_t
                
                nodos_unicos = list(set(edges_stats['Origen'].tolist() + edges_stats['Destino'].tolist()))
                mapa_nodos = {nodo: f"N{i}" for i, nodo in enumerate(nodos_unicos)}
                
                mermaid_code = "flowchart TD\n"
                
                def sort_nodes(item):
                    if item[0] == "Inicio proceso": return 0
                    if item[0] == "Fin proceso": return 2
                    return 1
                
                nodos_ordenados = sorted(mapa_nodos.items(), key=sort_nodes)
                
                for nombre_real, nodo_id in nodos_ordenados:
                    nombre_limpio = re.sub(r'[^a-zA-Z0-9 áéíóúÁÉÍÓÚñÑ.,_-]', ' ', str(nombre_real)).strip()
                    if not nombre_limpio: nombre_limpio = "Etapa_Desconocida"
                    
                    mermaid_code += f'    {nodo_id}(["{nombre_limpio}"])\n'
                    
                    color_fondo, color_texto, color_borde, ancho_borde = "#e5e7eb", "#000", "#9ca3af", "1px"
                    
                    if nombre_real == "Inicio proceso":
                        color_fondo, color_borde, ancho_borde = "transparent", "#22c55e", "2px"
                    elif nombre_real == "Fin proceso":
                        color_fondo, color_borde, ancho_borde = "transparent", "#f43f5e", "2px"
                    elif resaltar_cuellos and nombre_real in node_stats and len(tiempos_validos) > 0:
                        t_prom = node_stats[nombre_real]['Tiempo_Promedio']
                        if t_prom > 0:
                            if rango_t == 0:
                                color_fondo, color_texto = "#fff0f3", "#000"
                            else:
                                idx = int(round(4 * (t_prom - min_t) / rango_t))
                                idx = max(0, min(4, idx)) 
                                colores = [("#fff0f3", "#000"), ("#f2bac9", "#000"), ("#e4849e", "#000"), ("#d64e74", "#fff"), ("#c9184a", "#fff")]
                                color_fondo, color_texto = colores[idx]
                                    
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
                    dash_style = ",stroke-dasharray: 5 5" if is_rework else ""

                    label = f"{formato_latino(tiempo)} días" if "Tiempo" in metrica_grafo else f"{formato_latino(freq, 0)} casos"
                    
                    mermaid_code += f'    {origen_id} -->|"{label}"| {destino_id}\n'
                    
                    grosor = int(round(2.0 + (freq / max_frecuencia) * 4.0))
                    estilos_flechas += f'    linkStyle {idx} stroke-width:{grosor}px,stroke:{color_linea}{dash_style}\n'
                
                mermaid_code += estilos_flechas
                render_mermaid(mermaid_code)
                
                st.markdown("""
                    <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 20px; font-family: Arial, sans-serif; font-size: 13px; margin-top: 15px; padding: 12px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
                        <div style="display: flex; align-items: center; gap: 5px;">
                            <b>Flujo:</b>
                            <span style="display: inline-block; width: 25px; border-top: 2px solid slategray; margin-left: 5px;"></span> Normal
                            <span style="display: inline-block; width: 25px; border-top: 2px dashed #00D2FF; margin-left: 10px;"></span> Reproceso
                        </div>
                        <div style="display: flex; align-items: center; gap: 5px; margin-left: 15px;">
                            <b>Tiempo etapa:</b>
                            <span style="margin-left: 5px; margin-right: 3px;">Mínimo</span>
                            <span style="display: inline-block; width: 14px; height: 14px; background-color: #fff0f3; border: 1px solid #ccc; border-radius: 3px;"></span>
                            <span style="display: inline-block; width: 14px; height: 14px; background-color: #f2bac9; border: 1px solid #ccc; border-radius: 3px;"></span>
                            <span style="display: inline-block; width: 14px; height: 14px; background-color: #e4849e; border: 1px solid #ccc; border-radius: 3px;"></span>
                            <span style="display: inline-block; width: 14px; height: 14px; background-color: #d64e74; border: 1px solid #ccc; border-radius: 3px;"></span>
                            <span style="display: inline-block; width: 14px; height: 14px; background-color: #c9184a; border: 1px solid #ccc; border-radius: 3px;"></span>
                            <span style="margin-left: 3px;">Máximo</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

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
        N_MIN_VALIDO = 5  

        def calcular_predicciones(df_agrupado, col_agrupacion, col_valor, rename_col):
            resultados = []
            for grupo, datos in df_agrupado.groupby(col_agrupacion):
                valores = datos[col_valor].dropna().values
                n = len(valores)
                if n == 0:
                    continue

                media = np.mean(valores)
                std   = np.std(valores, ddof=1) if n > 1 else 0.0

                margen = z_score * std * np.sqrt(1 + 1 / n) if n > 1 else 0.0
                lim_inf = max(0.0, media - margen)
                lim_sup = media + margen

                p5  = np.percentile(valores, 5)
                p95 = np.percentile(valores, 95)

                q1, q3 = np.percentile(valores, [25, 75])
                iqr = q3 - q1
                fence_sup = q3 + 1.5 * iqr
                fence_inf = q1 - 1.5 * iqr
                n_outliers = int(np.sum((valores > fence_sup) | (valores < fence_inf)))
                pct_outliers = (n_outliers / n * 100) if n > 0 else 0.0

                fiabilidad = "⚠ Pocos datos" if n < N_MIN_VALIDO else (
                    f"⚠ {n_outliers} outlier{'s' if n_outliers != 1 else ''} ({formato_latino(pct_outliers, 1)}%)"
                    if n_outliers > 0 else "✓ OK"
                )

                resultados.append({
                    rename_col: grupo,
                    "Registros": n,
                    "Promedio": media,
                    "Desviación": std,
                    f"Límite Inferior ({conf_text}%)": lim_inf,
                    f"Límite Superior ({conf_text}%)": lim_sup,
                    "P5 (empírico)": p5,
                    "P95 (empírico)": p95,
                    "Calidad": fiabilidad,
                })

            return pd.DataFrame(resultados)

        li_col = f"Límite Inferior ({conf_text}%)"
        ls_col = f"Límite Superior ({conf_text}%)"

        formato_columnas = {
            "Promedio":       lambda x: formato_latino(x) + " días",
            "Desviación":     lambda x: formato_latino(x),
            li_col:           lambda x: formato_latino(x) + " días",
            ls_col:           lambda x: formato_latino(x) + " días",
            "P5 (empírico)":  lambda x: formato_latino(x) + " días",
            "P95 (empírico)": lambda x: formato_latino(x) + " días",
            "Registros":      lambda x: formato_latino(x, 0),
        }

        # ------------------ SECCIÓN 1 ------------------
        st.subheader("1. Predicción de Tiempos por Recurso")
        st.caption("⏱ Tiempo promedio que cada recurso demora en procesar las etapas que tiene asignadas.")
        
        df_recursos_pred = df_trans[df_trans['Recurso_Origen'] != 'Sistema']
        stats_totales = calcular_predicciones(df_recursos_pred, 'Recurso_Origen', 'Duracion', 'Recurso')

        if not stats_totales.empty:
            dict_cal_t = stats_totales.set_index('Recurso')['Calidad'].to_dict()
            stats_totales['Recurso'] = stats_totales['Recurso'].apply(
                lambda x: f'<span title="{dict_cal_t.get(x,"")}" style="cursor:help; border-bottom:1px dotted #c9184a; color:#c9184a;">{x}</span>' if str(dict_cal_t.get(x,"")).startswith("⚠") else x
            )
            stats_totales.drop(columns=['Calidad'], inplace=True)
            estilo_totales = stats_totales.style.hide(axis="index").format(formato_columnas)
            mostrar_tabla_html(estilo_totales)
            mostrar_nota_outliers()
        else:
            st.info("No se encontraron recursos en los datos.")

        # ------------------ SECCIÓN 2 ------------------
        st.subheader("2. Predicción de Tiempos por Etapa")
        st.caption("⏱ Tiempo de permanencia en cada etapa: días transcurridos desde el inicio de la etapa hasta el inicio de la siguiente.")

        df_etapas = df_trans[(df_trans['Origen'] != 'Inicio proceso') & (df_trans['Destino'] != 'Fin proceso')]
        stats_etapas = calcular_predicciones(df_etapas, 'Origen', 'Duracion', 'Etapa')

        if not stats_etapas.empty:
            dict_cal_e = stats_etapas.set_index('Etapa')['Calidad'].to_dict()
            stats_etapas['Etapa'] = stats_etapas['Etapa'].apply(
                lambda x: f'<span title="{dict_cal_e.get(x,"")}" style="cursor:help; border-bottom:1px dotted #c9184a; color:#c9184a;">{x}</span>' if str(dict_cal_e.get(x,"")).startswith("⚠") else x
            )
            stats_etapas.drop(columns=['Calidad'], inplace=True)
            estilo_etapas = stats_etapas.style.hide(axis="index").format(formato_columnas)
            mostrar_tabla_html(estilo_etapas)
            mostrar_nota_outliers()

        # ------------------ SECCIÓN 3 ------------------
        st.subheader("3. Predicción de Tiempo Total por Variante (Top 5)")
        st.caption(f"Solo se muestran las 5 variantes más frecuentes con al menos {N_MIN_VALIDO} casos.")

        stats_var = calcular_predicciones(df_var, 'Nombre_Variante', 'Duracion_Total', 'Variante')
        stats_var_validas = stats_var[stats_var['Registros'] >= N_MIN_VALIDO].sort_values('Registros', ascending=False).head(5)
        n_excluidas = len(stats_var) - len(stats_var[stats_var['Registros'] >= N_MIN_VALIDO])

        if n_excluidas > 0:
            st.info(f"{n_excluidas} variante(s) excluida(s) por tener menos de {N_MIN_VALIDO} casos.")

        if not stats_var_validas.empty:
            dict_cal_v = stats_var_validas.set_index('Variante')['Calidad'].to_dict()
            diccionario_rutas = df_var.set_index('Nombre_Variante')['Ruta'].to_dict()
            
            def format_var_tooltip(var):
                calidad = dict_cal_v.get(var, "")
                ruta = diccionario_rutas.get(var, "")
                if str(calidad).startswith("⚠"):
                    tooltip_text = f"Ruta: {ruta} &#10;Alerta: {calidad}"
                    return f'<span title="{tooltip_text}" style="cursor: help; border-bottom: 1px dotted #c9184a; color: #c9184a;">{var}</span>'
                else:
                    return f'<span title="Ruta: {ruta}" style="cursor: help; border-bottom: 1px dotted #888;">{var}</span>'
                    
            stats_var_validas['Variante'] = stats_var_validas['Variante'].apply(format_var_tooltip)
            stats_var_validas.drop(columns=['Calidad'], inplace=True)
            estilo_var = stats_var_validas.style.hide(axis="index").format(formato_columnas)
            mostrar_tabla_html(estilo_var)
            mostrar_nota_outliers()
        else:
            st.warning(f"No hay variantes con al menos {N_MIN_VALIDO} casos para mostrar.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style="background:#f0f9ff; border-left:4px solid #0ea5e9; padding:12px 16px;
                        border-radius:6px; font-size:13px; font-family:Arial,sans-serif; margin-top:10px;">
                <b>Nota metodológica</b><br>
                • Los <b>límites de predicción</b> estiman el rango donde caerá la duración de un <i>caso nuevo</i>
                con el nivel de confianza seleccionado, usando la fórmula matemática <code>μ ± z · σ · √(1 + 1/n)</code>.<br>
                • Los <b>percentiles P5 y P95</b> son empíricos y no asumen distribución normal, siendo más robustos
                ante la asimetría típica de los tiempos de proceso.<br>
                • El <b>tiempo por etapa</b> mide el tiempo de permanencia en cada estado. Dado que el sistema registra
                únicamente el inicio de cada etapa, el tiempo entre dos registros consecutivos equivale al tiempo de permanencia en la etapa de origen.<br>
                • Los grupos con menos de {N_MIN_VALIDO} casos se excluyen o se marcan como poco fiables por falta de suficiencia estadística.
            </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------
    # PESTAÑA 3: RESUMEN EJECUTIVO
    # ------------------------------------------
    with tab3:
        st.markdown("### Resumen Ejecutivo")
        st.caption("Diagnóstico at-a-glance para la toma de decisiones. Basado en el universo completo de casos cargados.")

        N_MIN_RESUMEN = 5
        z_resumen = statistics.NormalDist().inv_cdf((1 + 95 / 100) / 2)  

        df_etapas_res = df_trans[(df_trans['Origen'] != 'Inicio proceso') & (df_trans['Destino'] != 'Fin proceso')]
        etapa_stats = df_etapas_res.groupby('Origen').agg(
            Promedio=('Duracion', 'mean'),
            Casos=('ID', 'count')
        ).reset_index().rename(columns={'Origen': 'Etapa'})
        etapa_stats = etapa_stats[etapa_stats['Promedio'] > 0].sort_values('Promedio', ascending=False)

        def pred_variante(valores, z):
            n = len(valores)
            if n < N_MIN_RESUMEN:
                return None, None, None
            media = np.mean(valores)
            std   = np.std(valores, ddof=1) if n > 1 else 0.0
            margen = z * std * np.sqrt(1 + 1/n)
            return media, max(0, media - margen), media + margen

        pronostico_rows = []
        diccionario_rutas_res = df_var.set_index('Nombre_Variante')['Ruta'].to_dict()
        for var, grp in df_var.groupby('Nombre_Variante'):
            vals = grp['Duracion_Total'].dropna().values
            media, li, ls = pred_variante(vals, z_resumen)
            if media is not None:
                pronostico_rows.append({
                    'Variante': var,
                    'Ruta': diccionario_rutas_res.get(var, ''),
                    'Casos': len(vals),
                    'Promedio': media,
                    'Li95': li,
                    'Ls95': ls,
                })
        df_pronostico = pd.DataFrame(pronostico_rows).sort_values('Casos', ascending=False).head(5) if pronostico_rows else pd.DataFrame()

        col_cb, col_rec = st.columns(2)

        with col_cb:
            st.markdown("#### ① Cuellos de botella por etapa")
            st.caption("Etapas ordenadas por tiempo de permanencia promedio. El tamaño del punto indica el volumen de casos que pasan por esa etapa.")

            if not etapa_stats.empty:
                # SE MODIFICA PARA QUE EL COLOR DEPENDA SOLO DEL PROMEDIO (DÍAS)
                etapa_stats['Casos_txt'] = etapa_stats['Casos'].apply(lambda x: formato_latino(x, 0))
                etapa_stats['Promedio_txt'] = etapa_stats['Promedio'].apply(lambda x: formato_latino(x, 1))

                fig_cb = px.bar(
                    etapa_stats,
                    x='Promedio', y='Etapa',
                    orientation='h',
                    color='Promedio', # <-- Corrección: Color solo por Promedio de días
                    color_continuous_scale=["#fff0f3", "#f2bac9", "#e4849e", "#d64e74", "#c9184a"],
                    text=etapa_stats['Promedio'].apply(lambda x: f"{formato_latino(x)} días"),
                    custom_data=['Casos_txt', 'Promedio_txt'],
                    labels={'Promedio': 'Días promedio', 'Etapa': ''},
                )
                fig_cb.update_traces(
                    textposition='outside',
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Promedio: %{customdata[1]} días<br>Casos: %{customdata[0]}<extra></extra>",
                )
                fig_cb.update_layout(
                    height=380,
                    font=dict(family="Arial", size=13),
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=60, t=10, b=40),
                    yaxis=dict(categoryorder='total ascending'),
                    xaxis_title="Días promedio de permanencia",
                    xaxis=dict(rangemode='tozero')
                )
                st.plotly_chart(fig_cb, use_container_width=True)

                peor = etapa_stats.iloc[0]
                st.markdown(f"""
                    <div style="background:#fff0f3; border-left:4px solid #c9184a; padding:10px 14px;
                                border-radius:6px; font-size:13px; font-family:Arial,sans-serif;">
                        <b>⚠ Mayor cuello de botella:</b> {peor['Etapa']}<br>
                        Promedio de <b>{formato_latino(peor['Promedio'])} días</b> · {formato_latino(peor['Casos'], 0)} casos
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Sin datos suficientes para calcular cuellos de botella.")

        with col_rec:
            st.markdown("#### ② Recursos con sobrecarga")
            st.caption("Cada punto es un recurso. Eje Y: tiempo promedio por etapa. Tamaño: volumen de etapas procesadas.")

            df_recursos_res = df_trans[df_trans['Recurso_Origen'] != 'Sistema']
            recurso_stats = df_recursos_res.groupby('Recurso_Origen').agg(
                Promedio=('Duracion', 'mean'),
                Casos=('ID', 'count')
            ).reset_index().rename(columns={'Recurso_Origen': 'Recurso'})

            if not recurso_stats.empty:
                recurso_stats['Promedio_txt'] = recurso_stats['Promedio'].apply(lambda x: formato_latino(x, 1))
                recurso_stats['Casos_txt'] = recurso_stats['Casos'].apply(lambda x: formato_latino(x, 0))

                fig_rec = px.scatter(
                    recurso_stats,
                    x='Recurso', y='Promedio',
                    size='Casos',
                    color='Promedio',
                    color_continuous_scale=["#e0f2fe", "#0ea5e9", "#0369a1"],
                    size_max=60,
                    text='Recurso', # <-- AQUÍ SE MUESTRA SOLO EL NOMBRE DEL RECURSO
                    custom_data=['Casos_txt', 'Promedio_txt'],
                    labels={'Promedio': 'Tiempo promedio (días)', 'Recurso': ''},
                )
                fig_rec.update_traces(
                    textposition='top center',
                    hovertemplate="<b>%{x}</b><br>Tiempo promedio: %{customdata[1]} días<br>Etapas procesadas: %{customdata[0]}<extra></extra>",
                )
                fig_rec.update_layout(
                    height=380,
                    font=dict(family="Arial", size=13),
                    coloraxis_showscale=False,
                    margin=dict(l=10, r=30, t=10, b=60),
                    xaxis=dict(showticklabels=False),
                    yaxis_title="Tiempo promedio de procesamiento (días)",
                    yaxis=dict(rangemode='tozero') 
                )
                st.plotly_chart(fig_rec, use_container_width=True)

                recurso_stats['Score_R'] = (
                    (recurso_stats['Promedio'] / recurso_stats['Promedio'].max()) * 0.5 +
                    (recurso_stats['Casos']    / recurso_stats['Casos'].max())    * 0.5
                )
                peor_r = recurso_stats.sort_values('Score_R', ascending=False).iloc[0]
                st.markdown(f"""
                    <div style="background:#e0f2fe; border-left:4px solid #0369a1; padding:10px 14px;
                                border-radius:6px; font-size:13px; font-family:Arial,sans-serif;">
                        <b>⚠ Recurso más crítico:</b> {peor_r['Recurso']}<br>
                        Promedio de <b>{formato_latino(peor_r['Promedio'])} días</b> · {formato_latino(peor_r['Casos'], 0)} etapas procesadas
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Sin datos de recursos para analizar.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ================================================================
        # FILA 2: Pronóstico casos futuros
        # ================================================================
        st.markdown("#### ③ Pronóstico para casos futuros (Top 5 variantes)")
        st.caption("Estimación de duración total para un caso nuevo según la variante de proceso. Intervalo de predicción al 95% de confianza.")

        if not df_pronostico.empty:
            total_casos_resumen = len(df_var)
            df_pronostico['Pct'] = (df_pronostico['Casos'] / total_casos_resumen) * 100
            
            df_pronostico['Promedio_txt'] = df_pronostico['Promedio'].apply(lambda x: formato_latino(x, 1))
            df_pronostico['Li95_txt'] = df_pronostico['Li95'].apply(lambda x: formato_latino(x, 1))
            df_pronostico['Ls95_txt'] = df_pronostico['Ls95'].apply(lambda x: formato_latino(x, 1))
            df_pronostico['Casos_txt'] = df_pronostico['Casos'].apply(lambda x: formato_latino(x, 0))
            
            df_pronostico['Label_Casos'] = df_pronostico.apply(lambda r: f"{r['Casos_txt']} ({formato_latino(r['Pct'], 1)}%)", axis=1)

            fig_pron = px.scatter(
                df_pronostico,
                x='Variante', y='Promedio',
                error_y=df_pronostico['Ls95'] - df_pronostico['Promedio'],
                error_y_minus=df_pronostico['Promedio'] - df_pronostico['Li95'],
                text='Label_Casos',
                size='Casos',
                size_max=30,
                color_discrete_sequence=['#0ea5e9'],
                custom_data=['Li95_txt', 'Ls95_txt', 'Casos_txt', 'Promedio_txt'], 
                labels={'Promedio': 'Días (promedio)', 'Variante': ''},
            )
            fig_pron.update_traces(
                textposition='middle right',
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Promedio: %{customdata[3]} días<br>"
                    "Intervalo 95%: %{customdata[0]} – %{customdata[1]} días<br>"
                    "Casos: %{customdata[2]}<extra></extra>"
                )
            )
            
            for _, row in df_pronostico.iterrows():
                # Etiqueta Límite Superior
                fig_pron.add_annotation(
                    x=row['Variante'], 
                    y=row['Ls95'],
                    text=formato_latino(row['Ls95'], 1),
                    showarrow=False,
                    xanchor='left',
                    yanchor='middle',
                    xshift=8, 
                    font=dict(size=11, color='#0ea5e9')
                )
                # Etiqueta Límite Inferior
                fig_pron.add_annotation(
                    x=row['Variante'], 
                    y=row['Li95'],
                    text=formato_latino(row['Li95'], 1),
                    showarrow=False,
                    xanchor='left',
                    yanchor='middle',
                    xshift=8,
                    font=dict(size=11, color='#0ea5e9')
                )

            fig_pron.update_layout(
                height=350,
                font=dict(family="Arial", size=13),
                margin=dict(l=10, r=80, t=20, b=60),
                yaxis_title="Duración estimada (días)",
                yaxis=dict(rangemode='tozero')
            )
            st.plotly_chart(fig_pron, use_container_width=True)

            tabla_pron = df_pronostico[['Variante', 'Casos', 'Promedio', 'Li95', 'Ls95']].copy()
            tabla_pron.columns = ['Variante', 'Casos', 'Promedio', 'Límite Inf. (95%)', 'Límite Sup. (95%)']
            
            tabla_pron['Variante'] = tabla_pron['Variante'].apply(
                lambda v: f'<span title="{diccionario_rutas_res.get(v, "")}" style="cursor: help; border-bottom: 1px dotted #888;">{v}</span>'
            )
            
            fmt_pron = {
                'Promedio':            lambda x: formato_latino(x) + " días",
                'Límite Inf. (95%)':   lambda x: formato_latino(x) + " días",
                'Límite Sup. (95%)':   lambda x: formato_latino(x) + " días",
                'Casos':               lambda x: formato_latino(x, 0),
            }
            estilo_pron = tabla_pron.style.hide(axis="index").format(fmt_pron)
            mostrar_tabla_html(estilo_pron)
        else:
            st.info(f"No hay variantes con al menos {N_MIN_RESUMEN} casos para generar pronósticos.")