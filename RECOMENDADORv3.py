# ==========================================================
# APP DE RECOMENDACIÓN DE LUGARES (VERSIÓN 4 PÁGINAS + CORRELACIÓN)
# ==========================================================
import streamlit as st
import pandas as pd
import random
from surprise import Dataset, Reader, KNNBasic
import os
import io

# --- Imports para las visualizaciones ---
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- Imports para el mapa Folium ---
import folium
from streamlit_folium import st_folium

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Recomendador de Restaurantes", page_icon="🧑‍🍳", layout="wide")

# --- Función de limpieza de Coordenadas (CORREGIDA) ---
def clean_coordinate(coord_str):
    """Limpia formatos como '-36.834.407' a '-36.834407' de forma robusta"""
    try:
        s = str(coord_str).strip().replace(" ", "").replace(",", ".")
        try:
            return pd.to_numeric(s)
        except ValueError:
            pass
        s_no_dots = s.replace(".", "")
        if s_no_dots in ['', 'nan']: return pd.NA
        if s_no_dots.startswith("-"):
            if len(s_no_dots) > 8: s_clean = s_no_dots[:3] + "." + s_no_dots[3:]
            else: s_clean = s_no_dots[:2] + "." + s_no_dots[2:]
        else:
            if len(s_no_dots) > 7: s_clean = s_no_dots[:2] + "." + s_no_dots[2:]
            else: s_clean = s_no_dots[:1] + "." + s_no_dots[1:]
        return pd.to_numeric(s_clean)
    except:
        return pd.NA

# --- FUNCIONES DE CARGA DE DATOS (CON CACHÉ) ---
@st.cache_data
def load_metadata(file):
    """Carga y limpia el CSV/Excel de lugares (lugares.csv)"""
    if file.name.endswith('.csv'):
        try:
            df_raw = pd.read_csv(file, sep=';', on_bad_lines='skip')
            if df_raw.shape[1] == 1: file.seek(0); df_raw = pd.read_csv(file, sep=',', on_bad_lines='skip')
        except UnicodeDecodeError:
            file.seek(0); df_raw = pd.read_csv(file, sep=';', on_bad_lines='skip', encoding='latin1')
            if df_raw.shape[1] == 1: file.seek(0); df_raw = pd.read_csv(file, sep=',', on_bad_lines='skip', encoding='latin1')
    else:
        df_raw = pd.read_excel(file)

    unnamed_cols = [col for col in df_raw.columns if 'Unnamed:' in col]
    df = df_raw.drop(columns=unnamed_cols, errors='ignore')

    df = df.rename(columns={
        'id_lugar': 'poi_id', 'nombre_lugar': 'poi_name', 'categoria': 'category', 'Categoría': 'category',
        'etiquetas': 'tags', 'presupuesto': 'avg_cost', 'valoracion': 'avg_rating', 'rating': 'avg_rating',
        'rating_promedio': 'avg_rating', 'lat': 'lat', 'latitud': 'lat', 'lon': 'lon', 'longitud': 'lon'
    })
    if 'avg_rating' in df.columns:
        if df['avg_rating'].dtype == 'object':
            df['avg_rating'] = df['avg_rating'].astype(str).str.replace(',', '.', regex=False)
        df['avg_rating'] = pd.to_numeric(df['avg_rating'], errors='coerce')
    if 'avg_cost' in df.columns:
        if df['avg_cost'].dtype == 'object':
             df['avg_cost'] = df['avg_cost'].astype(str).str.replace(',', '.', regex=False)
        df['avg_cost'] = pd.to_numeric(df['avg_cost'], errors='coerce')
        diccionario_presupuesto = {
            1: "Bajo ($)", 2: "Medio-Bajo ($$)", 3: "Medio ($$$)", 
            4: "Medio-Alto ($$$$)", 5: "Alto ($$$$$)"
        }
        df['presupuesto_texto'] = df['avg_cost'].map(diccionario_presupuesto).fillna("No definido")
    if 'lat' in df.columns and 'lon' in df.columns:
        df['lat'] = df['lat'].apply(clean_coordinate)
        df['lon'] = df['lon'].apply(clean_coordinate)
    return df

@st.cache_data
def load_interactions(file):
    """Carga y limpia el CSV/Excel de interacciones (interacciones.csv)"""
    if file.name.endswith('.csv'):
        try:
            df_raw = pd.read_csv(file, sep=';', on_bad_lines='skip')
            if df_raw.shape[1] == 1: file.seek(0); df_raw = pd.read_csv(file, sep=',', on_bad_lines='skip')
        except UnicodeDecodeError:
            file.seek(0); df_raw = pd.read_csv(file, sep=';', on_bad_lines='skip', encoding='latin1')
            if df_raw.shape[1] == 1: file.seek(0); df_raw = pd.read_csv(file, sep=',', on_bad_lines='skip', encoding='latin1')
    else:
        df_raw = pd.read_excel(file)

    df = df_raw.rename(columns={'id_usuario': 'user_id', 'id_lugar': 'poi_id', 'valoracion': 'rating'})
    if 'rating' in df.columns:
        if df['rating'].dtype == 'object':
            df['rating'] = df['rating'].astype(str).str.replace(',', '.', regex=False)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    return df[['user_id', 'poi_id', 'rating']]

# --- FUNCIONES DEL MODELO ---
def train_model(_combined_df, model_type='IBCF'):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(_combined_df[['user_id', 'poi_id', 'rating']], reader)
    trainset = data.build_full_trainset()
    sim_options = {'name': 'cosine', 'user_based': (model_type == 'UBCF')}
    model = KNNBasic(sim_options=sim_options)
    model.fit(trainset)
    return model

def get_recommendations(model, metadata_df, user_id, rated_poi_ids):
    poi_id_to_name = pd.Series(metadata_df.poi_name.values, index=metadata_df.poi_id).to_dict()
    all_poi_ids = metadata_df['poi_id'].unique()
    predictions = []
    for poi_id in all_poi_ids:
        if poi_id not in rated_poi_ids:
            predicted_rating = model.predict(user_id, poi_id).est
            predictions.append((poi_id, predicted_rating))
    top_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)
    return top_recommendations[:5], poi_id_to_name

# --- Función auxiliar para las 4 nubes de palabras ---
def generate_wordcloud_plt(df, title):
    """Genera una figura de Matplotlib para una nube de palabras"""
    st.subheader(title)
    st.caption(f"**Total de restaurantes en este rango:** {len(df)}")
    if df.empty or 'tags' not in df.columns:
        st.info("No hay datos de etiquetas en este intervalo.")
        return
    text_data = " ".join(tag for tag in df['tags'].dropna().astype(str).str.replace(',', ' '))
    if not text_data.strip():
        st.info("No hay etiquetas para mostrar en este intervalo.")
        return
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', collocations=False).generate(text_data)
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear'); ax.axis('off')
        st.pyplot(fig_wc, use_container_width=True) 
    except Exception as e: 
        st.error(f"Error al generar nube: {e}")

# ==========================================================
# INTERFAZ DE STREAMLIT
# ==========================================================
st.title("📍🧑‍🍳 Sistema de Recomendación de Restaurantes")

st.sidebar.header("Configuración de Datos")
file_lugares = st.sidebar.file_uploader("1. Sube tu archivo de lugares (lugares.csv)", type=["csv", "xlsx"])
file_interacciones = st.sidebar.file_uploader("2. Sube tu archivo de interacciones (interacciones.csv)", type=["csv", "xlsx"])

if file_lugares and file_interacciones:
    
    metadata_df = load_metadata(file_lugares)
    interactions_df = load_interactions(file_interacciones)
    
    st.sidebar.success("¡Archivos cargados correctamente!")
    
    page = st.sidebar.radio("Elige una vista:", 
                            ("Conociendo el Gran Concepción", 
                             "Top 5 Restaurantes",
                             "Nubes por intervalo de valoración",
                             "Encuentra tus próximos restaurantes"))
    
    # ==========================================================
    # PÁGINA 1: NUBE DE ETIQUETAS (GENERAL)
    # ==========================================================
    if page == "Conociendo el Gran Concepción":
        
        st.header("☁️ Nube Gastronómica ☁️")
        st.markdown("Las palabras más grandes representan las etiquetas más comunes en *todo* el Gran Concepción.")
        
        if 'tags' in metadata_df.columns:
            text_data = " ".join(tag for tag in metadata_df['tags'].dropna().astype(str).str.replace(',', ' '))
            if text_data:
                try:
                    wordcloud = WordCloud(width=1600, height=600, background_color='white', colormap='viridis', collocations=False).generate(text_data)
                    fig_wc, ax = plt.subplots(figsize=(20, 7))
                    ax.imshow(wordcloud, interpolation='bilinear'); ax.axis('off')
                    st.pyplot(fig_wc, use_container_width=True) 
                except Exception as e: st.error(f"No se pudo generar la nube de palabras: {e}")
            else: st.info("No hay etiquetas para mostrar.")
        else: st.warning("El archivo de lugares no contiene una columna 'tags' (etiquetas).")

    # ==========================================================
    # PÁGINA 2: RANKING Y CORRELACIÓN
    # ==========================================================
    elif page == "Top 5 Restaurantes":
        
        st.header("🏆 Ranking Gastronómico 🏆")
        if 'category' in metadata_df.columns and 'avg_rating' in metadata_df.columns:
            opciones_cat = ["Todas"] + metadata_df['category'].dropna().unique().tolist()
            cat_select = st.selectbox("Filtrar por categoría:", opciones_cat)
            
            df_filtrado = metadata_df if cat_select == "Todas" else metadata_df[metadata_df['category'] == cat_select]
            df_filtrado = df_filtrado.sort_values(by="avg_rating", ascending=False).head(5)
            
            st.subheader(f"Top 5 Lugares con Mejor Valoración ({cat_select})")
            
            col1, col2 = st.columns(2)
            cols_list = [col1, col2, col1, col2, col1] 
            
            for i, (index, row) in enumerate(df_filtrado.iterrows()):
                with cols_list[i]:
                    with st.container(border=True):
                        st.markdown(f"### {i+1}. {row['poi_name']}")
                        st.markdown(f"**Rating Promedio (Real):** {row['avg_rating']:.2f} 🌟") 
                        st.caption(f"Categoría: {row.get('category', 'N/A')}")
                        st.caption(f"Presupuesto: {row.get('presupuesto_texto', 'N/A')}")
                        
        else: st.warning("Se necesitan columnas 'category' y 'avg_rating' para el gráfico.")
        
        st.divider()
        
        # --- ¡NUEVO! Gráfico de Correlación ---
        st.header("📈 Correlación Presupuesto vs. Valoración")
        if 'avg_cost' in metadata_df.columns and 'avg_rating' in metadata_df.columns:
            
            # Asegurarnos de que no hay NaN en las columnas clave para el gráfico
            df_scatter = metadata_df.dropna(subset=['avg_cost', 'avg_rating'])
            
            fig_scatter = px.scatter(
                df_scatter,
                x="avg_cost",
                y="avg_rating",
                title="Relación entre Presupuesto y Valoración",
                labels={'avg_cost': 'Nivel de Presupuesto (1-5)', 'avg_rating': 'Valoración Promedio (1-5)'},
                trendline="ols", # Añade la línea de tendencia
                hover_name="poi_name",
                hover_data=["presupuesto_texto", "category"]
            )
            
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        else:
            st.warning("Se necesitan las columnas 'avg_cost' (presupuesto) y 'avg_rating' (valoracion) para mostrar este gráfico.")

    
    # ==========================================================
    # PÁGINA 3: NUBES POR RATING
    # ==========================================================
    elif page == "Nubes por intervalo de valoración":
        
        st.header("☁️ Nubes según intervalo de valoración ☁️")
        st.markdown("Descubre qué etiquetas son más comunes para cada intervalo de valoración.")
        
        df_3_35 = metadata_df[(metadata_df['avg_rating'] >= 3) & (metadata_df['avg_rating'] < 3.5)]
        df_35_4 = metadata_df[(metadata_df['avg_rating'] >= 3.5) & (metadata_df['avg_rating'] < 4)]
        df_4_45 = metadata_df[(metadata_df['avg_rating'] >= 4) & (metadata_df['avg_rating'] < 4.5)]
        df_45_5 = metadata_df[(metadata_df['avg_rating'] >= 4.5) & (metadata_df['avg_rating'] <= 5)]

        col1, col2 = st.columns(2)
        with col1:
            generate_wordcloud_plt(df_3_35, "Valoración 3.0 - 3.49 😐")
        with col2:
            generate_wordcloud_plt(df_35_4, "Valoración 3.5 - 3.99 🙂")
        
        st.divider()
        
        col3, col4 = st.columns(2)
        with col3:
            generate_wordcloud_plt(df_4_45, "Valoración 4.0 - 4.49 😁")
        with col4:
            generate_wordcloud_plt(df_45_5, "Valoración 4.5 - 5.0 🤤")
            
    # ==========================================================
    # PÁGINA 4: RECOMENDADOR PERSONALIZADO
    # ==========================================================
    elif page == "Encuentra tus próximos restaurantes":

        if 'rated_places' not in st.session_state: st.session_state.rated_places = []
        if 'seen_places' not in st.session_state: st.session_state.seen_places = set()
        if 'current_place' not in st.session_state: st.session_state.current_place = None
        if 'recommendations_generated' not in st.session_state: st.session_state.recommendations_generated = False

        MIN_RATINGS_RECO = 10
        metadata_df_valid = metadata_df.dropna(subset=['poi_name'])
        
        if len(st.session_state.rated_places) < MIN_RATINGS_RECO:
            
            st.session_state.recommendations_generated = False 
            st.header(f"¡Hola! Ayúdanos a conocerte")
            st.markdown(f"Necesitamos que valores **{MIN_RATINGS_RECO - len(st.session_state.rated_places)}** lugares más.")
            st.progress(len(st.session_state.rated_places) / MIN_RATINGS_RECO)
            
            if st.session_state.current_place is None:
                available_places = metadata_df_valid[~metadata_df_valid['poi_id'].isin(st.session_state.seen_places)]
                if not available_places.empty:
                    st.session_state.current_place = available_places.sample(1).iloc[0]
                else:
                    st.warning("¡Vaya, te hemos preguntado por todos los lugares!"); st.session_state.rated_places = []

            if st.session_state.current_place is not None:
                place = st.session_state.current_place
                st.subheader(f"¿Has visitado {place['poi_name']}?")
                col1, col2 = st.columns(2)
                if col1.button("No, no he ido"):
                    st.session_state.seen_places.add(place['poi_id']); st.session_state.current_place = None; st.rerun()
                if col2.button("Sí, he ido"):
                    st.session_state.visited_current = True
            
            if st.session_state.get('visited_current', False):
                place = st.session_state.current_place
                rating = st.slider(f"¿Qué nota le pones a **{place['poi_name']}**?", 1.0, 5.0, 3.0, 0.1)
                if st.button("Guardar Valoración"):
                    st.session_state.rated_places.append({'user_id': 'usuario_prueba', 'poi_id': place['poi_id'], 'rating': rating})
                    st.session_state.seen_places.add(place['poi_id'])
                    st.session_state.current_place = None; st.session_state.visited_current = False
                    st.success(f"¡Valoración guardada! Faltan {MIN_RATINGS_RECO - len(st.session_state.rated_places)}.")
                    st.rerun()

        else: 
            
            if not st.session_state.recommendations_generated:
                st.success("¡Genial! Ya tienes el mínimo de valoraciones. ¡Listo para recomendar!")
                new_ratings_df = pd.DataFrame(st.session_state.rated_places)
                combined_interactions_df = pd.concat([interactions_df, new_ratings_df], ignore_index=True)
                st.header("Elige tu motor de recomendación")
                model_choice = st.selectbox("¿Qué modelo quieres usar?", ("Filtrado Basado en Ítems (IBCF - Recomendado)", "Filtrado Basado en Usuarios (UBCF)"))
                model_type = 'IBCF' if 'IBCF' in model_choice else 'UBCF'

                if st.button(f"Generar recomendaciones con {model_type}", type="primary"):
                    with st.spinner(f"Entrenando modelo {model_type} y buscando tus recomendaciones..."):
                        model = train_model(combined_interactions_df, model_type)
                        rated_ids = new_ratings_df['poi_id'].unique()
                        top_5, id_to_name_map = get_recommendations(model, metadata_df_valid, 'usuario_prueba', rated_ids)
                        st.session_state.top_5 = top_5
                        st.session_state.id_to_name_map = id_to_name_map
                        st.session_state.recommendations_generated = True
                        st.rerun()

            if st.session_state.get('recommendations_generated', False):
                st.header("¡Aquí tienes tu Top 5 de recomendaciones personalizadas!")
                top_5 = st.session_state.top_5
                id_to_name_map = st.session_state.id_to_name_map

                st.write("Aquí están tus 5 lugares recomendados:")
                col1, col2 = st.columns(2)
                cols_list = [col1, col2, col1, col2, col1] 
                
                for i, (poi_id, est_rating) in enumerate(top_5):
                    with cols_list[i]:
                        with st.container(border=True):
                            poi_name = id_to_name_map.get(poi_id, poi_id)
                            st.markdown(f"### {i+1}. {poi_name}")
                            st.markdown(f"**Rating Estimado:** {est_rating:.2f} 🌟")
                            try:
                                place_info = metadata_df_valid[metadata_df_valid['poi_id'] == poi_id].iloc[0]
                                st.caption(f"Categoría: {place_info.get('category', 'N/A')}")
                                st.caption(f"Presupuesto: {place_info.get('presupuesto_texto', 'N/A')}")
                            except:
                                pass
                
                st.markdown("---"); st.subheader("Mapa de tus recomendaciones")
                recommended_poi_ids = [poi_id for poi_id, _ in top_5]
                map_data_df = metadata_df[metadata_df['poi_id'].isin(recommended_poi_ids)]

                if not map_data_df.empty and 'lat' in map_data_df.columns:
                    map_data_df = map_data_df.dropna(subset=['lat', 'lon'])
                    if not map_data_df.empty:
                        location_center = [map_data_df['lat'].mean(), map_data_df['lon'].mean()]
                        m = folium.Map(location=location_center, zoom_start=14, tiles='CartoDB positron')
                        for _, row in map_data_df.iterrows():
                            popup_html = f"""
                            <b>{row['poi_name']}</b><br>
                            Valoración Promedio: {row.get('avg_rating', 0):.1f} 🌟<br>
                            Presupuesto: {row.get('presupuesto_texto', 'N/A')}
                            """
                            folium.Marker(
                                location=[row['lat'], row['lon']],
                                popup=folium.Popup(popup_html, max_width=300),
                                tooltip=row['poi_name'],
                                icon=folium.Icon(color='orange', icon='star')
                            ).add_to(m)
                        st_folium(m, width=725, height=500, use_container_width=True)
                    else: st.warning("No se pudieron encontrar coordenadas válidas para los lugares recomendados.")
                else: st.warning("No se pueden mostrar las recomendaciones en el mapa (faltan datos de lat/lon).")
            
            if st.button("Valorar de nuevo"):
                st.session_state.rated_places = []; st.session_state.seen_places = set()
                st.session_state.current_place = None; st.session_state.visited_current = False
                st.session_state.recommendations_generated = False
                st.rerun()

else:
    st.warning("Por favor, sube los archivos `lugares.csv/xlsx` e `interacciones.csv/xlsx` en el panel lateral para comenzar.")