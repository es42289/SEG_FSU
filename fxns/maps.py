import pandas as pd, plotly.express as px, streamlit as st
# import folium
# from streamlit_folium import st_folium

@st.fragment()
def map_wells(map_data):
    map_data['TYPE'] = map_data.apply(
        lambda row: "OIL" if row['CUMOIL_BBL'] > row['CUMGAS_MCF'] else "GAS", axis=1
    )
    map_data['BOE_BBL'] = map_data['CUMOIL_BBL'] + (map_data['CUMGAS_MCF'] / 6)

    # Rename columns to what st.map expects
    # Drop any rows with missing coordinates
    map_data = map_data.dropna()
    
    if not map_data.empty:
        # Create a custom hover text with "bold" column names using <b> tags
        map_data['hover_text'] = map_data.apply(
            lambda row: "<br>".join([
                f"<b>{col}</b>: {row[col]}" for col in map_data.columns
            ]), axis=1
        )

        fig = px.scatter_mapbox(
            map_data,
            lat="LATITUDE",
            lon="LONGITUDE",
            color="TYPE",
            size="BOE_BBL",
            color_discrete_map={
            "OIL": "lightgreen",
            "GAS": "red"
            },
            # hover_data=well_data.columns,
            hover_name="hover_text",  # forces rich hover display
            hover_data=None  # disables auto columns
        )
        # Enable HTML in hover (hovertemplate bypasses auto format)
        fig.update_traces(
            hovertemplate="%{hovertext}<extra></extra>"
        )

        # Set the mapbox style (you can change this dynamically later)
        fig.update_layout(
            mapbox_style="satellite-streets",#"open-street-map",#"carto-positron",  # Options: "satellite-streets", ,"open-street-map" etc.
            mapbox_accesstoken="pk.eyJ1IjoiZXM0MjI4OSIsImEiOiJjbTl1dGdsNzQwZDh6Mm1vc25hNWxwMzJiIn0.6sgv-zhF_RmEOm2C-B7XUA",
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        fig.update_layout(
            legend=dict(
                title = 'Well Type',
                x=0.01,       # x-position (0: left, 1: right)
                y=0.99,       # y-position (0: bottom, 1: top)
                bgcolor="rgba(255,255,255,0.7)",  # optional translucent background
                bordercolor="black",
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=0, b=0)  # optional: remove extra whitespace
        )
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

        # @st.fragment()
# def map_wells(map_data):
#     map_data['TYPE'] = map_data.apply(
#         lambda row: "OIL" if row['CUMOIL_BBL'] > row['CUMGAS_MCF'] else "GAS", axis=1
#     )
#     map_data['BOE_BBL'] = map_data['CUMOIL_BBL'] + (map_data['CUMGAS_MCF'] / 6)

#     map_data = map_data.dropna(subset=['LATITUDE', 'LONGITUDE'])

#     if not map_data.empty:
#         # Normalize BOE_BBL to 1-10
#         min_boe = map_data['BOE_BBL'].min()
#         max_boe = map_data['BOE_BBL'].max()
#         map_data['radius'] = 1 + 29 * (map_data['BOE_BBL'] - min_boe) / (max_boe - min_boe)

#         m = folium.Map(
#             location=[map_data['LATITUDE'].mean(), map_data['LONGITUDE'].mean()],
#             zoom_start=10,
#             tiles="CartoDB positron"
#         )

#         for _, row in map_data.iterrows():
#             color = 'green' if row['TYPE'] == 'OIL' else 'red'
#             popup_text = "<br>".join([f"{col}: {row[col]}" for col in map_data.columns])
#             tooltip_text = "<br>".join([f"<b>{col}</b>: {row[col]}" for col in map_data.columns])
            
#             folium.CircleMarker(
#                 location=[row['LATITUDE'], row['LONGITUDE']],
#                 radius=row['radius'],
#                 color='black',
#                 weight=1,
#                 fill=True,
#                 fill_color=color,
#                 fill_opacity=0.5,
#                 popup=folium.Popup(popup_text, max_width=300),
#                 tooltip=folium.Tooltip(tooltip_text)
#             ).add_to(m)

#         st_folium(m, width=800, height=600)