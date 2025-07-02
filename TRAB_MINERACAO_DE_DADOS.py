import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import unicodedata
import plotly.express as px
import webbrowser
from sklearn.cluster import KMeans
import numpy as np

# Caminho do dataset local
customers_path = r"C:\Users\guilh\Downloads\archive\olist_customers_dataset.csv"
orders_path = r"C:\Users\guilh\Downloads\archive\olist_orders_dataset.csv"
payments_path = r"C:\Users\guilh\Downloads\archive\olist_order_payments_dataset.csv"

# Carregar os datasets
customers_df = pd.read_csv(customers_path)
orders_df = pd.read_csv(orders_path)
payments_df = pd.read_csv(payments_path)

# ---------------------------
# Tratamento de Dados
# ---------------------------
# Limpeza de dados: remover linhas com valores essenciais ausentes e duplicatas
customers_df = customers_df.dropna(subset=['customer_id', 'customer_city', 'customer_state']).drop_duplicates()
orders_df = orders_df.dropna(subset=['order_id', 'customer_id']).drop_duplicates()
payments_df = payments_df.dropna(subset=['order_id', 'payment_value']).drop_duplicates()

# Padronizar nomes de cidades (removemos acentos e converter para minúsculas)
def normalize_text(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()

customers_df['customer_city'] = customers_df['customer_city'].apply(normalize_text)

# Filtramos apenas os clientes do estado de São Paulo
customers_sp = customers_df[customers_df['customer_state'] == 'SP']

# Somar pagamentos por pedido para evitar duplicidade
pagamentos_agrupados = payments_df.groupby('order_id', as_index=False)['payment_value'].sum()

# Mesclamos os datasets para obter informações completas (agora sem duplicidade)
orders_customers = pd.merge(orders_df, customers_sp, on='customer_id')
orders_payments = pd.merge(orders_customers, pagamentos_agrupados, on='order_id')

# Agrupamos por cidade para obter o total de compras (quantidade e valor)
city_stats = (
    orders_payments.groupby('customer_city')
    .agg(total_count=('order_id', 'size'), total_value=('payment_value', 'sum'))
    .reset_index()
)

# Dataset de coordenadas 
city_coordinates = {
    'sao paulo': [-23.55052, -46.633308],
    'campinas': [-22.909938, -47.062633],
    'santos': [-23.960833, -46.333889],
    'sorocaba': [-23.5015, -47.4526],
    'ribeirao preto': [-21.1775, -47.8103],
   
}

# Filtrar cidades que possuem coordenadas conhecidas
city_stats_with_coords = city_stats[city_stats['customer_city'].isin(city_coordinates.keys())].copy()
city_stats_with_coords['lat'] = city_stats_with_coords['customer_city'].map(lambda c: city_coordinates[c][0])
city_stats_with_coords['lon'] = city_stats_with_coords['customer_city'].map(lambda c: city_coordinates[c][1])

# ---------------------------
# Sugerir múltiplos centros de distribuição (clusters)
# ---------------------------

# Defina o número de centros de distribuição desejados 
n_centros = 3

# Dados para clustering: latitude, longitude, ponderado pelo valor total de pedidos
coords = city_stats_with_coords[['lat', 'lon']].values
weights = city_stats_with_coords['total_value'].values

# Repetir as coordenadas proporcionalmente ao valor total (para ponderar o cluster)
rep_coords = np.repeat(coords, np.ceil(weights / weights.max() * 100).astype(int), axis=0)

# Aplicar KMeans para encontrar os centros ideais
kmeans = KMeans(n_clusters=n_centros, random_state=42, n_init=10)
kmeans.fit(rep_coords)
centros = kmeans.cluster_centers_

# ---------------------------
# Mapa com Geolocalização
# ---------------------------
map_sp = folium.Map(location=[-23.55052, -46.633308], zoom_start=7)

# Adicionar marcadores para as cidades
for _, row in city_stats_with_coords.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=6 + np.log1p(row['total_value'])/2,  # raio proporcional ao valor
        popup=f"Cidade: {row['customer_city'].title()}<br>Total de Pedidos: {row['total_count']}<br>Valor Total: R${row['total_value']:.2f}",
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(map_sp)

# Adicionar marcadores dos centros de distribuição sugeridos
for i, (lat, lon) in enumerate(centros):
    folium.Marker(
        location=[lat, lon],
        popup=f"Centro de Distribuição Sugerido #{i+1}",
        icon=folium.Icon(color='red', icon='star')
    ).add_to(map_sp)

# Salvar e abrir o mapa
map_file = "mapa_centros_distribuicao.html"
map_sp.save(map_file)
print(f"Mapa salvo como '{map_file}'.")
webbrowser.open(map_file)

# ---------------------------
# Visualização com Gráfico Interativo
# ---------------------------
# Gráfico de barras de valor total por cidade
fig = px.bar(
    city_stats_with_coords.sort_values('total_value', ascending=False),
    x='customer_city',
    y='total_value',
    title='Valor Total de Pedidos por Cidade',
    labels={'customer_city': 'Cidade', 'total_value': 'Valor Total de Pedidos (R$)'},
    hover_data={'total_count': True}
)

# Personalizar o popup (hover) para ficar mais visual
fig.update_traces(
    hovertemplate=
        '<b>Cidade:</b> %{x}<br>' +
        '<b>Valor Total:</b> R$ %{y:,.2f}<br>' +
        '<b>Total de Pedidos:</b> %{customdata[0]}<extra></extra>',
    customdata=city_stats_with_coords.sort_values('total_value', ascending=False)[['total_count']].values
)

fig.update_layout(
    xaxis_title='Cidade',
    yaxis_title='Valor Total de Pedidos (R$)',
    xaxis_tickangle=45
)
fig.show()

# Exibir tabela das cidades
print("Cidades e valores totais de pedidos:")
print(city_stats_with_coords[['customer_city', 'total_count', 'total_value']].sort_values('total_value', ascending=False))