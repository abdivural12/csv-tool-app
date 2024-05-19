import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
import logging
from prophet import Prophet
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
from langchain.tools import tool
import streamlit as st
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fonction pour parser les dates
def parse_dates(df):
    try:
        df['date'] = pd.to_datetime(df['year'] * 1000 + df['day_of_year'], format='%Y%j') + pd.to_timedelta(df['minute_of_day'], unit='m')
        df.set_index('date', inplace=True)
    except Exception as e:
        logging.error("Failed to parse dates: %s", e)
        raise

# Définition des outils
@tool
def load_time_series(file_path: str) -> pd.DataFrame:
    """
    Load time series data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded time series data with parsed dates.
    """
    try:
        df = pd.read_csv(file_path)
        parse_dates(df)
        return df
    except Exception as e:
        logging.error("Error loading time series data from file %s: %s", file_path, e)
        raise

@tool
def calculate_monthly_average_temperature(file_path: str) -> pd.DataFrame:
    """
    Calculate monthly average temperature from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the monthly average temperatures.
    """
    try:
        df = pd.read_csv(file_path)
        parse_dates(df)
        monthly_avg = df.resample('M').agg({'tre200s0': 'mean'})
        return monthly_avg.reset_index()
    except Exception as e:
        logging.error("Error processing file %s: %s", file_path, e)
        raise

@tool
def kmeans_cluster_time_series(file_path: str, n_clusters: int) -> pd.DataFrame:
    """
    Apply KMeans clustering to time series data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        n_clusters (int): Number of clusters for KMeans.
    
    Returns:
        pd.DataFrame: DataFrame with cluster labels added.
    """
    try:
        df = pd.read_csv(file_path)
        parse_dates(df)
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(df[['tre200s0']].values.reshape(-1, 1))
        df['cluster'] = clusters
        return df.reset_index()
    except Exception as e:
        logging.error("Error during KMeans clustering: %s", e)
        raise

@tool
def cluster_temperatures_tslearn(file_path: str, n_clusters: int = 4) -> pd.DataFrame:
    """
    Cluster the temperatures using time series clustering with tslearn.
    
    Args:
        file_path (str): Path to the CSV file containing temperature data.
        n_clusters (int): Number of clusters for the clustering algorithm.
    
    Returns:
        pd.DataFrame: DataFrame containing the cluster labels for each time series.
    """
    try:
        data = pd.read_csv(file_path)
        parse_dates(data)
        data['month'] = data.index.month
        monthly_avg_temp = data.groupby(['name', 'month'])['tre200s0'].mean().reset_index()
        pivot_monthly_avg_temp = monthly_avg_temp.pivot(index='name', columns='month', values='tre200s0')
        pivot_monthly_avg_temp_filled = pivot_monthly_avg_temp.fillna(pivot_monthly_avg_temp.mean())
        formatted_dataset = to_time_series_dataset(pivot_monthly_avg_temp_filled.to_numpy())
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=33)
        labels = model.fit_predict(formatted_dataset)
        result_df = pd.DataFrame({'name': pivot_monthly_avg_temp_filled.index, 'cluster': labels})
        
        plt.figure(figsize=(10, 6))
        for i, center in enumerate(model.cluster_centers_):
            plt.plot(center.ravel(), label=f'Cluster {i}')
        plt.title('Centres des Clusters de Température Moyenne Mensuelle par Station')
        plt.xlabel('Mois')
        plt.ylabel('Température Moyenne (°C)')
        plt.xticks(ticks=range(12), labels=range(1, 13))
        plt.legend()
        plt.show()
        
        return result_df
    
    except Exception as e:
        logging.error("Error clustering temperatures with tslearn: %s", e)
        raise

@tool
def predict_future_temperatures(file_path: str, periods: int = 12) -> pd.DataFrame:
    """
    Predict future temperatures using the Prophet model.
    
    Args:
        file_path (str): Path to the CSV file containing temperature data.
        periods (int): Number of periods (months) to forecast into the future.
    
    Returns:
        pd.DataFrame: DataFrame containing the forecasted temperatures.
    """
    try:
        df = pd.read_csv(file_path)
        parse_dates(df)
        df = df.resample('M').agg({'tre200s0': 'mean'}).reset_index()
        df.rename(columns={'date': 'ds', 'tre200s0': 'y'}, inplace=True)
        
        model = Prophet()
        model.fit(df)
        
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        
        fig1 = model.plot(forecast)
        plt.title('Forecasted Temperatures')
        plt.xlabel('Date')
        plt.ylabel('Temperature')
        plt.show()
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    except Exception as e:
        logging.error("Error predicting future temperatures: %s", e)
        raise

# Fonction principale de l'application Streamlit
def main():
    st.title("Time Series Analysis Application")
    
    # Charger les secrets de Streamlit
    load_dotenv()

    # Récupérer la clé API depuis les secrets Streamlit
    api_key = st.secrets["OPEN_API_KEY"]
    
    # Remplacer par le chemin réel de votre fichier CSV
    file_path = "data/meteo_idaweb.csv"
    
    # Vérifier que le fichier existe
    if not os.path.isfile(file_path):
        st.error(f"No such file or directory: '{file_path}'")
        return

    # Créer et configurer l'agent CSV avec les outils
    csv_agent = create_csv_agent(
        OpenAI(api_key=api_key, temperature=0),
        file_path,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=[load_time_series, calculate_monthly_average_temperature, kmeans_cluster_time_series, cluster_temperatures_tslearn, predict_future_temperatures]
    )
    
    # Options pour l'utilisateur
    question = st.selectbox(
        "Choose a question:",
        [
            "Load the CSV data from the file",
            "Calculate monthly average temperature",
            "Apply KMeans clustering",
            "Cluster temperatures using tslearn",
            "Predict the future temperatures for the next 12 months"
        ]
    )
    
    if st.button("Ask"):
        # Utiliser l'agent pour répondre à la question
        response = csv_agent(question)
        st.write(response)

if __name__ == "__main__":
    main()
