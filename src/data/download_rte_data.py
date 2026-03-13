import os
import requests
from pathlib import Path

def download_rte_data(output_dir: str = "data/raw", filename: str = "eco2mix_national.csv"):
    """
    Télécharge les données de consommation nationale (éco2mix) définitives 
    depuis l'API OpenDataSoft (ODRÉ - RTE).
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / filename
    
    # URL d'export CSV pour les données éco2mix nationales consolidées/définitives
    # On récupère tout le dataset via l'endpoint d'export csv d'OpenDataSoft
    url = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B"
    
    print(f"Début du téléchargement (cela peut prendre quelques minutes)...")
    print(f"URL: {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    print(f"Données téléchargées avec succès : {output_path}")
    print(f"Taille du fichier : {os.path.getsize(output_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    download_rte_data()
