import h5py
import numpy as np
import random
import neurokit2 as nk
import pandas as pd

# Função para listar as chaves (datasets) de um arquivo HDF5
def list_hdf5_keys(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())  # Listar as chaves dos datasets (batimentos)
            print(f"Chaves no arquivo {file_path}: {keys}")
            return keys
    except Exception as e:
        print(f"Erro ao acessar o arquivo {file_path}: {e}")
        return []

# Função para imprimir um batimento aleatório de um arquivo HDF5
def print_random_beat(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            # Listar as chaves (datasets de batimentos)
            keys = list(f.keys())
            if not keys:
                print(f"O arquivo {file_path} não contém dados.")
                return

            # Selecionar um batimento aleatório
            random_key = random.choice(keys)
            signal_data = f[random_key][...]  # Carregar os dados do batimento
            print(f"Batimento aleatório ({random_key}):")
            print(signal_data[:10])  # Exibir os primeiros 10 pontos do batimento
            return signal_data
    except Exception as e:
        print(f"Erro ao acessar o arquivo {file_path}: {e}")
        return None

# Função para tentar extrair características do batimento
def extract_features_from_beat(beat, sampling_rate=480):
    try:
        # Limpeza do sinal ECG
        ecg_cleaned = nk.ecg_clean(beat, sampling_rate=sampling_rate)
        
        # Delineamento do ECG (localização de ondas P, QRS, T)
        delineate_signals = nk.ecg_delineate(ecg_cleaned, sampling_rate=sampling_rate, method="dwt", show=False)
        
        # Extrair as características do ECG
        features = pd.DataFrame(delineate_signals).mean().to_dict()
        features["signal_length"] = len(beat)  # Adicionar o comprimento do sinal como característica
        
        return features
    except Exception as e:
        print(f"Erro ao extrair características: {e}")
        return None

# Função principal para rodar o pipeline
def main():
    # Defina o caminho para um arquivo HDF5 específico
    file_path = "/home/leovitor/Documents/Dados/chagas_processed/train/faleceu/sinal108.h5"  # Substitua pelo caminho real
    
    # Listar as chaves (datasets) no arquivo
    print("Listando as chaves do arquivo HDF5...")
    list_hdf5_keys(file_path)
    
    # Imprimir um batimento aleatório
    print("Selecionando um batimento aleatório...")
    beat = print_random_beat(file_path)
    
    if beat is not None:
        # Tentar extrair características do batimento
        print("Extraindo características do batimento...")
        features = extract_features_from_beat(beat)
        
        if features:
            print("Características extraídas com sucesso:")
            print(features)
        else:
            print("Falha ao extrair características do batimento.")

if __name__ == "__main__":
    main()
