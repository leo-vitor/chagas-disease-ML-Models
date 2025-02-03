import neurokit2 as nk
import numpy as np
import h5py
import pandas as pd

# Função para extrair características HRV
def features_extractor_from_hdf5(file_path):
    # Carregar os dados (substitua esta parte com o código necessário para carregar os dados)
    r_peaks, sampling_rate, signal = load_data_from_hdf5(file_path)
    
    # Verificar o tipo do sinal
    if isinstance(signal, dict):
        print(f"Erro: O sinal extraído está em formato de dicionário, esperado um array numérico.")
        return None
    elif not isinstance(signal, (np.ndarray, list)):
        print(f"Erro: O sinal não é um formato numérico válido, tipo encontrado: {type(signal)}.")
        return None

    # Garantir que signal é um numpy array
    signal = np.array(signal)
    
    # Verificar a forma do sinal
    print(f"Processando sinal com {len(signal)} pontos.")
    
    # Calcular as características HRV
    try:
        delineate_signals = nk.ecg_delineate(signal, sampling_rate=sampling_rate, method="dwt", show=False)
        
        # Exemplo de sinais delineados (para debug)
        print(f"Exemplo de sinais delineados (primeiras linhas):")
        print(pd.DataFrame(delineate_signals).head())
        
        # Extrair as características
        features = pd.DataFrame(delineate_signals).mean().to_dict()
        features["signal_length"] = len(signal)  # Adiciona o comprimento do sinal como feature
        
        return features
    
    except Exception as e:
        print(f"Erro ao calcular características HRV: {e}")
        return None

# Função para carregar os dados (sinal, r_peaks e taxa de amostragem) do arquivo HDF5
def load_data_from_hdf5(file_path):
    # Abre o arquivo HDF5
    try:
        with h5py.File(file_path, 'r') as hdf5_file:
            # Exemplo de como acessar os batimentos e outros dados
            # Ajuste a chave conforme o formato do seu arquivo HDF5
            beat_key = 'beat_3597'  # Exemplo de batimento a ser carregado
            signal = hdf5_file[beat_key][:]
            
            # Supondo que o arquivo contenha r_peaks e sampling_rate
            r_peaks = hdf5_file['r_peaks'][:]
            sampling_rate = hdf5_file['sampling_rate'][()]
            
            print(f"Carregando o batimento: {beat_key}")
            print(f"Forma do sinal: {signal.shape}, Taxa de amostragem: {sampling_rate}")
            
            return r_peaks, sampling_rate, signal
    
    except Exception as e:
        print(f"Erro ao carregar dados do arquivo HDF5: {e}")
        return None, None, None

# Função principal para executar o pipeline
def main():
    # Caminho do arquivo HDF5 (substitua com o caminho real)
    file_path = "caminho/do/seu/arquivo.hdf5"
    
    # Extrair as características HRV do arquivo
    file_features = features_extractor_from_hdf5(file_path)
    
    # Verificar se as características foram extraídas com sucesso
    if file_features is not None:
        print("Características HRV extraídas com sucesso:")
        print(file_features)
    else:
        print("Falha ao extrair características HRV.")

if __name__ == "__main__":
    main()
