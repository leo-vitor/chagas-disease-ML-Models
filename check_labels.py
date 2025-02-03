import os
import h5py
import numpy as np

def check_labels(data_dir, split):
    """
    Função para verificar os rótulos dos arquivos em uma divisão específica (train, val ou test).
    
    Args:
        data_dir (str): Diretório onde os dados estão armazenados.
        split (str): Divisão dos dados ('train', 'val' ou 'test').
    
    Returns:
        None
    """
    # Diretórios esperados para as classes
    classes = {'faleceu': 1, 'sobreviveu_5_anos': 0}
    labels_count = {label: 0 for label in classes.values()}
    
    for class_name, label in classes.items():
        class_dir = os.path.join(data_dir, split, class_name)
        if not os.path.exists(class_dir):
            print(f"Diretório não encontrado: {class_dir}")
            continue
        
        print(f"\nArquivos no diretório: {class_dir}")
        
        for file_name in os.listdir(class_dir):
            if file_name.endswith('.h5'):
                file_path = os.path.join(class_dir, file_name)
                try:
                    with h5py.File(file_path, 'r') as hf:
                        datasets = list(hf.keys())
                        print(f"Arquivo: {file_name}, Rótulo: {label}, Datasets: {datasets}")
                        labels_count[label] += 1
                except Exception as e:
                    print(f"Erro ao carregar o arquivo {file_path}: {e}")
    
    print("\nResumo dos rótulos:")
    for label, count in labels_count.items():
        print(f"Rótulo {label}: {count} arquivos")


# Diretório principal onde os dados estão armazenados
data_dir = "/home/leovitor/Documents/Dados/chagas_processed/"

# Chamar a função para verificar os rótulos em cada divisão
for split in ['train', 'val', 'test']:
    print(f"\n=== Verificando rótulos para o conjunto {split} ===")
    check_labels(data_dir, split)
