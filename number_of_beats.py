import os
import h5py
import numpy as np 
def count_beats_in_folder(folder_path):
    """
    Conta o número total de batimentos (beats) em uma pasta. 
    Cada arquivo HDF5 pode conter vários batimentos.
    """
    total_beats = 0
    
    if not os.path.exists(folder_path):
        print(f"Pasta não encontrada: {folder_path}")
        return total_beats

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.h5'):
            file_path = os.path.join(folder_path, file_name)
            try:
                with h5py.File(file_path, 'r') as hf:
                    # Contar batimentos em cada dataset no arquivo HDF5
                    for dataset_name in hf.keys():
                        dataset = hf[dataset_name]
                        if isinstance(dataset, h5py.Dataset):
                            data = np.array(dataset)
                            if len(data.shape) == 1 and data.shape[0] == 2400:  # Verificar formato esperado
                                total_beats += 1
            except Exception as e:
                print(f"Erro ao processar {file_path}: {e}")
    
    return total_beats

def count_beats_in_dataset(data_dir):
    """
    Conta o número total de batimentos nas pastas de treino, validação e teste,
    separando pelas classes.
    """
    results = {}
    splits = ['train', 'val', 'test']
    classes = ['faleceu', 'sobreviveu_5_anos']

    for split in splits:
        results[split] = {}
        for class_folder in classes:
            folder_path = os.path.join(data_dir, split, class_folder)
            count = count_beats_in_folder(folder_path)
            results[split][class_folder] = count
    
    return results

# Caminho para os dados
data_dir = '/home/leovitor/Documents/Dados/chagas_processed/'

# Contar batimentos
beat_counts = count_beats_in_dataset(data_dir)

# Exibir os resultados
print("Contagem de batimentos por pasta:")
for split, classes in beat_counts.items():
    print(f"\n{split.capitalize()}:")
    for class_folder, count in classes.items():
        print(f"  {class_folder}: {count} batimentos")
