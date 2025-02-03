import h5py

# Caminho para o arquivo .h5
file_path = '/home/leovitor/Documents/Dados/chagas_processed/val/faleceu/sinal006.h5'

# Abrir o arquivo .h5
with h5py.File(file_path, 'r') as hf:
    # Exibir todas as chaves (datasets) no arquivo HDF5
    print("Chaves no arquivo HDF5:", list(hf.keys()))
    
    # Explorar as estruturas e dados
    for key in hf.keys():
        print(f"\nExplorando o dataset '{key}':")
        dataset = hf[key]
        
        # Mostrar as informações sobre o dataset
        print(f"  - Tipo: {type(dataset)}")
        print(f"  - Forma: {dataset.shape}")
        print(f"  - Tipo de dado: {dataset.dtype}")
        print(f"  - Primeiros 5 elementos: {dataset[:5]}")
        
        # Caso o dataset seja grande, imprima as primeiras linhas para verificar
        if dataset.shape[0] > 5:
            print(f"  - Exemplos adicionais: {dataset[5:10]}")
