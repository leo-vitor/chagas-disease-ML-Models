import os
import h5py
import matplotlib.pyplot as plt

def listar_arquivos_h5(pasta_origem):
    """
    Lista todos os arquivos .h5 na pasta de origem.
    """
    return [arquivo for arquivo in os.listdir(pasta_origem) if arquivo.endswith('.h5')]

def carregar_e_visualizar_batimentos(pasta_arquivos, arquivo_h5):
    """
    Carrega os batimentos do arquivo .h5 e os visualiza.
    """
    caminho_arquivo = os.path.join(pasta_arquivos, arquivo_h5)
    
    # Abrindo o arquivo HDF5
    with h5py.File(caminho_arquivo, 'r') as f:
        # Listando os datasets dentro do arquivo
        batimentos = list(f.keys())
        print(f"Batimentos encontrados no arquivo {arquivo_h5}: {batimentos}")

        # Visualizando alguns batimentos
        for i, batimento in enumerate(batimentos[:5]):  # Visualiza os 5 primeiros batimentos
            dados_batimento = f[batimento][:]
            plt.figure(figsize=(10, 4))
            plt.plot(dados_batimento)
            plt.title(f"Batimento {i+1} - {arquivo_h5}")
            plt.xlabel("Amostras")
            plt.ylabel("Amplitude")
            plt.show()

def visualizar_batimentos(pasta_saida):
    """
    Função principal para visualizar os batimentos nos arquivos .h5.
    """
    # Listando todos os arquivos .h5 nas pastas de treino, validação e teste
    for set_type in ['train', 'val', 'test']:
        for class_name in ['faleceu', 'sobreviveu_5_anos']:
            pasta_arquivos = os.path.join(pasta_saida, set_type, class_name)
            
            if not os.path.exists(pasta_arquivos):
                continue

            arquivos_h5 = listar_arquivos_h5(pasta_arquivos)

            for arquivo_h5 in arquivos_h5:
                carregar_e_visualizar_batimentos(pasta_arquivos, arquivo_h5)

if __name__ == "__main__":
    # Defina a pasta onde os arquivos .h5 foram salvos
    pasta_saida = '/home/leovitor/Documents/Dados/chagas_processed'

    # Visualizar os batimentos
    visualizar_batimentos(pasta_saida)
