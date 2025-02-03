import os
import shutil
import pandas as pd
import numpy as np
import h5py
import neurokit2 as nk
import scipy.signal as signal
from sklearn.model_selection import train_test_split

def gerar_lista_falecer(caminho_planilha):
    """
    Gera uma lista de arquivos da classe 'faleceu' com base nas condições:
    - Falecimento por morte súbita cardíaca (MSC)
    - Dentro de 5 anos
    """
    df = pd.read_excel(caminho_planilha)
    filtered_ids = df.loc[
        (df.iloc[:, 7] == 1) & 
        (df.iloc[:, 75].str.contains('MSC', case=False, na=False)),
        df.columns[0]
    ]
    return [f"sinal{int(id_):03d}.txt" for id_ in filtered_ids]

def listar_arquivos(pasta_origem):
    """Lista todos os arquivos disponíveis na pasta de origem."""
    return [arquivo for arquivo in os.listdir(pasta_origem) if os.path.isfile(os.path.join(pasta_origem, arquivo))]

def separar_arquivos(pasta_origem, lista_falecer):
    """
    Organiza os arquivos em pastas específicas de acordo com sua classe.
    - 'faleceu'
    - 'sobreviveu_5_anos'
    """
    pasta_faleceu = os.path.join(pasta_origem, "faleceu")
    pasta_sobreviveu = os.path.join(pasta_origem, "sobreviveu_5_anos")
    os.makedirs(pasta_faleceu, exist_ok=True)
    os.makedirs(pasta_sobreviveu, exist_ok=True)

    arquivos_disponiveis = listar_arquivos(pasta_origem)
    for arquivo in arquivos_disponiveis:
        caminho_origem = os.path.join(pasta_origem, arquivo)
        destino = pasta_faleceu if arquivo in lista_falecer else pasta_sobreviveu
        shutil.move(caminho_origem, os.path.join(destino, arquivo))

def normalize_signal(ecg_signal):
    """Normaliza o sinal ECG: z-score."""
    return (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

def lowpass_filter(ecg_signal, sample_rate, cutoff=40):
    """Aplica um filtro passa-baixa ao sinal ECG."""
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(1, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, ecg_signal)

def median_filter(ecg_signal, window_size=5):
    """Aplica um filtro mediano ao sinal ECG."""
    return signal.medfilt(ecg_signal, kernel_size=window_size)

def process_ecg_file(file_path, output_folder, set_type, class_name, window_size=2400, sample_rate=480):
    """
    Processa um arquivo de ECG e salva as janelas não sobrepostas em arquivos HDF5.
    Garante que todas as janelas tenham o tamanho máximo possível.
    """
    try:
        # Obtém o nome do arquivo sem a extensão
        file_name = os.path.basename(file_path).split('.')[0]
        
        # Caminho onde o arquivo HDF5 será salvo
        set_folder = os.path.join(output_folder, set_type, class_name)
        os.makedirs(set_folder, exist_ok=True)
        
        # Verifica se o arquivo .h5 já existe
        h5_file_path = os.path.join(set_folder, f'{file_name}.h5')
        if os.path.exists(h5_file_path):
            print(f"Arquivo {h5_file_path} já existe. Pulando processamento.")
            return

        # Carrega e processa o sinal ECG
        ecg_signal = np.loadtxt(file_path)
        filtered_signal = lowpass_filter(ecg_signal, sample_rate)
        normalized_signal = normalize_signal(filtered_signal)
        filtered_signal_median = median_filter(normalized_signal)
        
        remaining_length = len(filtered_signal_median)
        segments = []

        while remaining_length >= window_size:
            segment = filtered_signal_median[:window_size]
            segments.append(segment)
            remaining_length -= window_size

        # Se houver resto, cria uma última janela com o tamanho restante
        if remaining_length > 0:
            last_segment = filtered_signal_median[-remaining_length:]
            segments.append(last_segment)

        # Salva os segmentos processados
        save_segments(segments, output_folder, set_type, class_name, file_name)

    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")

def save_segments(segments, output_folder, set_type, class_name, file_name):
    """
    Salva os segmentos processados em arquivos HDF5.
    """
    set_folder = os.path.join(output_folder, set_type, class_name)
    os.makedirs(set_folder, exist_ok=True)
    with h5py.File(os.path.join(set_folder, f'{file_name}.h5'), 'w') as f:
        for i, segment in enumerate(segments):
            f.create_dataset(f'segment_{i}', data=segment.astype(np.float32))

def process_all_ecg_files(input_folder, output_folder):
    """
    Processa todos os arquivos de ECG organizados nas classes:
    - 'faleceu'
    - 'sobreviveu_5_anos'
    """
    for class_name in ['faleceu', 'sobreviveu_5_anos']:
        folder_path = os.path.join(input_folder, class_name)
        if not os.path.exists(folder_path):
            continue

        # Listar os arquivos da classe e dividir entre os conjuntos
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
        train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

        # Processar os arquivos de cada conjunto
        for file_path in train_files:
            process_ecg_file(file_path, output_folder, 'train', class_name)

        for file_path in val_files:
            process_ecg_file(file_path, output_folder, 'val', class_name)

        for file_path in test_files:
            process_ecg_file(file_path, output_folder, 'test', class_name)

if __name__ == "__main__":
    caminho_planilha = '/home/leovitor/Documents/Dados/Dataset_Unificado_r1.xlsx'
    pasta_origem = '/home/leovitor/Documents/Dados/chagas_msc'
    pasta_saida = '/home/leovitor/Documents/Dados/chagas_processed'

    lista_falecer = gerar_lista_falecer(caminho_planilha)
    separar_arquivos(pasta_origem, lista_falecer)
    process_all_ecg_files(pasta_origem, pasta_saida)

