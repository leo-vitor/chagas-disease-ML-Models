import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(csv_file):
    """
    Carrega os dados de um arquivo CSV, separa as variáveis independentes (X) e a variável alvo (y),
    e divide os dados em conjuntos de treinamento e teste.
    
    Args:
    - csv_file (str): Caminho para o arquivo CSV contendo os dados.

    Returns:
    - X_train_scaled: Dados de treino normalizados.
    - X_test_scaled: Dados de teste normalizados.
    - y_train: Rótulos de treino.
    - y_test: Rótulos de teste.
    """
    # Carregar os dados
    df = pd.read_csv(csv_file)
    
    # Verificar se as colunas esperadas estão presentes
    required_columns = ['file_name', 'class'] + [col for col in df.columns if col not in ['file_name', 'class']]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("O arquivo CSV está faltando uma ou mais colunas necessárias.")
    
    # Tratar valores nulos (caso existam)
    df = df.dropna()  # Alternativamente, você pode usar df.fillna() para imputação

    # Separar variáveis independentes (X) e dependentes (y)
    X = df.drop(['file_name', 'class'], axis=1)
    y = df['class'].map({'faleceu': 1, 'sobreviveu_5_anos': 0})
    
    # Dividir em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalização dos dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
