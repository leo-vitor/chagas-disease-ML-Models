import os
import pandas as pd
from cnn_model import create_model as create_cnn_model
# from lstm_model import create_lstm_model  # Comentar a importação do modelo LSTM
# from sklearn.ensemble import RandomForestClassifier  # Comentar a importação do modelo RandomForest
# from xgboost import XGBClassifier  # Comentar a importação do modelo XGBoost
from utils import load_and_prepare_data
# from features_extractor import features_extractor_from_hdf5  # Comentar a extração de features de HDF5
from evaluate import evaluate_models

def main():
    # Caminho para os arquivos processados
    processed_data_folder = '/home/leovitor/Documents/Dados/chagas_processed'

    # Extrair as features dos batimentos processados
    features = []
    labels = []
    
    # Carregar arquivos HDF5 e extrair as features (apenas para a CNN)
    for class_name in ['faleceu', 'sobreviveu_5_anos']:
        class_folder = os.path.join(processed_data_folder, 'train', class_name)
        for file_name in os.listdir(class_folder):
            if file_name.endswith('.h5'):
                file_path = os.path.join(class_folder, file_name)
                # Extrair features do arquivo HDF5 (focar apenas na CNN)
                file_features = features_extractor_from_hdf5(file_path)
                features.append(file_features)
                labels.extend([class_name] * len(file_features))  # Adicionar as labels (faleceu ou sobreviveu)
    
    # Converter a lista de features em DataFrame
    features_df = pd.concat(features, ignore_index=True)

    # Preparar os dados para o modelo de aprendizado de máquina
    X_train, X_test, y_train, y_test = load_and_prepare_data(features_df, labels)
    
    # Inicializar o modelo de CNN
    cnn_model = create_cnn_model(input_shape=X_train.shape[1:])
    
    # Treinamento do modelo de CNN
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    
    # Avaliar o modelo CNN
    # Como os outros modelos foram comentados, podemos apenas avaliar a CNN
    evaluate_models(cnn_model)  # Agora, apenas a CNN será avaliada
    
    # Salvar o modelo treinado da CNN
    save_model(cnn_model, "cnn_model")

def save_model(model, model_name):
    """
    Função para salvar o modelo treinado em disco.
    
    Args:
    - model: O modelo a ser salvo.
    - model_name (str): Nome do modelo para o arquivo.
    """
    model_dir = "/home/leovitor/Documents/Modelos/"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    
    if hasattr(model, 'save'):
        model.save(model_path)  # Para modelos de DL
    else:
        import joblib
        joblib.dump(model, model_path)  # Para modelos de ML
    
    print(f"Modelo {model_name} salvo em: {model_path}")

if __name__ == "__main__":
    main()
