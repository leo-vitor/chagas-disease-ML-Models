import os
import tensorflow as tf
import numpy as np
import h5py
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Função para carregar os dados do arquivo HDF5
def load_hdf5_data_generator(file_path):
    with h5py.File(file_path, 'r') as hf:
        label = 1 if 'faleceu' in file_path else 0
        for dataset_name in hf.keys():
            dataset = hf[dataset_name]
            if isinstance(dataset, h5py.Dataset):
                try:
                    data = np.array(dataset)
                except Exception as e:
                    print(f"Erro ao carregar dataset {dataset_name} no arquivo {file_path}: {e}")
                    continue
                if len(data.shape) == 1 and data.shape[0] == 2400:
                    pass
                else:
                    print(f"Warning: Dataset {dataset_name} tem forma inválida: {data.shape}. Ignorando.")
                    continue
                yield tf.convert_to_tensor(data, dtype=tf.float32), label

# Criar o dataset TensorFlow a partir de arquivos
def create_tf_dataset(data_dir, split, batch_size=64):
    def generator():
        for folder in ['faleceu', 'sobreviveu_5_anos']:
            path = os.path.join(data_dir, split, folder)
            if os.path.exists(path):
                for file_name in os.listdir(path):
                    if file_name.endswith('.h5'):
                        file_path = os.path.join(path, file_name)
                        try:
                            for data, label in load_hdf5_data_generator(file_path):
                                data = tf.expand_dims(data, axis=-1)
                                yield data, label
                        except Exception as e:
                            print(f"Erro ao processar arquivo {file_path}: {e}")
            else:
                print(f"Diretório {path} não encontrado. Ignorando.")
    
    output_signature = (
        tf.TensorSpec(shape=(2400, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE).repeat()

# Definir o modelo LSTM
# No arquivo lstm_model.py

def create_lstm_model(input_shape=(2400, 1)):
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        LSTM(64, activation='tanh', return_sequences=True),
        BatchNormalization(),
        Dropout(0.5),

        LSTM(128, activation='tanh', return_sequences=True),
        BatchNormalization(),
        LSTM(128, activation='tanh', return_sequences=True),
        BatchNormalization(),
        Dropout(0.5),

        GlobalAveragePooling1D(),
        Dropout(0.5),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model


# Função para treinar o modelo LSTM
def train_lstm_model(data_dir, model_dir='saved_models', batch_size=64, epochs=20):
    input_shape = (2400, 1)
    
    # Criar datasets
    train_dataset = create_tf_dataset(data_dir, 'train', batch_size)
    val_dataset = create_tf_dataset(data_dir, 'val', batch_size)
    test_dataset = create_tf_dataset(data_dir, 'test', batch_size)
    
    # Pesos das classes
    class_weight_dict = {0: 0.3, 1: 0.7}
    print(f"Class weights: {class_weight_dict}")
    
    # Definir o modelo
    model = create_lstm_model(input_shape)
    
    # Compilar o modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
    
    # Configuração dos callbacks
    os.makedirs(model_dir, exist_ok=True)
    
    checkpoint = ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'), monitor='val_loss', save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6) 
    
    # Treinamento do modelo
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weight_dict,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Retornar o modelo e o histórico para compatibilidade com a pipeline
    return model, history

# Função principal para ser chamada no pipeline
def main_lstm(data_dir):
    model, history = train_lstm_model(data_dir)
    
    # Gerar a acurácia no conjunto de teste
    test_dataset = create_tf_dataset(data_dir, 'test', batch_size=64)
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(test_dataset)
    
    # Gerar as métricas de classificação no conjunto de teste
    y_true = []
    y_pred = []
    
    for x_batch, y_batch in test_dataset:
        y_true.extend(y_batch.numpy())
        y_pred.extend(model.predict(x_batch))
    
    # Calcular as métricas de avaliação
    y_pred_bin = (np.array(y_pred) > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_bin)
    conf_matrix = confusion_matrix(y_true, y_pred_bin)
    precision = precision_score(y_true, y_pred_bin)
    recall = recall_score(y_true, y_pred_bin)
    auc = roc_auc_score(y_true, y_pred)
    
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test precision: {test_precision:.4f}")
    print(f"Test recall: {test_recall:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    # Retornar as métricas para a pipeline
    return {
        'model': model,
        'history': history,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'precision': precision,
        'recall': recall,
        'auc': auc,
    }
