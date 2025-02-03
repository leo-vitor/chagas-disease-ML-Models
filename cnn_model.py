import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import h5py
import json
import random

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
                    data = (data - np.mean(data)) / np.std(data)  # Normalização
                    yield tf.convert_to_tensor(data, dtype=tf.float32), label

# Criar o dataset TensorFlow a partir de arquivos
def create_tf_dataset(data_dir, split, batch_size=64, fraction=0.5):
    def generator():
        for folder in ['faleceu', 'sobreviveu_5_anos']:
            path = os.path.join(data_dir, split, folder)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.endswith('.h5')]
                if fraction < 1.0:
                    files = random.sample(files, int(len(files) * fraction))
                for file_name in files:
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
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset.repeat()  # Garantir que o dataset seja repetido

# Função para calcular os pesos fixos das classes
def calculate_class_weights():
    weights = {0: 0.7, 1: 0.3}
    return weights

# Definir o modelo simplificado
def create_model(input_shape=(2400, 1)):
    model = Sequential([
        Conv1D(32, kernel_size=10, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),

        Conv1D(64, kernel_size=10, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.5),

        Conv1D(128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dropout(0.5),

        Dense(128, activation='relu'),  # Reduzido de 256
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.summary()
    return model

# Treinar o modelo
def train_model(data_dir, model_dir='saved_models', batch_size=64, epochs=10):
    input_shape = (2400, 1)
    train_dataset = create_tf_dataset(data_dir, 'train', batch_size)  # Batch size aumentado
    val_dataset = create_tf_dataset(data_dir, 'val', batch_size)
    
    class_weights = calculate_class_weights()
    print(f"Class weights: {class_weights}")
    
    model = create_model(input_shape)
    
    optimizer = Adam(learning_rate=0.001)  # Taxa de aprendizado inicial mais alta
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
    
    os.makedirs(model_dir, exist_ok=True)
    callbacks = [
        ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'), save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)  # Redução mais agressiva da taxa de aprendizado
    ]
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history, val_dataset

# Função para calcular o AUC pós-treinamento
def calculate_auc(model, val_dataset):
    y_true, y_pred = [], []
    for x, y in val_dataset.unbatch():
        y_true.append(y.numpy())
        y_pred.append(model.predict(x[None, ...])[0][0])
    auc_score = roc_auc_score(y_true, y_pred)
    print(f"AUC Score: {auc_score}")
    return auc_score

# Função para salvar as métricas em um arquivo JSON
def save_metrics(history, output_path='training_metrics.json'):
    metrics = {
        'train': {key: values for key, values in history.history.items() if not key.startswith('val_')},
        'val': {key[4:]: values for key, values in history.history.items() if key.startswith('val_')}
    }
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métricas salvas em {output_path}")

# Função para visualizar o desempenho e salvar os gráficos
def plot_metrics(history, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['accuracy', 'loss', 'auc', 'precision']
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history[metric], label=f'Treino {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validação {metric}')
        plt.title(f'Evolução de {metric.capitalize()} durante o treinamento')
        plt.xlabel('Época')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid()
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'{metric}.png')
        plt.savefig(save_path)
        print(f'Gráfico de {metric} salvo em: {save_path}')
        plt.close()

# Função principal atualizada
def main(data_dir):
    model, history, val_dataset = train_model(data_dir)
    save_metrics(history, output_path='training_metrics.json')
    plot_metrics(history, output_dir='plots')
    auc_score = calculate_auc(model, val_dataset)
    print(f"AUC calculado: {auc_score}")
    print("Treinamento concluído.")
    os.system("shutdown now")

if __name__ == "__main__":
    data_dir = '/home/leovitor/Documents/Dados/chagas_processed/'
    main(data_dir)
