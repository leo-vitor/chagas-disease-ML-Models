import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_and_prepare_data  # Função externa para carregar e preparar os dados


def train_random_forest(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    return rf_model, accuracy, report, conf_mat


def main():
    csv_file = "/home/leovitor/Documents/Dados/features_chagas.csv"
    
    # Carregar e preparar os dados
    X_train, X_test, y_train, y_test = load_and_prepare_data(csv_file)
    
    # Treinar modelo Random Forest
    rf_model, accuracy, report, conf_mat = train_random_forest(X_train, X_test, y_train, y_test)
    
    # Imprimir resultados
    print(f"Acurácia: {accuracy:.4f}")
    print("Relatório de Classificação:")
    print(report)
    
    # Plotar matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt="d")
    plt.title("Matriz de Confusão - Random Forest")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()


if __name__ == "__main__":
    main()
