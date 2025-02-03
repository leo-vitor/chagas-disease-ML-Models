import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_and_prepare_data  # Importando a função de utils.py

# Função para treinar o modelo XGBoost
def treinar_xgboost(X_train, X_test, y_train, y_test):
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    return xgb_model, accuracy, report, conf_mat

# Função principal
def main():
    csv_file = "/home/leovitor/Documents/Dados/features_chagas.csv"
    
    # Usando a função de utils.py para carregar e preparar os dados
    X_train, X_test, y_train, y_test = load_and_prepare_data(csv_file)
    
    # Treinando o modelo
    xgb_model, accuracy, report, conf_mat = treinar_xgboost(X_train, X_test, y_train, y_test)
    
    # Imprimindo resultados
    print(f"Acurácia: {accuracy:.4f}")
    print("Relatório de Classificação:")
    print(report)
    
    # Plotando a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt="d")
    plt.title("Matriz de Confusão - XGBoost")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()

if __name__ == "__main__":
    main()
