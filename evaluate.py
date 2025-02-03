import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_and_prepare_data


def evaluate_models(rf_model, xgb_model, cnn_model, lstm_model):
    csv_file = "/home/leovitor/Documents/Dados/features_chagas.csv"
    X_train, X_test, y_train, y_test = load_and_prepare_data(csv_file)
    
    models_results = []
    models_labels = ['Random Forest', 'XGBoost', 'CNN', 'LSTM']
    
    for model, label in zip([rf_model, xgb_model, cnn_model, lstm_model], models_labels):
        # Verificar se o modelo tem o método predict_proba
        if hasattr(model, 'predict_proba'):
            predictions = model.predict_proba(X_test)[:, 1]
        else:
            predictions = model.predict(X_test)
        
        # Calcular as métricas
        accuracy = accuracy_score(y_test, (predictions > 0.5).astype(int))
        report = classification_report(y_test, (predictions > 0.5).astype(int), output_dict=True)
        conf_mat = confusion_matrix(y_test, (predictions > 0.5).astype(int))
        auc_roc = roc_auc_score(y_test, predictions)
        fpr, tpr, _ = roc_curve(y_test, predictions)
        precision, recall, _ = precision_recall_curve(y_test, predictions)
        pr_auc = auc(recall, precision)
        
        print(f"\nResultados para {label}:")
        print(f"Acurácia: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"AUC-PR: {pr_auc:.4f}")
        print("Relatório de Classificação:")
        print(pd.DataFrame(report).T)
        print("Matriz de Confusão:")
        print(conf_mat)
        
        models_results.append({
            'auc_roc': auc_roc,
            'auc_pr': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'predictions': predictions
        })
    
    plot_roc_curves(models_results, models_labels, y_test)
    compare_confusion_matrices(models_results, models_labels, y_test)
    
    # Resumo dos resultados
    results_summary = pd.DataFrame({
        'Modelo': models_labels,
        'AUC-ROC': [result['auc_roc'] for result in models_results],
        'AUC-PR': [result['auc_pr'] for result in models_results]
    })
    print(results_summary.sort_values('AUC-ROC', ascending=False))


def plot_roc_curves(models_results, labels, y_test):
    plt.figure(figsize=(10, 8))
    for i, result in enumerate(models_results):
        plt.plot(result['fpr'], result['tpr'],
                 lw=2, label=f'{labels[i]} (AUC = %0.2f)' % result['auc_roc'])
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falso Positivo')
    plt.ylabel('Taxa de Verdadeiro Positivo')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()


def compare_confusion_matrices(models_results, labels, y_test):
    fig, axs = plt.subplots(2, 2, figsize=(20, 18))
    for i, (result, label) in enumerate(zip(models_results, labels)):
        row = i // 2
        col = i % 2
        sns.heatmap(
            confusion_matrix(y_test, (result['predictions'] > 0.5).astype(int)),
            annot=True, cmap="Blues", fmt="d", ax=axs[row, col], cbar=False
        )
        axs[row, col].set_title(label)
        axs[row, col].set_xlabel('Predito')
        axs[row, col].set_ylabel('Real')
    plt.tight_layout()
    plt.show()
