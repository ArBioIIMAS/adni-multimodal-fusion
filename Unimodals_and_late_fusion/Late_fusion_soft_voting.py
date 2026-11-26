#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from skorch import NeuralNetClassifier
from Joint_Search_Uni_kc_r2 import DynamicMLP

# Configuración
MODALIDADES = ['DTI', 'Metabólicos', 'Morfológicos']
ALGORITMOS = ['CatBoost', 'RF', 'SVM', 'KNN', 'ElasticNet', 'MLP']
CLASS_NAMES = ['CN', 'DCL', 'EA'] 

def evaluar_fusion_multiclase(FUSION):
    # 1. Cargar ganadores
    winners = {m: pd.read_csv(f'{m}_modelos_seleccionados.csv') for m in MODALIDADES}
    
    # 2. Cargar datos de test (Asegúrate de que el orden sea consistente)
    data_test = {m: pd.read_csv(f'../data/test_{m}.csv') for m in MODALIDADES}
    y_test = data_test['DTI']['Target_class'].values
    classes = np.unique(y_test)
    n_classes = len(classes)
    
    # Binarizar etiquetas para ROC multiclase
    y_test_bin = label_binarize(y_test, classes=classes)
    if n_classes == 2: # Ajuste para caso binario
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    rows_metrics = []

    for algo in ALGORITMOS:
        print(f"Procesando Fusión: {algo}...")
        all_probs = []

        for modo in MODALIDADES:
            fila = winners[modo][winners[modo]['Modelo'] == algo]
            if fila.empty: continue
            
            # Cargar modelo de la modalidad
            seed = fila['Seed'].values[0]
            model = joblib.load(f"{modo}/modelo_{algo}_s{seed}.joblib")
            
            X_test = data_test[modo].drop(columns=['Target_class', 'Sujeto'], errors='ignore')
            # Forzamos conversión explícita a float32 para compatibilidad con PyTorch/skorch
            X_test_vals = X_test.values.astype(np.float32)

            # Obtener probabilidades
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test_vals)
                print('Se utiliza model.predict_proba')
            else:
                d = model.decision_function(X_test_vals)
                probs = softmax(d, axis=1)
                print('Se utiliza model.decision_function')

            all_probs.append(probs)


        # 3. Fusión Tardía (Promedio de Probabilidades)
        avg_probs = np.mean(all_probs, axis=0)
        y_pred = np.argmax(avg_probs, axis=1)

        # Convertimos la lista de probabilidades en un array de 3 dimensiones y promediamos
        # Esto nos da la "certeza promedio" del consejo de expertos
        avg_probs = np.mean(all_probs, axis=0)

        # Guardar predicciones y probabilidades
        # Creamos el diccionario base
        data_detalle = {
            'Sujeto': data_test['DTI']['Sujeto'],
            'Target_Real': y_test,
            'Pred_Final': y_pred
        }
        # Añadimos las probabilidades de CADA modalidad por separado
        # all_probs es una lista: [Probs_DTI, Probs_Meta, Probs_Morf]
        for i, modo in enumerate(MODALIDADES):
            probs_modo = all_probs[i]
            for c in range(n_classes):
                # Ejemplo de columna: Prob_EA_DTI
                data_detalle[f'Prob_{CLASS_NAMES[c]}_{modo}'] = probs_modo[:, c]
        # Añadimos también el promedio (la voz del consenso)
        for c in range(n_classes):
            data_detalle[f'Prob_{CLASS_NAMES[c]}_AVG'] = avg_probs[:, c]
        # Convertir a DataFrame y guardar
        df_detalle = pd.DataFrame(data_detalle)
        archivo_salida = f'Detalle_Sujetos_{algo}_{FUSION}.csv'
        df_detalle.to_csv(archivo_salida, index=False)
        
        # 4. Métricas con classification_report (formato dict para CSV)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Estructura para el CSV
        res_row = {
            'Algoritmo': algo,
            'Acc_Global': accuracy_score(y_test, y_pred),
            'Macro_F1': report['macro avg']['f1-score'],
            'Macro_Prec': report['macro avg']['precision'],
            'Macro_Rec': report['macro avg']['recall']
        }
        
        # Añadir métricas individuales por clase dinámicamente
        for i, c_name in enumerate(CLASS_NAMES):
            c_str = str(i)
            res_row[f'F1_Clase_{c_name}'] = report[c_str]['f1-score']
            res_row[f'Prec_Clase_{c_name}'] = report[c_str]['precision']
            res_row[f'Rec_Clase_{c_name}'] = report[c_str]['recall']

        rows_metrics.append(res_row)

        # 5. Matriz de Confusión
        plt.figure(figsize=(6,5))
        #sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='viridis')
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f'Matriz Fusión {algo}')
        plt.savefig(f'{FUSION}/cm_fusion_{algo}.png')
        plt.close()

        # 6. Curva ROC Multiclase (One-vs-Rest)
        plt.figure()
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], avg_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Clase {CLASS_NAMES[i]} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Multiclase - {algo}')
        plt.legend()
        plt.savefig(f'{FUSION}/roc_fusion_{algo}.png')
        plt.close()

    # 7. Exportar a CSV
    pd.DataFrame(rows_metrics).to_csv(f'Metricas_{FUSION}_Final.csv', index=False)
    print(f"CSV generado: Metricas_{FUSION}_Final.csv")

if __name__ == '__main__':
    FUSION = 'Soft_voting'
    if not os.path.exists(FUSION):
        os.makedirs(FUSION)
    evaluar_fusion_multiclase(FUSION)


# In[ ]:




