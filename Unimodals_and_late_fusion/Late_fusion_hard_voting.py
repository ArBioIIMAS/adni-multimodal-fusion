#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from scipy import stats
from scipy.special import softmax
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
        preds_list = []

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
            preds_list.append(model.predict(X_test_vals))
        
        #print("preds_list shape (as np array):", np.array(preds_list).shape)
        #print("all_probs shape (as np array):", np.array(all_probs).shape)
        
        # 3. Fusión Tardía (Hard voting)
        preds_array = np.array(preds_list) # Shape: (modalidades, sujetos)
        y_pred, _ = stats.mode(preds_array, axis=0)
        y_pred = y_pred.flatten()


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

        # 6. Eliminamos el bloque de ROC en hard voting y sustituir por
        # una figura que sí es coherente con la decisión por moda:
        # un bar chart de accuracy por clase, que es lo que el voto duro optimiza
        
        class_acc = {}
        for i, c_name in enumerate(CLASS_NAMES):
            mask = y_test == i
            class_acc[c_name] = accuracy_score(y_test[mask], y_pred[mask])
        
        plt.figure(figsize=(5, 4))
        plt.bar(class_acc.keys(), class_acc.values(), color=['steelblue','darkorange','green'])
        plt.ylim(0, 1)
        plt.title(f'Accuracy por clase — Hard Voting {algo}')
        plt.ylabel('Accuracy')
        plt.savefig(f'{FUSION}/acc_clase_{algo}.png')
        plt.close()

    # 7. Exportar a CSV
    pd.DataFrame(rows_metrics).to_csv(f'Metricas_{FUSION}_Final.csv', index=False)
    print(f"CSV generado: Metricas_{FUSION}_Final.csv")

if __name__ == '__main__':
    FUSION = 'Hard_voting'
    if not os.path.exists(FUSION):
        os.makedirs(FUSION)
    evaluar_fusion_multiclase(FUSION)


# In[ ]:




