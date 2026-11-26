#!/usr/bin/env python
# coding: utf-8

# # Late Fusion — Meta-modelo (Stacking)
# 
# Tres estrategias:
# - **`por_algoritmo`**: un mismo algoritmo en las 3 modalidades (itera sobre todos los algoritmos disponibles)
# - **`mejores_por_modalidad`**: el mejor modelo por CV_Acc de cada modalidad
# - **`comite_topN`**: los N mejores de cada modalidad (Top-2 y Top-3)
# 
# El CSV de salida sigue el mismo formato que `Metricas_Soft_voting_Final.csv`.

# In[1]:


import pandas as pd
import numpy as np
import joblib
import csv
import os
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.base import clone
import torch
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif
from skorch import NeuralNetClassifier
from Joint_Search_Uni_kc_r2 import DynamicMLP


  
# ── Configuración ─────────────────────────────────────────────────────────────
MODALIDADES = ['DTI', 'Metabólicos', 'Morfológicos']
CLASS_NAMES = {0: 'CN', 1: 'DCL', 2: 'EA'}

# ── Helper: guardar métricas en formato homologado ────────────────────────────
def metricas_a_fila(algoritmo, y_test, y_pred):
    """
    Convierte classification_report a una sola fila (mismo formato que
    Metricas_Soft_voting_Final.csv y Metricas_Hard_voting_Final.csv).
    """
    r = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    fila = {
        'Algoritmo':    algoritmo,
        'Acc_Global':   accuracy_score(y_test, y_pred),
        'Macro_F1':     r['macro avg']['f1-score'],
        'Macro_Prec':   r['macro avg']['precision'],
        'Macro_Rec':    r['macro avg']['recall'],
    }
    # Métricas por clase
    for cls_id, cls_name in CLASS_NAMES.items():
        key = str(cls_id)
        if key in r:
            fila[f'F1_Clase_{cls_name}']   = r[key]['f1-score']
            fila[f'Prec_Clase_{cls_name}'] = r[key]['precision']
            fila[f'Rec_Clase_{cls_name}']  = r[key]['recall']
    return fila


# ── Helper: cargar datos ──────────────────────────────────────────────────────
def cargar_datos():
    train_data = {m: pd.read_csv(f'../data/train_{m}.csv') for m in MODALIDADES}
    test_data  = {m: pd.read_csv(f'../data/test_{m}.csv')  for m in MODALIDADES}
    winners    = {m: pd.read_csv(f'{m}_modelos_seleccionados.csv') for m in MODALIDADES}
    y_train = train_data['DTI']['Target_class'].values
    y_test  = test_data['DTI']['Target_class'].values
    return train_data, test_data, winners, y_train, y_test


# ── Helper: construir matrices meta (OOF train + test) ────────────────────────
def construir_matrices_meta(modelos_comite, train_data, test_data, y_train, skf):
    """
    modelos_comite: lista de tuplas (id_name, algo, modalidad, seed)
    Retorna X_train_meta, X_test_meta, col_names
    """
    X_train_meta, X_test_meta, col_names = [], [], []

    for id_name, algo, modo, seed in modelos_comite:
        pipeline = joblib.load(os.path.join(modo, f'modelo_{algo}_s{seed}.joblib'))
        
        # 1. Cargar arreglos forzando float32
        X_tr_raw = train_data[modo].drop(columns=['Target_class','Sujeto'], errors='ignore').values.astype(np.float32)
        X_ts_raw = test_data[modo].drop(columns=['Target_class','Sujeto'], errors='ignore').values.astype(np.float32)
        
        # 2. Transformar con scaler + selector y reconvertir a float32
        X_tr_t = pipeline[:-1].transform(X_tr_raw).astype(np.float32)
        X_ts_t = pipeline[:-1].transform(X_ts_raw).astype(np.float32)
        
        clf = pipeline[-1]  # solo el clasificador
        oof = cross_val_predict(clone(clf), X_tr_t, y_train, cv=skf, method='predict_proba')
        
        X_train_meta.append(oof)
        X_test_meta.append(clf.predict_proba(X_ts_t))

        for c in np.unique(y_train):
            col_names.append(f'{id_name}_c{c}')

    return np.hstack(X_train_meta), np.hstack(X_test_meta), col_names


print('✅ Helpers cargados.')


# ---
# ## Estrategia 1 — `por_algoritmo`
# 
# El mismo algoritmo actúa sobre las 3 modalidades. Se itera sobre **todos los algoritmos disponibles** en los archivos `*_modelos_seleccionados.csv` (no solo RF).

# In[2]:


def ejecutar_stacking_por_algoritmo():
    """
    Para cada algoritmo disponible en los datos, construye un meta-modelo
    usando ese algoritmo en las 3 modalidades.
    
    Guarda resultados en: Metricas_Stacking_por_algoritmo.csv
    Guarda pesos en:      pesos_meta_modelo_por_algoritmo.csv
    """
    train_data, test_data, winners, y_train, y_test = cargar_datos()

    # ── Descubrir todos los algoritmos disponibles en las 3 modalidades ──
    algoritmos_por_modalidad = {m: set(winners[m]['Modelo'].unique()) for m in MODALIDADES}
    # Solo los que aparecen en las 3 modalidades simultáneamente
    algoritmos_comunes = set.intersection(*algoritmos_por_modalidad.values())
    algoritmos_a_usar = sorted(algoritmos_comunes)
    print(f'Algoritmos a evaluar: {algoritmos_a_usar}')

    resultados, pesos_rows = [], []

    for algo in algoritmos_a_usar:
        print(f'  → {algo}', end=' ')
        try:
            # Para cada modalidad, tomar la semilla del modelo con mejor CV_Acc
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            modelos_comite = []
            for modo in MODALIDADES:
                fila_algo = (winners[modo][winners[modo]['Modelo'] == algo]
                             .sort_values('CV_Acc', ascending=False)
                             .iloc[0])
                seed = int(fila_algo['Seed'])
                
                id_name = f'{modo}_{algo}_s{seed}'
                modelos_comite.append((id_name, algo, modo, seed))

            X_tr_m, X_ts_m, col_names = construir_matrices_meta(
                modelos_comite, train_data, test_data, y_train, skf)

            # Meta-modelo
            meta = LogisticRegression(max_iter=1000, random_state=42)
            meta.fit(X_tr_m, y_train)
            y_pred = meta.predict(X_ts_m)

            resultados.append(metricas_a_fila(algo, y_test, y_pred))

            # Guardar pesos del meta-modelo
            for i, coefs in enumerate(meta.coef_):
                for j, val in enumerate(coefs):
                    pesos_rows.append({
                        'Algoritmo':     algo,
                        'Clase':         i,
                        'Modalidad_Prob': j,
                        'Col_nombre':    col_names[j] if j < len(col_names) else f'feat_{j}',
                        'Peso':          val
                    })
            print('✔')

        except Exception as e:
            print(f'✗  Error: {e}')

    # ── Guardar CSVs ─────────────────────────────────────────────────────────
    df_res = pd.DataFrame(resultados).sort_values('Macro_F1', ascending=False)
    df_res.to_csv('Metricas_Stacking_por_algoritmo.csv', index=False)
    print(f'\n✅ Guardado: Metricas_Stacking_por_algoritmo.csv')

    pd.DataFrame(pesos_rows).to_csv('pesos_meta_modelo_por_algoritmo.csv', index=False)
    print(f'✅ Guardado: pesos_meta_modelo_por_algoritmo.csv')

    print('\nResultados:')
    print(df_res[['Algoritmo', 'Acc_Global', 'Macro_F1', 'Macro_Prec', 'Macro_Rec']].to_string(index=False))
    return df_res


df_stacking_algoritmo = ejecutar_stacking_por_algoritmo()


# ---
# ## Estrategia 2 — `mejores_por_modalidad`
# 
# El mejor modelo por CV_Acc de cada modalidad entra al meta-modelo.

# In[3]:


def ejecutar_stacking_mejores_por_modalidad():
    """
    Toma el mejor modelo (CV_Acc más alto) de cada modalidad
    y construye un único meta-modelo con ellos.
    
    Guarda resultados en: Metricas_Stacking_mejores_por_modalidad.csv
    Guarda pesos en:      pesos_meta_modelo_mejores_por_modalidad.csv
    """
    train_data, test_data, winners, y_train, y_test = cargar_datos()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    modelos_comite = []
    print('Selección del comité (mejor por modalidad):')
    for modo in MODALIDADES:
        mejor = winners[modo].sort_values('CV_Acc', ascending=False).iloc[0]
        algo  = mejor['Modelo']
        seed  = int(mejor['Seed'])
        print(f'  {modo}: {algo} (s{seed})  CV_Acc={mejor["CV_Acc"]:.4f}')
        modelos_comite.append((f'Best_{modo}_{algo}_s{seed}', algo, modo, seed))

    X_tr_m, X_ts_m, col_names = construir_matrices_meta(
        modelos_comite, train_data, test_data, y_train, skf)

    meta = LogisticRegression(max_iter=1000, random_state=42)
    meta.fit(X_tr_m, y_train)
    y_pred = meta.predict(X_ts_m)

    # Métricas
    df_res = pd.DataFrame([metricas_a_fila('Mejores_por_modalidad', y_test, y_pred)])
    df_res.to_csv('Metricas_Stacking_mejores_por_modalidad.csv', index=False)
    print(f'\n✅ Guardado: Metricas_Stacking_mejores_por_modalidad.csv')
    print(classification_report(y_test, y_pred, output_dict=True, zero_division=0,
                                target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]))

    # Pesos del meta-modelo
    pesos_rows = []
    for i, coefs in enumerate(meta.coef_):
        for j, val in enumerate(coefs):
            pesos_rows.append({
                'Clase':         i,
                'Modalidad_Prob': j,
                'Col_nombre':    col_names[j] if j < len(col_names) else f'feat_{j}',
                'Peso':          val
            })
    pd.DataFrame(pesos_rows).to_csv('pesos_meta_modelo_mejores_por_modalidad.csv', index=False)
    print('✅ Guardado: pesos_meta_modelo_mejores_por_modalidad.csv')
    return df_res


df_stacking_mejores = ejecutar_stacking_mejores_por_modalidad()


# ---
# ## Estrategia 3 — Comité Top-N
# 
# Los N mejores modelos (por CV_Acc) de **cada modalidad** forman el consejo. Se prueba con Top-2 y Top-3.

# In[4]:


def ejecutar_stacking_comite(n_mejores=2):
    """
    n_mejores: cantidad de modelos a tomar de cada modalidad (2 o 3).
    
    Guarda resultados en: Metricas_Stacking_Top{n}.csv
    Guarda pesos en:      Pesos_Comite_Top{n}.csv
    """
    print(f'\n=== Meta-modelo con los {n_mejores} mejores por modalidad ===')
    train_data, test_data, winners, y_train, y_test = cargar_datos()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    modelos_comite = []
    for modo in MODALIDADES:

        top_n = (winners[modo].sort_values('CV_Acc', ascending=False).head(n_mejores))
        for _, fila in top_n.iterrows():
            algo = fila['Modelo']
            seed = int(fila['Seed'])
            modelos_comite.append((f'{modo}_{algo}_s{seed}', algo, modo, seed))
            print(f'  [{modo}] {algo} s{seed}  CV_Acc={fila["CV_Acc"]:.4f}')

    print(f'Total modelos en el consejo: {len(modelos_comite)}')

    X_tr_m, X_ts_m, col_names = construir_matrices_meta(
        modelos_comite, train_data, test_data, y_train, skf)

    meta = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    meta.fit(X_tr_m, y_train)
    y_pred = meta.predict(X_ts_m)

    # Métricas — formato homologado
    df_res = pd.DataFrame([metricas_a_fila(f'Top{n_mejores}', y_test, y_pred)])
    df_res.to_csv(f'Metricas_Stacking_Top{n_mejores}.csv', index=False)
    print(f'✅ Guardado: Metricas_Stacking_Top{n_mejores}.csv')
    print(classification_report(y_test, y_pred, output_dict=True, zero_division=0,
                                target_names=[CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]))

    # Pesos
    coef_df = pd.DataFrame(meta.coef_, columns=col_names)
    coef_df.to_csv(f'Pesos_Comite_Top{n_mejores}.csv', index=False)
    print(f'✅ Guardado: Pesos_Comite_Top{n_mejores}.csv')

    return accuracy_score(y_test, y_pred)


acc_2 = ejecutar_stacking_comite(n_mejores=2)
acc_3 = ejecutar_stacking_comite(n_mejores=3)

print(f'\nResumen Accuracy:')
print(f'  Top-2: {acc_2:.4f}')
print(f'  Top-3: {acc_3:.4f}')


# ---
# ## Resumen consolidado de todas las estrategias

# In[5]:


# Consolidar todos los CSVs de stacking en una sola tabla comparativa
archivos = {
    'por_algoritmo':        'Metricas_Stacking_por_algoritmo.csv',
    'mejores_por_modalidad':'Metricas_Stacking_mejores_por_modalidad.csv',
    'Top-2':                'Metricas_Stacking_Top2.csv',
    'Top-3':                'Metricas_Stacking_Top3.csv',
}

partes = []
for etiqueta, path in archivos.items():
    if os.path.exists(path):
        df_tmp = pd.read_csv(path)
        df_tmp['Estrategia'] = etiqueta
        partes.append(df_tmp)

if partes:
    df_resumen = pd.concat(partes, ignore_index=True)
    cols_orden = ['Estrategia', 'Algoritmo', 'Acc_Global', 'Macro_F1',
                  'Macro_Prec', 'Macro_Rec',
                  'F1_Clase_CN', 'F1_Clase_DCL', 'F1_Clase_EA']
    cols_ok = [c for c in cols_orden if c in df_resumen.columns]
    print(df_resumen[cols_ok].sort_values('Macro_F1', ascending=False).to_string(index=False))
    df_resumen.to_csv('Metricas_Stacking_RESUMEN.csv', index=False)
    print('\n✅ Guardado: Metricas_Stacking_RESUMEN.csv')

