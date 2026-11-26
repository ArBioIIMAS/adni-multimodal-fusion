import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_curve, auc, ConfusionMatrixDisplay, classification_report)

# Modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import SGDClassifier
from catboost import CatBoostClassifier
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

from functools import partial
import joblib
import json
import random

# --- 1. FUNCIÓN DE SEMILLA GLOBAL (REPRODUCIBILIDAD) ---
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 1. ARQUITECTURA MLP DINÁMICA ---
class DynamicMLP(nn.Module):
    def __init__(self, hidden_sizes=[128, 64], num_classes=3, dropout_rate=0.3):
        super(DynamicMLP, self).__init__()
        self.input_layer = nn.LazyLinear(hidden_sizes[0])
        layers = []
        prev = hidden_sizes[0]
        for h in hidden_sizes[1:]:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout_rate))
            prev = h
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev, num_classes)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        return self.output_layer(self.hidden_layers(x))

def get_classifiers_config(seed):
    mutual_info_wrs = partial(mutual_info_classif, random_state=seed)
    rf_selector     = RandomForestClassifier(n_estimators=100, random_state=seed)

    classifiers_config = {
        'RF': {
            'model': RandomForestClassifier(random_state=seed),
            'params': {'model__criterion': ['gini', 'entropy', 'log_loss'], # default gini
                        'model__n_estimators': [40, 60], # default 100
                        'model__max_depth': [3, None], # default None
                        'model__min_samples_split': [2, 3] # default 2
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=seed),
            'params': {
                'model__C': [0.1, 1, 10],# default 1
                'model__gamma': [0.001], # default scale
                'model__kernel': ['linear', 'rbf'] # default rbf
            }
        },
        'CatBoost': {
            'model': CatBoostClassifier(silent=True, random_state=seed),
            'params': {
                'model__iterations': [1000], # default 1000
                'model__depth': [5,7], # default 6
                #'model__learning_rate': [0.01],# The default value is defined automatically
                'model__l2_leaf_reg': [1, 2] # default 3
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'model__n_neighbors': [3, 5, 7], # default 5
                'model__weights': ['distance'], # default uniform
                'model__leaf_size': [5, 15, 30], # default 30
                'model__metric': ['euclidean'] # default minkowski 
            }
        },
        'ElasticNet': {
            'model': SGDClassifier(penalty='elasticnet', max_iter=1500, random_state=seed),
            'params': {'model__loss':['log_loss'],# default hinge, but ‘hinge’ gives a linear SVM
                        'model__alpha': [0.0001], # default 0.0001
                        'model__l1_ratio': [0.15, 0.5, 0.65]# default 0.15 -> 0 
                      }
        },
        'MLP': {
            'model': NeuralNetClassifier(
                DynamicMLP,
                criterion=nn.CrossEntropyLoss,
                optimizer=optim.Adam,
                max_epochs=500,
                train_split=None,
                verbose=0,
                callbacks=[('early_stopping',
                            EarlyStopping(patience=25, monitor='train_loss'))],
            ),
            'params': {
                'model__lr': [0.0005],
                #'model__batch_size': [None], # default de 128?
                'model__module__hidden_sizes': [[128, 64, 32]],
                'model__module__dropout_rate': [0.4, 0.5] # 
            }
        },
    }
    return classifiers_config, mutual_info_wrs, rf_selector


n_BM = {'DTI':144, 'Metabólicos':48, 'Morfológicos':36}
def main(modo):
    seeds = [7, 14, 21, 28, 35] # Usamos semillas para evaluar robustez
    

    # Carga de datos y REGLA 1 (Descartar Sujeto)
    train_df = pd.read_csv(f'../data/train_{modo}.csv')
    test_df = pd.read_csv(f'../data/test_{modo}.csv')
    
    X_train = train_df.drop(columns=['Sujeto', 'Target_class']).values.astype(np.float32)
    y_train = train_df['Target_class'].values.astype(np.int64)
    X_test = test_df.drop(columns=['Sujeto', 'Target_class']).values.astype(np.float32)
    y_test = test_df['Target_class'].values.astype(np.int64)
    feature_names = train_df.drop(columns=['Sujeto', 'Target_class']).columns

    #k_values = [12, 24, 36, 48, 60]
    #k_values = [int( 0.2 * n_BM[modo] * i ) for i in range(1,6)]
    # En lugar de k_values = [12, 24, 36, 48, 60]
    n_features = X_train.shape[1]
    k_base = [12, 24, 36, 48, 60]
    k_values = [k for k in k_base if k <= n_BM[modo]]

    summary_final = []
    CLASS_NAMES = ['CN', 'DCL', 'EA']

        
    for s in seeds:
        set_seed(s) # Congelamos el azar específico de esta iteración
        # Diccionario para acumular resultados de diferentes semillas
        seed_results = []
        # REGLA 2: StratifiedKFold k=6
        skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=s)
        # Guardar los índices de los Folds de la Validación Cruzada
        cv_indices = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            cv_indices.append({'train': train_idx.tolist(), 'val': val_idx.tolist()})
        # Guardar en un JSON o CSV
        with open(f'{modo}/cv_indices_s{s}.json', 'w') as f:
            json.dump(cv_indices, f)
        # 3. GridSearch de clasificadores (igual que Joint Search)
        classifiers_config, mutual_info_wrs, rf_selector = get_classifiers_config(s)
        
        for name, config in classifiers_config.items():
            print(f"\n>>> Iniciando Búsqueda Conjunta para: {name}")

            # Pipeline Antifuga
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif)), # Placeholder
                ('model', config['model'])
            ])

            # BÚSQUEDA DE ORDEN SUPERIOR: Selector + k + Hiperparámetros
            param_grid = []
            
            # Bloque SelectKBest (ANOVA y Mutual Info)
            for func in [f_classif, mutual_info_wrs]:
                param_grid.append({
                    'selector': [SelectKBest(func)],
                    'selector__k': k_values,
                    **config['params']
                })

            # Bloque SelectFromModel (RF, CatBoost)
            selectors = [('RF_Sel', rf_selector)]#, ('CB_Sel', cb_selector)]
            for s_name, s_obj in selectors:
                param_grid.append({
                    'selector': [SelectFromModel(s_obj, threshold=-np.inf)],
                    'selector__max_features': k_values,
                    **config['params']
                })

            grid = GridSearchCV(pipeline, param_grid, cv=skf, scoring='accuracy', n_jobs=1)
            grid.fit(X_train, y_train)

            # --- REGISTRO Y TRAZABILIDAD ---
            cv_res = pd.DataFrame(grid.cv_results_)
            # Guardar log completo de trazabilidad
            cv_res.to_csv(f'{modo}/grid_full_log_{name}_s{s}.csv', index=False)
            
            # Identificar selector ganador
            best_sel = grid.best_params_['selector']
            if isinstance(best_sel, SelectKBest):
                # Verificamos si es un partial para extraer el nombre de la función base
                func = best_sel.score_func
                func_name = func.func.__name__ if hasattr(func, 'func') else func.__name__
                
                origin = f"KBest_{func_name}"
                k_win = grid.best_params_['selector__k']
            else:
                origin = f"FromModel_{type(best_sel.estimator).__name__}"
                k_win = grid.best_params_['selector__max_features']


            # 1. Obtener el score de importancia del selector
            selector = grid.best_estimator_.named_steps['selector']
            # 2. Extraer puntuaciones según el tipo de selector
            if hasattr(selector, 'scores_'):
                # Caso: SelectKBest (ANOVA / Mutual Info)
                importancias = selector.scores_
            elif hasattr(selector, 'estimator_'):
                # Caso: SelectFromModel (Random Forest / CatBoost)
                # Accedemos al modelo interno que hizo la selección
                importancias = selector.estimator_.feature_importances_
            else:
                # Caso: Otros (como RFE o si no hay puntuaciones disponibles)
                importancias = np.ones(len(feature_names)) # Dummy para evitar error
            df_importancia = pd.DataFrame({'ROI': feature_names, 'Importancia': importancias})
            importancia_ordenada = df_importancia.sort_values(by='Importancia', ascending=False)
            importancia_ordenada.to_csv(f'{modo}/importancia_ROIs_{name}_s{s}.csv')

            # Evaluación Final
            best_pipe = grid.best_estimator_
            y_pred = best_pipe.predict(X_test)
            y_score = best_pipe.predict_proba(X_test) if hasattr(best_pipe, "predict_proba") else None
            # --- EVALUACIÓN FINAL CORREGIDA ---
            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro') # Faltaba esta
            rec  = recall_score(y_test, y_pred, average='macro')
            f1   = f1_score(y_test, y_pred, average='macro')

            mejores_hiperparametros = str(grid.best_params_)
            # Guardar en el diccionario del CSV
            summary_final.append({
                'Seed': s, 
                'Modelo': name, 
                'Selector': origin, 
                'k': k_win, 
                'CV_Acc': grid.best_score_, 
                'Test_Acc': acc, 
                'Precision': prec, # Añadida al CSV
                'Recall': rec, 
                'F1': f1,
                'Params': mejores_hiperparametros  # <--- NUEVA COLUMNA
            })

            # Reporte Detallado
            with open(f'{modo}/reporte_final_{name}_s{s}.txt', 'w') as f:
                f.write(f"MODELO: {name} | SEMILLA: {s}\n")
                f.write(f"GANADOR: {origin} con k={k_win}\n")
                f.write(f"Mejores Parámetros: {grid.best_params_}\n")
                # Esta línea ahora reporta las 4 métricas clave:
                f.write(f"Métricas Test: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}\n")
                f.write("Métricas por clase:\n")
                f.write(classification_report(y_test, y_pred,
                                              target_names=CLASS_NAMES,
                                              zero_division=0))
                f.write(f"ROIs: {importancia_ordenada['ROI'].tolist()}\n")
            # Gráfico: Matriz de Confusión
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['CN', 'DCL', 'EA'], cmap='Blues')
            plt.title(f'Matriz {name} (s {s})')
            plt.savefig(f'{modo}/cm_final_{name}_s{s}.png')
            plt.close()
            # Curva ROC Multiclase
            y_score_valid = (y_score is not None and
                 not np.any(np.isnan(y_score)) and
                 not np.any(np.isinf(y_score)))
            if y_score_valid:
                y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
                plt.figure(figsize=(7, 5))
                for i, (cname, color) in enumerate(
                        zip(CLASS_NAMES, ['steelblue','darkorange','green'])):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    plt.plot(fpr, tpr, color=color, label=f'{cname} (AUC = {auc(fpr, tpr):.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.title(f'ROC {name} - Selector {origin} (s {s})')
                plt.legend()
                plt.savefig(f'{modo}/roc_{name}_s{s}.png')
                plt.close()

            if name == 'MLP':
                # 1. Extraer el mejor modelo de la búsqueda
                mejor_mlp = grid.best_estimator_.named_steps['model']
                historia = mejor_mlp.history_
                
                # 2. Graficar Train vs Validation Loss
                # Nota: Si no usas split de validación en skorch, solo graficamos train_loss
                train_loss = historia[:, 'train_loss']
                
                plt.figure(figsize=(10, 5))
                plt.plot(train_loss, label='Training loss')
                plt.title(f'Loss vs. epochs (k={k_win}, s={s})')
                plt.xlabel('Epochs')
                plt.ylabel('Loss (CrossEntropy)')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'{modo}/curva_loss_final_{name}_s{s}.png')
                plt.close()

                # Guardar los pesos (parámetros de las capas)
                #torch.save(mejor_mlp.module_.state_dict(), f'{modo}/pesos_mejor_mlp_{name}_s{s}.pt')
                #print(f"Pesos del mejor MLP guardados en {modo}/pesos_mejor_mlp_{name}_s{s}.pt")
            #else:
                # Guardar el pipeline completo (incluye Scaler, Selector y el Modelo entrenado)
            joblib.dump(best_pipe, f'{modo}/modelo_{name}_s{s}.joblib')
            print("Pipeline guardado con éxito. Incluye el escalado y la selección de ROIs.")

    # Guardar resumen maestro
    pd.DataFrame(summary_final).to_csv(f'{modo}_resumen_maestro_joint.csv', index=False)
    print("\nProceso finalizado. Todos los logs y resultados guardados.")

if __name__ == '__main__':
    for modo in ['DTI', 'Metabólicos', 'Morfológicos']:
        if not os.path.exists(modo):
            os.makedirs(modo)
        main(modo)
