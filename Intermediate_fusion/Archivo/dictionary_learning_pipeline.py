"""
dictionary_learning_pipeline.py
================================
Pipeline completo para clasificación multimodal de Alzheimer usando Dictionary Learning.

Flujo:
  PASO 1 — Aprender representaciones por modalidad (fit SOLO en train)
  PASO 2 — Fusión intermedia (hstack de representaciones latentes)
  PASO 3 — Pipeline de clasificación antifuga con GridSearchCV y 5 semillas

Uso:
    python dictionary_learning_pipeline.py

Salidas (en results/):
    log_total.csv              — log completo de todos los experimentos
    resumen_resultados.txt     — métricas finales resumidas
    confusion_matrix_*.png     — matrices de confusión
    roc_curve_*.png            — curvas ROC multiclase
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

RESULTS_DIR = 'dict_l_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [7, 14, 21, 28, 35]
N_COMPONENTS = 10  # dimensión del espacio latente por modalidad


# ── PASO 1: Carga de datos ─────────────────────────────────────────────────────

def load_modality(train_path, test_path):
    """Carga una modalidad, separa features y etiquetas."""
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    y_train = train_df['Target_class'].values
    y_test  = test_df['Target_class'].values

    X_train = train_df.drop(columns=['Sujeto', 'Target_class']).values.astype(np.float64)
    X_test  = test_df.drop(columns=['Sujeto', 'Target_class']).values.astype(np.float64)

    return X_train, X_test, y_train, y_test


def learn_representations(X_train, X_test, n_components=N_COMPONENTS, random_state=42):
    """
    Aprende representaciones latentes con DictionaryLearning.

    CORRECCIÓN DE LEAKAGE:
    - StandardScaler se fitea SOLO en X_train
    - DictionaryLearning se fitea SOLO en X_train
    - X_test se transforma con los objetos ya fiteados en train
    """
    # Escalar — fit SOLO en train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)       # transform, no fit_transform

    # Dictionary Learning — fit SOLO en train
    dl = DictionaryLearning(
        n_components=n_components,
        max_iter=500,
        random_state=random_state,
        transform_algorithm='lasso_lars',
        fit_algorithm='lars',
        n_jobs=1,
    )
    X_lat_train = dl.fit_transform(X_train_scaled)
    X_lat_test  = dl.transform(X_test_scaled)       # transform, no fit_transform

    return X_lat_train, X_lat_test


# ── PASO 2: Fusión intermedia ──────────────────────────────────────────────────

def fuse_representations(lats_train, lats_test):
    """Concatena horizontalmente las representaciones de las 3 modalidades."""
    X_train_fused = np.hstack(lats_train)
    X_test_fused  = np.hstack(lats_test)
    return X_train_fused, X_test_fused


# ── PASO 3: Pipeline de clasificación ─────────────────────────────────────────

CLASSIFIERS = {
    'SVM': {
        'model': SVC(probability=True),
        'params': {
            'classifier__C':      [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params': {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth':    [None, 10],
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'params': {
            'classifier__n_neighbors': [3, 5, 7],
            'classifier__weights':     ['uniform', 'distance'],
        }
    },
}

K_FEATURES = [5, 10, 15]  # valores de k para SelectKBest


def run_classification(X_train, X_test, y_train, y_test):
    """
    Ejecuta el pipeline antifuga con GridSearchCV y 5 semillas.
    Devuelve lista de dicts con resultados y un resumen por clasificador.
    """
    all_rows   = []   # para log_total.csv
    summary    = {}   # para resumen_resultados.txt

    for clf_name, clf_config in CLASSIFIERS.items():
        seed_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}

        for seed in SEEDS:
            np.random.seed(seed)

            skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)

            # Pipeline antifuga: Scaler + Selector + Clasificador
            pipeline = Pipeline([
                ('scaler',     StandardScaler()),
                ('selector',   SelectKBest(score_func=f_classif)),
                ('classifier', clf_config['model']),
            ])

            param_grid = {
                'selector__k': K_FEATURES,
                **clf_config['params'],
            }

            grid = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=skf,
                scoring='accuracy',
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X_train, y_train)

            # Evaluación en test set independiente
            y_pred  = grid.predict(X_test)
            y_proba = grid.predict_proba(X_test)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)

            # Validación: accuracy > 0.95 es señal de posible leakage
            if acc > 0.95:
                print(f"  ⚠️  {clf_name} seed={seed}: acc={acc:.4f} > 0.95 — verificar leakage")

            seed_metrics['acc'].append(acc)
            seed_metrics['prec'].append(prec)
            seed_metrics['rec'].append(rec)
            seed_metrics['f1'].append(f1)

            # Guardar fila en log
            all_rows.append({
                'Clasificador':   clf_name,
                'Semilla':        seed,
                'CV_Acc':         round(grid.best_score_, 4),
                'Test_Acc':       round(acc, 4),
                'Test_Prec_macro':round(prec, 4),
                'Test_Rec_macro': round(rec, 4),
                'Test_F1_macro':  round(f1, 4),
                'Best_k':         grid.best_params_.get('selector__k'),
                'Best_params':    str(grid.best_params_),
            })

            # Gráficos solo para la primera semilla (representativa)
            if seed == SEEDS[0]:
                _plot_confusion_matrix(y_test, y_pred, clf_name, seed)
                _plot_roc_curves(y_test, y_proba, clf_name, seed)

        # Resumen por clasificador (media ± std sobre 5 semillas)
        summary[clf_name] = {
            k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4))
            for k, v in seed_metrics.items()
        }
        print(f"  {clf_name}: acc={summary[clf_name]['acc'][0]:.4f}±{summary[clf_name]['acc'][1]:.4f}  "
              f"f1={summary[clf_name]['f1'][0]:.4f}±{summary[clf_name]['f1'][1]:.4f}")

    return all_rows, summary


# ── Gráficos ───────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_test, y_pred, clf_name, seed):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CN', 'DCL', 'EA'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix — {clf_name} (seed {seed})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'confusion_matrix_{clf_name}_seed{seed}.png'), dpi=100)
    plt.close()


def _plot_roc_curves(y_test, y_proba, clf_name, seed):
    classes    = [0, 1, 2]
    class_names = ['CN', 'DCL', 'EA']
    colors      = ['steelblue', 'darkorange', 'green']
    y_test_bin  = label_binarize(y_test, classes=classes)

    plt.figure(figsize=(7, 5))
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (OvR) — {clf_name} (seed {seed})')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'roc_curve_{clf_name}_seed{seed}.png'), dpi=100)
    plt.close()


# ── Guardar resultados ─────────────────────────────────────────────────────────

def save_log(all_rows):
    df = pd.DataFrame(all_rows)
    path = os.path.join(RESULTS_DIR, 'log_total.csv')
    df.to_csv(path, index=False)
    print(f"\nLog guardado: {path}  ({len(df)} filas)")


def save_summary(summary, X_train_shape, X_test_shape, y_train, y_test):
    path = os.path.join(RESULTS_DIR, 'resumen_resultados.txt')
    with open(path, 'w') as f:
        f.write("# Dictionary Learning — Resumen de Resultados\n\n")
        f.write(f"## Configuración\n")
        f.write(f"- n_components por modalidad: {N_COMPONENTS}\n")
        f.write(f"- Dimensión fusionada: {X_train_shape[1]} features\n")
        f.write(f"- Train: {X_train_shape[0]} sujetos  |  Test: {X_test_shape[0]} sujetos\n")
        f.write(f"- Clases train: {dict(zip(*np.unique(y_train, return_counts=True)))}\n")
        f.write(f"- Clases test:  {dict(zip(*np.unique(y_test,  return_counts=True)))}\n")
        f.write(f"- Semillas: {SEEDS}\n")
        f.write(f"- k features evaluados: {K_FEATURES}\n\n")

        f.write("## Métricas en Test (media ± std sobre 5 semillas)\n\n")
        for clf_name, m in summary.items():
            f.write(f"### {clf_name}\n")
            f.write(f"- Accuracy:  {m['acc'][0]:.4f} ± {m['acc'][1]:.4f}\n")
            f.write(f"- Precision: {m['prec'][0]:.4f} ± {m['prec'][1]:.4f}\n")
            f.write(f"- Recall:    {m['rec'][0]:.4f} ± {m['rec'][1]:.4f}\n")
            f.write(f"- F1 macro:  {m['f1'][0]:.4f} ± {m['f1'][1]:.4f}\n\n")

    print(f"Resumen guardado: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Dictionary Learning — Pipeline Multimodal Alzheimer")
    print("=" * 55)

    # ── PASO 1: Representaciones ──
    print("\n[PASO 1] Aprendiendo representaciones latentes...")

    X_dti_train,  X_dti_test,  y_train, y_test = load_modality(
        '../data/train_DTI.csv',  '../data/test_DTI.csv')
    X_meta_train,  X_meta_test,  _, _ = load_modality(
        '../data/train_Meta.csv',  '../data/test_Meta.csv')
    X_morfo_train, X_morfo_test, _, _ = load_modality(
        '../data/train_Morfo.csv', '../data/test_Morfo.csv')

    print(f"  DTI:   train={X_dti_train.shape}  test={X_dti_test.shape}")
    print(f"  Meta:  train={X_meta_train.shape}  test={X_meta_test.shape}")
    print(f"  Morfo: train={X_morfo_train.shape}  test={X_morfo_test.shape}")
    print(f"  Clases train: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Clases test:  {dict(zip(*np.unique(y_test,  return_counts=True)))}")

    dti_lat_train,  dti_lat_test  = learn_representations(X_dti_train,  X_dti_test)
    meta_lat_train, meta_lat_test = learn_representations(X_meta_train, X_meta_test)
    morfo_lat_train,morfo_lat_test= learn_representations(X_morfo_train,X_morfo_test)

    print(f"  Representaciones latentes: {dti_lat_train.shape[1]} dims × 3 modalidades")

    # ── PASO 2: Fusión ──
    print("\n[PASO 2] Fusión intermedia...")
    X_train_fused, X_test_fused = fuse_representations(
        [dti_lat_train,  meta_lat_train,  morfo_lat_train],
        [dti_lat_test,   meta_lat_test,   morfo_lat_test],
    )
    print(f"  X_train_fused: {X_train_fused.shape}")
    print(f"  X_test_fused:  {X_test_fused.shape}")

    assert X_train_fused.shape[0] == 42, f"Error: esperados 42 sujetos train, got {X_train_fused.shape[0]}"
    assert X_test_fused.shape[0]  == 18, f"Error: esperados 18 sujetos test, got {X_test_fused.shape[0]}"

    # ── PASO 3: Clasificación ──
    print("\n[PASO 3] Clasificación (3 modelos × 5 semillas × GridSearchCV)...")
    all_rows, summary = run_classification(
        X_train_fused, X_test_fused, y_train, y_test)

    # ── Guardar ──
    save_log(all_rows)
    save_summary(summary, X_train_fused.shape, X_test_fused.shape, y_train, y_test)

    print("\n" + "=" * 55)
    print("  Proceso completado. Resultados en results/")
    print("=" * 55)


if __name__ == '__main__':
    main()
