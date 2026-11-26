"""
fusion_gridsearch.py
=====================
Comparación de métodos de fusión multimodal para clasificación de Alzheimer.
Métodos: Dictionary Learning (DL), Autoencoder (AE), Deep Dictionary Learning (DDL).

Diseño experimental:
  - GridSearch conjunto: parámetros del modelo de representación + k_selector
  - Criterio de selección: accuracy CV (StratifiedKFold, k=6) con clasificador fijo
  - Clasificadores fijos (mejores hiperparámetros del experimento anterior):
      SVM(C=1, kernel='rbf'), RF(n_estimators=50), KNN(n_neighbors=5)
  - Espacios latentes comparables: latent_dim ∈ {8, 16, 32} para los 3 métodos
  - 5 semillas para estimación de varianza en evaluación final
  - Error de reconstrucción reportado como caracterización (no como criterio)

Uso:
    python fusion_gridsearch.py [--method dl|ae|ddl|all]

Salidas (en results_gridsearch/):
    gridsearch_log_{METHOD}.csv     — todas las configuraciones evaluadas
    best_config_{METHOD}.json       — mejor configuración por método y clasificador
    eval_log_{METHOD}.csv           — evaluación final (5 semillas)
    resumen_comparacion.txt         — tabla comparativa final
    confusion_matrix_*.png
    roc_curve_*.png
"""

import os
import sys
import json
import argparse
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import DictionaryLearning
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
)

warnings.filterwarnings('ignore')

RESULTS_DIR = 'results_gridsearch'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Configuración global ───────────────────────────────────────────────────────

SEEDS      = [7, 14, 21, 28, 35]
BATCH_SIZE = None
EPOCHS     = 150

# Clasificadores fijos — mejores hiperparámetros del experimento anterior
FIXED_CLASSIFIERS = {
    'SVM': SVC(C=1, kernel='linear', probability=True, random_state=42),
    'RF':  RandomForestClassifier(n_estimators=50, max_depth=None, random_state=42, min_samples_split=3),
    'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean', leaf_size=5),
}

# Espacio de búsqueda por método
SEARCH_SPACES = {
    'dl': {
        'latent_dim':        [8, 16, 32],
        'transform_algorithm': ['lasso_lars', 'omp'],
        'k_selector':        [10, 20, 'all'],
    },
    'ae': {
        'latent_dim': [8, 16, 32],
        'k_selector': [10, 20, 'all'],
    },
    'ddl': {
        'latent_dim':  [8, 16, 32],
        'l1_lambda':   [1e-3, 1e-4, 1e-5],
        'k_selector':  [10, 20, 'all'],
    },
}

MODALITIES = ['DTI', 'Meta', 'Morfo']


# ── Carga de datos ─────────────────────────────────────────────────────────────

def load_all_modalities():
    """Carga las 3 modalidades. Devuelve dicts con arrays numpy."""
    train, test = {}, {}
    for mod in MODALITIES:
        tr = pd.read_csv(f'../data/train_{mod}.csv')
        te = pd.read_csv(f'../data/test_{mod}.csv')
        train[mod] = tr.drop(columns=['Sujeto', 'Target_class']).values.astype(np.float32)
        test[mod]  = te.drop(columns=['Sujeto', 'Target_class']).values.astype(np.float32)

    y_train = pd.read_csv('../data/train_DTI.csv')['Target_class'].values
    y_test  = pd.read_csv('../data/test_DTI.csv')['Target_class'].values
    return train, test, y_train, y_test


# ── Modelos de representación ──────────────────────────────────────────────────

def fit_transform_dl(X_train, X_test, latent_dim, transform_algorithm, seed=42):
    """
    Dictionary Learning clásico (sklearn).
    SelectKBest se aplica ANTES del DL para reducir redundancia en la entrada.
    Devuelve (X_lat_train, X_lat_test, recon_error).
    """
    scaler        = StandardScaler()
    X_train_sc    = scaler.fit_transform(X_train)
    X_test_sc     = scaler.transform(X_test)

    dl = DictionaryLearning(
        n_components=latent_dim,
        transform_algorithm=transform_algorithm,
        fit_algorithm='lars',
        max_iter=500,
        random_state=seed,
        n_jobs=1,
    )
    X_lat_train = dl.fit_transform(X_train_sc)
    X_lat_test  = dl.transform(X_test_sc)

    # Error de reconstrucción: ||X - α·D||² / n
    D            = dl.components_                          # (n_components, n_features)
    X_recon      = X_lat_train @ D
    recon_error  = float(np.mean((X_train_sc - X_recon) ** 2))

    return X_lat_train, X_lat_test, recon_error


# ── Autoencoder ────────────────────────────────────────────────────────────────

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.ReLU(),
            nn.Linear(64, 128),         nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def _train_torch_model(model, X_scaled, epochs, seed, l1_lambda=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_tensor = torch.FloatTensor(X_scaled)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        z, xr = model(X_tensor)
        loss  = mse(xr, X_tensor)
        if l1_lambda > 0:
            loss = loss + l1_lambda * z.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def _get_latent_and_recon_error(model, X_scaled):
    model.eval()
    with torch.no_grad():
        xt        = torch.FloatTensor(X_scaled)
        z, xr     = model(xt)
    recon_error = float(nn.MSELoss()(xr, xt).item())
    return z.numpy(), recon_error


def fit_transform_ae(X_train, X_test, latent_dim, seed=42):
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = Autoencoder(X_train.shape[1], latent_dim)
    model = _train_torch_model(model, X_train_sc, EPOCHS, seed)

    X_lat_train, recon_error = _get_latent_and_recon_error(model, X_train_sc)
    X_lat_test,  _           = _get_latent_and_recon_error(model, X_test_sc)
    return X_lat_train, X_lat_test, recon_error


# ── Deep Dictionary Learning ───────────────────────────────────────────────────

class DDL(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, latent_dim), nn.ReLU(),           # sparsity via ReLU
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.ReLU(),
            nn.Linear(64, 128),         nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def fit_transform_ddl(X_train, X_test, latent_dim, l1_lambda, seed=42):
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = DDL(X_train.shape[1], latent_dim)
    model = _train_torch_model(model, X_train_sc, EPOCHS, seed,
                               l1_lambda=l1_lambda)

    X_lat_train, recon_error = _get_latent_and_recon_error(model, X_train_sc)
    X_lat_test,  _           = _get_latent_and_recon_error(model, X_test_sc)
    return X_lat_train, X_lat_test, recon_error


# ── Fusión ─────────────────────────────────────────────────────────────────────

def fuse(lats_train, lats_test):
    return np.hstack(lats_train), np.hstack(lats_test)


# ── Selector ───────────────────────────────────────────────────────────────────

def apply_selector(X_train_fused, X_test_fused, y_train, k):
    """
    Aplica SelectKBest sobre el espacio latente fusionado.
    k='all' devuelve el espacio completo sin modificar.
    Fit SOLO en train.
    """
    n_features = X_train_fused.shape[1]
    if k == 'all' or k >= n_features:
        return X_train_fused, X_test_fused, n_features

    sel           = SelectKBest(score_func=f_classif, k=k)
    X_train_sel   = sel.fit_transform(X_train_fused, y_train)
    X_test_sel    = sel.transform(X_test_fused)
    return X_train_sel, X_test_sel, k


# ── CV con clasificador fijo ───────────────────────────────────────────────────

def cv_accuracy(X_train, y_train, clf, seed):
    """Accuracy CV (StratifiedKFold k=6) con clasificador fijo."""
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)
    # Pipeline con scaler interno para el clasificador
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    clf),
    ])
    scores = cross_val_score(pipe, X_train, y_train,
                             cv=skf, scoring='accuracy', n_jobs=-1)
    return float(scores.mean()), float(scores.std())


# ── GridSearch de representación ───────────────────────────────────────────────

def gridsearch_representation(method, X_raw_train, X_raw_test,
                               y_train, y_test):
    """
    Itera sobre todas las combinaciones del espacio de búsqueda del método.
    Para cada combinación y cada clasificador fijo calcula accuracy CV.
    Devuelve DataFrame con todos los resultados.
    """
    space  = SEARCH_SPACES[method]
    keys   = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))

    rows = []
    total = len(combos)
    print(f"  {total} configuraciones × {len(FIXED_CLASSIFIERS)} clasificadores")

    for i, combo in enumerate(combos):
        cfg = dict(zip(keys, combo))
        print(f"  [{i+1}/{total}] {cfg} ...", end=' ', flush=True)

        # Aprender representaciones por modalidad
        lats_train, lats_test, recon_errors = [], [], []
        ok = True
        for mod in MODALITIES:
            try:
                if method == 'dl':
                    lt, lte, re = fit_transform_dl(
                        X_raw_train[mod], X_raw_test[mod],
                        cfg['latent_dim'], cfg['transform_algorithm'], seed=42)
                elif method == 'ae':
                    lt, lte, re = fit_transform_ae(
                        X_raw_train[mod], X_raw_test[mod],
                        cfg['latent_dim'], seed=42)
                elif method == 'ddl':
                    lt, lte, re = fit_transform_ddl(
                        X_raw_train[mod], X_raw_test[mod],
                        cfg['latent_dim'], cfg['l1_lambda'], seed=42)
                lats_train.append(lt)
                lats_test.append(lte)
                recon_errors.append(re)
            except Exception as e:
                print(f"ERROR ({mod}): {e}")
                ok = False
                break

        if not ok:
            continue

        X_fused_tr, X_fused_te = fuse(lats_train, lats_test)

        # Aplicar selector sobre espacio latente fusionado
        X_sel_tr, X_sel_te, k_used = apply_selector(
            X_fused_tr, X_fused_te, y_train, cfg['k_selector'])

        mean_recon = float(np.mean(recon_errors))

        # Evaluar cada clasificador fijo con CV
        clf_results = {}
        for clf_name, clf in FIXED_CLASSIFIERS.items():
            cv_mean, cv_std = cv_accuracy(X_sel_tr, y_train, clf,
                                          seed=SEEDS[0])
            clf_results[clf_name] = (cv_mean, cv_std)

        print(' '.join(f"{c}={v[0]:.3f}" for c, v in clf_results.items()))

        row = {**cfg,
               'k_used':      k_used,
               'recon_error': round(mean_recon, 6),
               'fused_dim':   X_fused_tr.shape[1]}
        for clf_name, (cv_mean, cv_std) in clf_results.items():
            row[f'cv_acc_{clf_name}']     = round(cv_mean, 4)
            row[f'cv_acc_std_{clf_name}'] = round(cv_std,  4)
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, f'gridsearch_log_{method}.csv')
    df.to_csv(path, index=False)
    print(f"\n  GridSearch log: {path}")
    return df


def select_best_configs(gs_df):
    """
    Para cada clasificador, selecciona la configuración con mayor cv_acc.
    Devuelve dict: clf_name -> row (dict).
    """
    best = {}
    for clf_name in FIXED_CLASSIFIERS:
        col = f'cv_acc_{clf_name}'
        idx = gs_df[col].idxmax()
        best[clf_name] = gs_df.loc[idx].to_dict()
    return best


# ── Evaluación final con 5 semillas ───────────────────────────────────────────

def final_evaluation(method, best_configs,
                     X_raw_train, X_raw_test, y_train, y_test):
    """
    Para cada (clasificador, mejor_config) entrena de nuevo el modelo de
    representación y evalúa en test con 5 semillas.
    """
    all_rows = []
    summary  = {}

    for clf_name, cfg in best_configs.items():
        clf = FIXED_CLASSIFIERS[clf_name]
        seed_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}

        for seed in SEEDS:
            # Re-entrenar representaciones con esta semilla
            lats_train, lats_test = [], []
            for mod in MODALITIES:
                if method == 'dl':
                    lt, lte, _ = fit_transform_dl(
                        X_raw_train[mod], X_raw_test[mod],
                        int(cfg['latent_dim']), cfg['transform_algorithm'], seed=seed)
                elif method == 'ae':
                    lt, lte, _ = fit_transform_ae(
                        X_raw_train[mod], X_raw_test[mod],
                        int(cfg['latent_dim']), seed=seed)
                elif method == 'ddl':
                    lt, lte, _ = fit_transform_ddl(
                        X_raw_train[mod], X_raw_test[mod],
                        int(cfg['latent_dim']), float(cfg['l1_lambda']),
                        seed=seed)
                lats_train.append(lt)
                lats_test.append(lte)

            X_fused_tr, X_fused_te = fuse(lats_train, lats_test)
            k = cfg['k_selector']
            X_sel_tr, X_sel_te, _ = apply_selector(
                X_fused_tr, X_fused_te, y_train, k)

            # Entrenar clasificador fijo en train completo, evaluar en test
            pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
            pipe.fit(X_sel_tr, y_train)
            y_pred  = pipe.predict(X_sel_te)
            y_proba = pipe.predict_proba(X_sel_te)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)

            if acc > 0.95:
                print(f"    ⚠️  {clf_name} seed={seed} acc={acc:.4f} > 0.95")

            seed_metrics['acc'].append(acc)
            seed_metrics['prec'].append(prec)
            seed_metrics['rec'].append(rec)
            seed_metrics['f1'].append(f1)

            all_rows.append({
                'Metodo':          method.upper(),
                'Clasificador':    clf_name,
                'Semilla':         seed,
                'Test_Acc':        round(acc,  4),
                'Test_Prec_macro': round(prec, 4),
                'Test_Rec_macro':  round(rec,  4),
                'Test_F1_macro':   round(f1,   4),
                'latent_dim':      int(cfg['latent_dim']),
                'k_selector':      k,
                'fused_dim':       X_fused_tr.shape[1],
                'recon_error':     round(float(cfg.get('recon_error', 0)), 6),
            })

            if seed == SEEDS[0]:
                _plot_confusion_matrix(y_test, y_pred,
                                       f"{method}_{clf_name}", seed)
                _plot_roc_curves(y_test, y_proba,
                                 f"{method}_{clf_name}", seed)

        summary[clf_name] = {
            k2: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4))
            for k2, v in seed_metrics.items()
        }
        print(f"  [{method.upper()}] {clf_name}: "
              f"acc={summary[clf_name]['acc'][0]:.4f}"
              f"±{summary[clf_name]['acc'][1]:.4f}  "
              f"f1={summary[clf_name]['f1'][0]:.4f}"
              f"±{summary[clf_name]['f1'][1]:.4f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, f'eval_log_{method}.csv'), index=False)
    return summary


# ── Gráficos ───────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_test, y_pred, tag, seed):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['CN', 'DCL', 'EA'])
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix — {tag} (seed {seed})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,
                f'confusion_matrix_{tag}_seed{seed}.png'), dpi=100)
    plt.close()


def _plot_roc_curves(y_test, y_proba, tag, seed):
    y_bin  = label_binarize(y_test, classes=[0, 1, 2])
    colors = ['steelblue', 'darkorange', 'green']
    names  = ['CN', 'DCL', 'EA']
    plt.figure(figsize=(7, 5))
    for i, (name, color) in enumerate(zip(names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC={auc(fpr,tpr):.2f})')
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC (OvR) — {tag} (seed {seed})')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,
                f'roc_curve_{tag}_seed{seed}.png'), dpi=100)
    plt.close()


# ── Resumen comparativo final ──────────────────────────────────────────────────

def save_comparison(all_summaries, all_best_configs):
    path = os.path.join(RESULTS_DIR, 'resumen_comparacion.txt')
    with open(path, 'a') as f:
        f.write("# Comparación de Métodos de Fusión Multimodal\n")
        f.write("# Criterio de selección: accuracy CV con clasificador fijo\n")
        f.write("# Evaluación final: media ± std sobre 5 semillas en test\n\n")

        for method, summary in all_summaries.items():
            f.write(f"## {method.upper()}\n\n")
            best = all_best_configs[method]
            for clf_name, m in summary.items():
                cfg = best[clf_name]
                f.write(f"### {clf_name}\n")
                f.write(f"- Mejor config: latent_dim={int(cfg['latent_dim'])}")
                if 'transform_algorithm' in cfg:
                    f.write(f", transform={cfg['transform_algorithm']}")
                if 'l1_lambda' in cfg:
                    f.write(f", l1={cfg['l1_lambda']}")
                f.write(f", k_selector={cfg['k_selector']}\n")
                f.write(f"- Recon error (train): {cfg.get('recon_error','N/A')}\n")
                f.write(f"- Accuracy:  {m['acc'][0]:.4f} ± {m['acc'][1]:.4f}\n")
                f.write(f"- Precision: {m['prec'][0]:.4f} ± {m['prec'][1]:.4f}\n")
                f.write(f"- Recall:    {m['rec'][0]:.4f} ± {m['rec'][1]:.4f}\n")
                f.write(f"- F1 macro:  {m['f1'][0]:.4f} ± {m['f1'][1]:.4f}\n\n")

    print(f"\nResumen comparativo: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='all',
                        choices=['dl', 'ae', 'ddl', 'all'])
    args = parser.parse_args()

    methods = ['dl', 'ae', 'ddl'] if args.method == 'all' else [args.method]

    print("Cargando datos...")
    X_raw_train, X_raw_test, y_train, y_test = load_all_modalities()
    print(f"  Train: {len(y_train)} sujetos | "
          f"Clases: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test:  {len(y_test)}  sujetos | "
          f"Clases: {dict(zip(*np.unique(y_test,  return_counts=True)))}")

    # Limpiar resumen si corremos todo de nuevo
    summary_path = os.path.join(RESULTS_DIR, 'resumen_comparacion.txt')
    if args.method == 'all' and os.path.exists(summary_path):
        os.remove(summary_path)

    all_summaries    = {}
    all_best_configs = {}

    for method in methods:
        print(f"\n{'='*55}")
        print(f"  MÉTODO: {method.upper()}")
        print(f"{'='*55}")

        # 1. GridSearch de representación
        print(f"\n[1/2] GridSearch ({method.upper()})...")
        gs_df = gridsearch_representation(
            method, X_raw_train, X_raw_test, y_train, y_test)

        # 2. Seleccionar mejor config por clasificador
        best_configs = select_best_configs(gs_df)
        best_path    = os.path.join(RESULTS_DIR, f'best_config_{method}.json')
        with open(best_path, 'w') as f:
            json.dump(best_configs, f, indent=2, default=str)
        print(f"\n  Mejores configuraciones:")
        for clf_name, cfg in best_configs.items():
            print(f"    {clf_name}: latent_dim={int(cfg['latent_dim'])}, "
                  f"k={cfg['k_selector']}, "
                  f"cv_acc={cfg[f'cv_acc_{clf_name}']:.4f}")

        # 3. Evaluación final con 5 semillas
        print(f"\n[2/2] Evaluación final ({method.upper()}, 5 semillas)...")
        summary = final_evaluation(
            method, best_configs, X_raw_train, X_raw_test, y_train, y_test)

        all_summaries[method]    = summary
        all_best_configs[method] = best_configs

    # 4. Resumen comparativo
    save_comparison(all_summaries, all_best_configs)

    print(f"\n{'='*55}")
    print(f"  Completado. Resultados en {RESULTS_DIR}/")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
