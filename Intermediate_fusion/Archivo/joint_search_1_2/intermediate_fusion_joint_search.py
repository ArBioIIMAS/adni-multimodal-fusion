"""
intermediate_fusion_joint_search.py
=====================================
Búsqueda conjunta para métodos de fusión intermedia multimodal (Alzheimer).
Métodos: Dictionary Learning (DL), Autoencoder (AE), Deep Dictionary Learning (DDL).

Diseño experimental equivalente al Joint Search de early fusion:
  - Mismo espacio de clasificadores: RF, SVM, CatBoost, KNN, ElasticNet, MLP
  - Mismo espacio de selectores: SelectKBest(f_classif), SelectKBest(mutual_info),
                                  SelectFromModel(RF)
  - GridSearch conjunto: parámetros del modelo de representación + selector + clasificador
  - Criterio: accuracy CV StratifiedKFold(k=6)
  - 5 semillas para estimación de varianza
  - Espacios latentes comparables: latent_dim ∈ {8, 16, 32}

Uso:
    python intermediate_fusion_joint_search.py --method dl
    python intermediate_fusion_joint_search.py --method ae
    python intermediate_fusion_joint_search.py --method ddl
    python intermediate_fusion_joint_search.py --method all

Salidas (en results_joint_{METHOD}/):
    resumen_maestro_{METHOD}.csv     — resumen por semilla y clasificador
    reporte_final_{CLF}_s{S}.txt     — detalle por clasificador y semilla
    cm_final_{CLF}_s{S}.png
    roc_{CLF}_s{S}.png
    gridsearch_repr_log_{METHOD}.csv — GridSearch de representación
    resumen_final_{METHOD}.txt       — media ± std sobre 5 semillas
"""

import os
import json
import argparse
import warnings
import itertools
import random
from functools import partial

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
from sklearn.feature_selection import (SelectKBest, f_classif,
                                        mutual_info_classif, SelectFromModel)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, auc, ConfusionMatrixDisplay)

from catboost import CatBoostClassifier
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

MODALITIES = ['DTI', 'Meta', 'Morfo']
SEEDS      = [7, 14, 21, 28, 35]
EPOCHS     = 150

# Espacios de búsqueda por método de representación
REPR_SPACES = {
    'dl': {
        'latent_dim':          [8, 16, 32],
        'transform_algorithm': ['lasso_lars', 'omp'],
    },
    'ae': {
        'latent_dim': [8, 16, 32],
    },
    'ddl': {
        'latent_dim': [8, 16, 32],
        'l1_lambda':  [1e-3, 1e-4, 1e-5],
    },
}

K_VALUES = [12, 24, 36, 48, 60]   # igual que Joint Search early fusion


# ── Reproducibilidad ───────────────────────────────────────────────────────────

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Arquitecturas PyTorch ──────────────────────────────────────────────────────

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


class DDL(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.ReLU(),
            nn.Linear(64, 128),         nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


class DynamicMLP(nn.Module):
    def __init__(self, hidden_sizes=[128, 64], num_classes=3, dropout_rate=0.3):
        super().__init__()
        self.input_layer = nn.LazyLinear(hidden_sizes[0])
        layers = []
        prev = hidden_sizes[0]
        for h in hidden_sizes[1:]:
            layers += [nn.Linear(prev, h), nn.ReLU(),
                       nn.BatchNorm1d(h), nn.Dropout(dropout_rate)]
            prev = h
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer  = nn.Linear(prev, num_classes)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        return self.output_layer(self.hidden_layers(x))


# ── Modelos de representación ──────────────────────────────────────────────────

def _train_torch(model, X_scaled, seed, l1_lambda=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_tensor = torch.FloatTensor(X_scaled)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    for _ in range(EPOCHS):
        model.train()
        z, xr = model(X_tensor)
        loss  = mse(xr, X_tensor)
        if l1_lambda > 0:
            loss = loss + l1_lambda * z.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def _get_latent(model, X_scaled):
    model.eval()
    with torch.no_grad():
        z, _ = model(torch.FloatTensor(X_scaled))
    return z.numpy()


def fit_transform_dl(X_train, X_test, latent_dim, transform_algorithm, seed=42):
    scaler     = StandardScaler()
    X_tr_sc    = scaler.fit_transform(X_train)
    X_te_sc    = scaler.transform(X_test)
    dl = DictionaryLearning(n_components=latent_dim,
                             transform_algorithm=transform_algorithm,
                             fit_algorithm='lars', max_iter=500,
                             random_state=seed, n_jobs=1)
    X_lat_tr = dl.fit_transform(X_tr_sc)
    X_lat_te = dl.transform(X_te_sc)
    recon    = float(np.mean((X_tr_sc - X_lat_tr @ dl.components_) ** 2))
    return X_lat_tr, X_lat_te, recon


def fit_transform_ae(X_train, X_test, latent_dim, seed=42):
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    model   = Autoencoder(X_train.shape[1], latent_dim)
    model   = _train_torch(model, X_tr_sc, seed)
    recon   = float(nn.MSELoss()(
        model(torch.FloatTensor(X_tr_sc))[1],
        torch.FloatTensor(X_tr_sc)).item())
    return _get_latent(model, X_tr_sc), _get_latent(model, X_te_sc), recon


def fit_transform_ddl(X_train, X_test, latent_dim, l1_lambda, seed=42):
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    model   = DDL(X_train.shape[1], latent_dim)
    model   = _train_torch(model, X_tr_sc, seed, l1_lambda=l1_lambda)
    recon   = float(nn.MSELoss()(
        model(torch.FloatTensor(X_tr_sc))[1],
        torch.FloatTensor(X_tr_sc)).item())
    return _get_latent(model, X_tr_sc), _get_latent(model, X_te_sc), recon


def learn_representations(method, X_raw_train, X_raw_test, repr_cfg, seed=42):
    """
    Aprende representaciones para las 3 modalidades y fusiona.
    Devuelve (X_fused_train, X_fused_test, mean_recon_error).
    """
    lats_tr, lats_te, recons = [], [], []
    # Forzar tipos correctos — pandas puede convertir int a float en DataFrames
    latent_dim = int(repr_cfg['latent_dim'])

    for mod in MODALITIES:
        if method == 'dl':
            lt, lte, re = fit_transform_dl(
                X_raw_train[mod], X_raw_test[mod],
                latent_dim, repr_cfg['transform_algorithm'], seed)
        elif method == 'ae':
            lt, lte, re = fit_transform_ae(
                X_raw_train[mod], X_raw_test[mod],
                latent_dim, seed)
        elif method == 'ddl':
            lt, lte, re = fit_transform_ddl(
                X_raw_train[mod], X_raw_test[mod],
                latent_dim, float(repr_cfg['l1_lambda']), seed)
        lats_tr.append(lt); lats_te.append(lte); recons.append(re)

    return np.hstack(lats_tr), np.hstack(lats_te), float(np.mean(recons))


# ── GridSearch de representación ───────────────────────────────────────────────

def gridsearch_representations(method, X_raw_train, X_raw_test,
                                y_train, seed, results_dir):
    """
    Para una semilla dada, itera sobre todas las configuraciones del modelo de
    representación y evalúa accuracy CV con SVM(rbf) como clasificador proxy
    (rápido). Devuelve la mejor configuración.

    Nota: usamos SVM(rbf) solo para seleccionar la mejor representación.
    El GridSearch de clasificadores completo se hace después sobre esa representación.
    """
    space  = REPR_SPACES[method]
    keys   = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))

    proxy_clf = Pipeline([
        ('scaler',    StandardScaler()),
        ('selector',  SelectKBest(f_classif, k=10)),
        ('clf',       SVC(C=1, kernel='rbf', random_state=seed)),
    ])
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)

    rows = []
    for combo in combos:
        cfg = dict(zip(keys, combo))
        try:
            X_fused_tr, _, recon = learn_representations(
                method, X_raw_train, X_raw_test, cfg, seed=42)
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(proxy_clf, X_fused_tr, y_train,
                                     cv=skf, scoring='accuracy', n_jobs=-1)
            rows.append({**cfg,
                         'cv_acc':     round(float(scores.mean()), 4),
                         'recon_error':round(recon, 6)})
        except Exception as e:
            print(f"    REPR cfg {cfg} failed: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, f'gridsearch_repr_log_{method}_s{seed}.csv'),
              index=False)

    best_idx = df['cv_acc'].idxmax()
    best_cfg = df.loc[best_idx].to_dict()
    # Convertir tipos numpy a Python nativos
    for k in keys:
        if isinstance(best_cfg[k], (np.integer,)):
            best_cfg[k] = int(best_cfg[k])
        elif isinstance(best_cfg[k], (np.floating,)):
            best_cfg[k] = float(best_cfg[k])
    print(f"    Mejor repr cfg (s={seed}): "
          f"{dict((k, best_cfg[k]) for k in keys)}  "
          f"cv_acc={best_cfg['cv_acc']:.4f}  "
          f"recon={best_cfg['recon_error']:.4f}")
    return best_cfg


# ── Clasificadores (igual que Joint Search) ────────────────────────────────────

def get_classifiers_config(seed):
    mutual_info_wrs = partial(mutual_info_classif, random_state=seed)
    rf_selector     = RandomForestClassifier(n_estimators=100, random_state=seed)

    classifiers_config = {
        'RF': {
            'model': RandomForestClassifier(),
            'params': {
                'model__criterion':        ['gini', 'entropy', 'log_loss'],
                'model__n_estimators':     [40, 60],
                'model__max_depth':        [3, None],
                'model__min_samples_split':[1, 3],
            }
        },
        'SVM': {
            'model': SVC(probability=True),
            'params': {
                'model__C':      [0.1, 1, 10],
                'model__gamma':  [0.001],
                'model__kernel': ['linear', 'rbf'],
            }
        },
        'CatBoost': {
            'model': CatBoostClassifier(silent=True),
            'params': {
                'model__iterations':  [1000],
                'model__depth':       [5, 7],
                'model__l2_leaf_reg': [1, 2],
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'model__n_neighbors': [3, 5, 7],
                'model__weights':     ['distance'],
                'model__leaf_size':   [5, 15, 30],
                'model__metric':      ['euclidean'],
            }
        },
        'ElasticNet': {
            'model': SGDClassifier(penalty='elasticnet'),
            'params': {
                'model__loss':     ['log_loss', 'perceptron'],
                'model__alpha':    [0.0001],
                'model__l1_ratio': [0.15, 0.5, 0.65],
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
                'model__lr':                    [0.0005],
                'model__module__hidden_sizes':  [[128, 64, 32]],
                'model__module__dropout_rate':  [0.4, 0.5],
            }
        },
    }
    return classifiers_config, mutual_info_wrs, rf_selector


# ── Pipeline principal por semilla ─────────────────────────────────────────────

def run_seed(method, seed, X_raw_train, X_raw_test, y_train, y_test,
             results_dir):
    set_seed(seed)
    print(f"\n  [seed={seed}] Buscando mejor configuración de representación...")

    # 1. GridSearch de representación (proxy rápido)
    best_repr_cfg = gridsearch_representations(
        method, X_raw_train, X_raw_test, y_train, seed, results_dir)

    # 2. Aprender representaciones con la mejor config
    X_fused_tr, X_fused_te, recon = learn_representations(
        method, X_raw_train, X_raw_test, best_repr_cfg, seed=seed)

    print(f"  Espacio fusionado: {X_fused_tr.shape}  recon_error={recon:.4f}")

    # 3. GridSearch de clasificadores (igual que Joint Search)
    classifiers_config, mutual_info_wrs, rf_selector = get_classifiers_config(seed)
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)

    seed_rows = []

    for clf_name, config in classifiers_config.items():
        print(f"    [{clf_name}] GridSearch clasificador...", end=' ', flush=True)

        pipeline = Pipeline([
            ('scaler',   StandardScaler()),
            ('selector', SelectKBest(f_classif)),
            ('model',    config['model']),
        ])

        param_grid = []
        for func in [f_classif, mutual_info_wrs]:
            param_grid.append({
                'selector':    [SelectKBest(func)],
                'selector__k': K_VALUES,
                **config['params'],
            })
        param_grid.append({
            'selector':             [SelectFromModel(rf_selector,
                                                     threshold=-np.inf)],
            'selector__max_features': K_VALUES,
            **config['params'],
        })

        try:
            grid = GridSearchCV(pipeline, param_grid, cv=skf,
                                scoring='accuracy', n_jobs=1)
            grid.fit(X_fused_tr, y_train)

            best_pipe = grid.best_estimator_
            y_pred    = best_pipe.predict(X_fused_te)
            y_score   = (best_pipe.predict_proba(X_fused_te)
                         if hasattr(best_pipe, 'predict_proba') else None)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)

            # Selector info
            best_sel = grid.best_params_['selector']
            if isinstance(best_sel, SelectKBest):
                func      = best_sel.score_func
                func_name = (func.func.__name__ if hasattr(func, 'func')
                             else func.__name__)
                origin = f"KBest_{func_name}"
                k_win  = grid.best_params_['selector__k']
            else:
                origin = f"FromModel_{type(best_sel.estimator).__name__}"
                k_win  = grid.best_params_['selector__max_features']

            print(f"acc_cv={grid.best_score_:.4f}  acc_test={acc:.4f}")

            if acc > 0.95:
                print(f"    ⚠️  acc={acc:.4f} > 0.95 — verificar leakage")

            seed_rows.append({
                'Seed':        seed,
                'Modelo':      clf_name,
                'Metodo':      method.upper(),
                'Selector':    origin,
                'k':           k_win,
                'latent_dim':  best_repr_cfg['latent_dim'],
                'recon_error': round(recon, 6),
                'CV_Acc':      round(grid.best_score_, 4),
                'Test_Acc':    round(acc,  4),
                'Precision':   round(prec, 4),
                'Recall':      round(rec,  4),
                'F1':          round(f1,   4),
                'Params':      str(grid.best_params_),
            })

            # Gráficos
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, display_labels=['CN', 'DCL', 'EA'], cmap='Blues')
            plt.title(f'CM {method.upper()} {clf_name} (s={seed})')
            plt.savefig(os.path.join(results_dir,
                        f'cm_{method}_{clf_name}_s{seed}.png'))
            plt.close()

            if y_score is not None:
                y_bin = label_binarize(y_test, classes=[0, 1, 2])
                plt.figure(figsize=(7, 5))
                for i, (name_, color) in enumerate(
                        zip(['CN','DCL','EA'], ['steelblue','darkorange','green'])):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                    plt.plot(fpr, tpr, color=color, lw=2,
                             label=f'{name_} (AUC={auc(fpr,tpr):.2f})')
                plt.plot([0,1],[0,1],'k--',lw=1)
                plt.title(f'ROC {method.upper()} {clf_name} (s={seed})')
                plt.legend(loc='lower right')
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir,
                            f'roc_{method}_{clf_name}_s{seed}.png'))
                plt.close()

            # Reporte individual
            with open(os.path.join(results_dir,
                      f'reporte_{method}_{clf_name}_s{seed}.txt'), 'w') as f:
                repr_keys = list(REPR_SPACES[method].keys())
                f.write(f"MÉTODO: {method.upper()} | MODELO: {clf_name} "
                        f"| SEMILLA: {seed}\n")
                f.write(f"Mejor repr config: "
                        f"{dict((k, best_repr_cfg[k]) for k in repr_keys)}\n")
                f.write(f"Recon error: {recon:.6f}\n")
                f.write(f"Selector ganador: {origin} con k={k_win}\n")
                f.write(f"Mejores params clf: {grid.best_params_}\n")
                f.write(f"Métricas Test: Acc={acc:.4f}, Prec={prec:.4f}, "
                        f"Rec={rec:.4f}, F1={f1:.4f}\n")

        except Exception as e:
            print(f"ERROR: {e}")

    return seed_rows


# ── Resumen final ──────────────────────────────────────────────────────────────

def save_final_summary(method, all_rows, results_dir):
    df   = pd.DataFrame(all_rows)
    path = os.path.join(results_dir, f'resumen_maestro_{method}.csv')
    df.to_csv(path, index=False)

    # Media ± std por clasificador
    summary_path = os.path.join(results_dir, f'resumen_final_{method}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"# {method.upper()} — Resumen Final\n")
        f.write(f"# Evaluación: media ± std sobre {len(SEEDS)} semillas\n\n")
        for clf_name in df['Modelo'].unique():
            sub = df[df['Modelo'] == clf_name]
            f.write(f"## {clf_name}\n")
            for metric in ['Test_Acc', 'Precision', 'Recall', 'F1']:
                m = sub[metric].mean()
                s = sub[metric].std()
                f.write(f"- {metric}: {m:.4f} ± {s:.4f}\n")
            f.write(f"- latent_dim más frecuente: "
                    f"{sub['latent_dim'].mode().iloc[0]}\n")
            f.write(f"- recon_error medio: {sub['recon_error'].mean():.6f}\n\n")

    print(f"\n  Resumen maestro: {path}")
    print(f"  Resumen final:   {summary_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='all',
                        choices=['dl', 'ae', 'ddl', 'all'])
    args    = parser.parse_args()
    methods = ['ae', 'ddl'] if args.method == 'all' else [args.method]

    # Cargar datos
    print("Cargando datos...")
    X_raw_train, X_raw_test = {}, {}
    for mod in MODALITIES:
        tr = pd.read_csv(f'../data/train_{mod}.csv')
        te = pd.read_csv(f'../data/test_{mod}.csv')
        X_raw_train[mod] = tr.drop(columns=['Sujeto','Target_class']).values.astype(np.float32)
        X_raw_test[mod]  = te.drop(columns=['Sujeto','Target_class']).values.astype(np.float32)
    y_train = pd.read_csv('../data/train_DTI.csv')['Target_class'].values
    y_test  = pd.read_csv('../data/test_DTI.csv')['Target_class'].values
    print(f"  Train: {len(y_train)}  Test: {len(y_test)}")
    print(f"  Clases train: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    for method in methods:
        results_dir = f'results_joint_{method}'
        os.makedirs(results_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  MÉTODO: {method.upper()}")
        print(f"{'='*60}")

        all_rows = []
        for seed in SEEDS:
            rows = run_seed(method, seed, X_raw_train, X_raw_test,
                            y_train, y_test, results_dir)
            all_rows.extend(rows)

        save_final_summary(method, all_rows, results_dir)

    print(f"\n{'='*60}")
    print("  Proceso completado.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
