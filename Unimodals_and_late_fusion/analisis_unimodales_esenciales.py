"""
analisis_unimodales.py
======================
Análisis exploratorio de resultados unimodales del Joint Search.

Entradas esperadas (en el directorio de trabajo):
    {Modalidad}_resumen_maestro_joint.csv       — una por modalidad
    {Modalidad}/importancia_ROIs_{Modelo}_s{S}.csv  — una por modelo x semilla

Salidas (en resultados_unimodales/):
    tablas/
        resumen_stats_{Modalidad}.csv           — media ± std por algoritmo
        resumen_comparativa_modalidades.csv     — comparativa entre modalidades
        selector_frecuencia_{Modalidad}.csv     — frecuencia de selector ganador
        k_frecuencia_{Modalidad}.csv            — frecuencia de k ganador
    figuras/
        01_cv_test_acc_{Modalidad}.png          — CV vs Test Acc por algoritmo
        02_metricas_radar_{Modalidad}.png       — radar Acc/Prec/Rec/F1
        03_comparativa_modalidades.png          — boxplot comparativa entre modalidades
        04_varianza_semillas_{Modalidad}.png    — variabilidad entre semillas
        05_selector_k_{Modalidad}.png           — distribución selector y k
        06_importancia_rois_{Modalidad}.png     — top ROIs agregados entre semillas
        07_heatmap_cv_acc_{Modalidad}.png       — heatmap semilla × algoritmo
"""
from copy import copy
import os
import re
import ast
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

# ── Configuración ──────────────────────────────────────────────────────────────
MODALIDADES  = ['Morfológicos', 'Metabólicos', 'DTI']
ALGORITMOS   = ['ElasticNet', 'KNN', 'SVM', 'RF', 'CatBoost', 'MLP']
METRICAS     = ['CV_Acc', 'Test_Acc', 'Precision', 'Recall', 'F1']
SEEDS        = [7, 14, 21, 28, 35]
OUT_DIR      = 'resultados_unimodales'
PALETTE      = sns.color_palette('tab10', n_colors=len(ALGORITMOS))
COLOR_MAP    = dict(zip(ALGORITMOS, PALETTE))
MOD_COLORS   = {'Morfológicos': '#DD8452', 'Metabólicos': '#55A868', 'DTI': '#4C72B0'}
# Paleta anatómica con lateralidad
COLORES_ROI = {
    'amig_der': '#E84545',  # rojo intenso
    'amig_izq': '#F4A0A0',  # rojo claro
    'hip_der':  '#2E86AB',  # azul intenso
    'hip_izq':  '#90CAE4',  # azul claro
    'tal_der':  '#4CAF50',  # verde intenso
    'tal_izq':  '#A8D8A8',  # verde claro
}
os.makedirs(f'{OUT_DIR}/tablas',  exist_ok=True)
os.makedirs(f'{OUT_DIR}/figuras', exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def cargar_resumen(modo):
    path = f'{modo}_resumen_maestro_joint.csv'
    if not os.path.exists(path):
        print(f'  ⚠️  No encontrado: {path}')
        return None
    df = pd.read_csv(path)
    df['Modalidad'] = modo
    return df


def parsear_params(params_str):
    """Extrae hiperparámetros relevantes del string de Params."""
    try:
        # Limpia el string de objetos sklearn no serializables
        clean = re.sub(r"SelectKBest\([^)]*\)", "None", params_str)
        clean = re.sub(r"SelectFromModel\([^)]*\)", "None", clean)
        d = ast.literal_eval(clean)
        # Quitar claves de selector (ya están en columnas propias)
        return {k: v for k, v in d.items()
                if k not in ('selector', 'selector__k', 'selector__max_features')}
    except Exception:
        return {}


def cargar_importancias(modo):
    """
    Carga todos los CSVs de importancia de ROIs de una modalidad
    y devuelve un DataFrame con la importancia media por ROI.
    """
    pattern = f'{modo}/importancia_ROIs_*_s*.csv'
    files = glob.glob(pattern)
    if not files:
        print(f'  ⚠️  Sin archivos de importancia en {modo}/')
        return None

    partes = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        # Normalizar a [0,1] dentro de cada run
        imp_min = df['Importancia'].min()
        imp_max = df['Importancia'].max()
        if imp_max > imp_min:
            df['Importancia'] = (df['Importancia'] - imp_min) / (imp_max - imp_min)
        partes.append(df)

    all_imp = pd.concat(partes, ignore_index=True)
    return all_imp


def get_color_roi(roi_name):
    """Extrae estructura y lateralidad del nombre del ROI."""
    for key, color in COLORES_ROI.items():
        if key in roi_name:
            return color
    return '#BBBBBB'  # gris para ROIs no identificados


def figura_top10_todas_modalidades(out_dir, modalidades):
    """
    Panel 3 columnas: top 10 ROIs por modalidad, coloreados por
    estructura anatómica y lateralidad.
    """
    fig, axes = plt.subplots(1, len(modalidades), figsize=(16, 6))

    for ax, modo in zip(axes, modalidades):
        all_imp = cargar_importancias(modo)   # ya normalizada [0,1] por run
        if all_imp is None:
            ax.set_visible(False)
            continue

        imp_media = (all_imp.groupby('ROI')['Importancia']
                     .mean()
                     .sort_values(ascending=False)
                     .head(10))

        colores = [get_color_roi(roi) for roi in imp_media.index[::-1]]

        ax.barh(imp_media.index[::-1], imp_media.values[::-1],
                color=colores, alpha=0.92, edgecolor='white', linewidth=0.5)

        ax.set_xlim(0, 1)
        ax.set_xlabel('Importancia media norm.', fontsize=10)
        ax.set_title(modo, fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.25)
        ax.tick_params(axis='y', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Leyenda compartida
    legend_elements = [
        mpatches.Patch(color='#E84545', label='Amígdala der.'),
        mpatches.Patch(color='#F4A0A0', label='Amígdala izq.'),
        mpatches.Patch(color='#2E86AB', label='Hipocampo der.'),
        mpatches.Patch(color='#90CAE4', label='Hipocampo izq.'),
        mpatches.Patch(color='#4CAF50', label='Tálamo der.'),
        mpatches.Patch(color='#A8D8A8', label='Tálamo izq.'),
        mpatches.Patch(color='#BBBBBB', label='Otros'),
    ]
    fig.legend(handles=legend_elements,
               loc='lower center', ncol=4,
               fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.08))

    plt.suptitle('Top 10 ROIs por modalidad\n(importancia normalizada [0,1], coloreada por región y lateralidad)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/figuras/09_top10_rois_todas_modalidades.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✅ Fig 9 guardada: Top 10 ROIs todas las modalidades')

def figura_selector_k_todas_modalidades(out_dir, modalidades, dfs):
    """
    Panel 3 filas × 2 columnas: selector ganador y k por modalidad.
    dfs: dict {modalidad: dataframe}
    """
    fig, axes = plt.subplots(len(modalidades), 2,
                             figsize=(12, 4 * len(modalidades)))

    short_labels = {
        'KBest_f_classif':                    'ANOVA',
        'KBest_mutual_info_classif':          'MI',
        'FromModel_RandomForestClassifier':   'RF-SFM',
    }
    col_sel = '#4C72B0'
    col_k   = '#DD8452'

    for i, modo in enumerate(modalidades):
        df = dfs.get(modo)
        ax_sel, ax_k = axes[i]

        if df is None:
            ax_sel.set_visible(False)
            ax_k.set_visible(False)
            continue

        # ── Selector ──────────────────────────────────────────────
        sel_counts = df['Selector'].value_counts()
        labels = [short_labels.get(s, s) for s in sel_counts.index]
        ax_sel.bar(labels, sel_counts.values, color=col_sel, alpha=0.85)
        ax_sel.set_title(f'{modo} — Selector', fontsize=11, fontweight='bold')
        ax_sel.set_ylabel('Frecuencia', fontsize=10)
        ax_sel.grid(axis='y', alpha=0.3)
        ax_sel.spines['top'].set_visible(False)
        ax_sel.spines['right'].set_visible(False)
        # Anotar conteo encima de cada barra
        for bar, val in zip(ax_sel.patches, sel_counts.values):
            ax_sel.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.1,
                        str(val), ha='center', va='bottom', fontsize=9)

        # ── k features ────────────────────────────────────────────
        k_counts = df['k'].value_counts().sort_index()
        ax_k.bar(k_counts.index.astype(str), k_counts.values,
                 color=col_k, alpha=0.85)
        ax_k.set_title(f'{modo} — k features', fontsize=11, fontweight='bold')
        ax_k.set_ylabel('Frecuencia', fontsize=10)
        ax_k.set_xlabel('k', fontsize=10)
        ax_k.grid(axis='y', alpha=0.3)
        ax_k.spines['top'].set_visible(False)
        ax_k.spines['right'].set_visible(False)
        for bar, val in zip(ax_k.patches, k_counts.values):
            ax_k.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.1,
                      str(val), ha='center', va='bottom', fontsize=9)

    plt.suptitle('Selector y número de features seleccionadas por modalidad',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/figuras/05_selector_k_todas_modalidades.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✅ Fig 5 guardada: Selector y k — panel unificado')

def figura_selector_k_unificada(out_dir, modalidades, dfs):
    MOD_COLORS = {
        'DTI':          '#4C72B0',
        'Metabólicos':  '#DD8452',
        'Morfológicos': '#55A868',
    }
    short_labels = {
        'KBest_f_classif':                  'ANOVA',
        'KBest_mutual_info_classif':        'MI',
        'FromModel_RandomForestClassifier': 'RF-SFM',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    width = 0.25

    # ── Panel A: Selector ─────────────────────────────────────────────────────
    all_selectors = []
    for df in dfs.values():
        all_selectors.extend(df['Selector'].unique().tolist())
    all_selectors = [short_labels.get(s, s)
                     for s in sorted(set(all_selectors))]
    x = np.arange(len(all_selectors))

    for i, modo in enumerate(modalidades):
        df = dfs.get(modo)
        if df is None: continue
        sel_mapped = df['Selector'].map(short_labels).fillna(df['Selector'])
        counts = [sel_mapped.value_counts().get(s, 0) for s in all_selectors]
        bars = ax1.bar(x + i * width, counts, width,
                       label=modo, color=MOD_COLORS[modo], alpha=0.87)
        for bar, val in zip(bars, counts):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.1,
                         str(val), ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(all_selectors, fontsize=11)
    ax1.set_ylabel('Frecuencia (modelos × semillas)', fontsize=11)
    ax1.set_title('Selector ganador por modalidad', fontsize=13)
    ax1.legend(title='Modalidad', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Panel B: k features ───────────────────────────────────────────────────
    all_k = sorted(set(k for df in dfs.values() for k in df['k'].unique()))
    x = np.arange(len(all_k))

    for i, modo in enumerate(modalidades):
        df = dfs.get(modo)
        if df is None: continue
        counts = [df['k'].value_counts().get(k, 0) for k in all_k]
        bars = ax2.bar(x + i * width, counts, width,
                       label=modo, color=MOD_COLORS[modo], alpha=0.87)
        for bar, val in zip(bars, counts):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.1,
                         str(val), ha='center', va='bottom', fontsize=9)

    ax2.set_xticks(x + width)
    ax2.set_xticklabels([str(k) for k in all_k], fontsize=11)
    ax2.set_xlabel('k features seleccionadas', fontsize=11)
    ax2.set_ylabel('Frecuencia (modelos × semillas)', fontsize=11)
    ax2.set_title('k features seleccionadas por modalidad', fontsize=13)
    ax2.legend(title='Modalidad', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/figuras/05_selector_k_unificado.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✅ Fig 5 guardada: Selector y k unificados')

# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS POR MODALIDAD
# ══════════════════════════════════════════════════════════════════════════════

todos = []
dfs_por_modalidad = {}

for modo in MODALIDADES:
    print(f'\n{"─"*55}')
    print(f'  Procesando: {modo}')
    print(f'{"─"*55}')

    df = cargar_resumen(modo)
    if df is None:
        continue
    dfs_por_modalidad[modo] = df
    todos.append(df)

    # ── 1. Tabla: media ± std por algoritmo ───────────────────────────────────
    stats_rows = []
    for algo, grp in df.groupby('Modelo'):
        row = {'Modelo': algo}
        for m in METRICAS:
            if m in grp.columns:
                row[f'{m}_mean'] = round(grp[m].mean(), 4)
                row[f'{m}_std']  = round(grp[m].std(),  4)
        # Selector más frecuente
        row['Selector_top'] = grp['Selector'].mode().iloc[0]
        row['k_top']        = int(grp['k'].mode().iloc[0])
        stats_rows.append(row)

    df_stats = pd.DataFrame(stats_rows).sort_values('CV_Acc_mean', ascending=False)
    df_stats.to_csv(f'{OUT_DIR}/tablas/resumen_stats_{modo}.csv', index=False)
    print(f'  ✅ Tabla stats guardada')
    print(df_stats[['Modelo','CV_Acc_mean','CV_Acc_std',
                    'Test_Acc_mean','Test_Acc_std','F1_mean']].to_string(index=False))

    # ── 2. Tabla: frecuencia de selector y k ──────────────────────────────────
    df['Selector'].value_counts().reset_index().rename(
        columns={'index':'Selector','Selector':'Count'}).to_csv(
        f'{OUT_DIR}/tablas/selector_frecuencia_{modo}.csv', index=False)
    df['k'].value_counts().reset_index().rename(
        columns={'index':'k','k':'Count'}).to_csv(
        f'{OUT_DIR}/tablas/k_frecuencia_{modo}.csv', index=False)



# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS COMPARATIVO ENTRE MODALIDADES
# ══════════════════════════════════════════════════════════════════════════════

if len(todos) > 1:
    print(f'\n{"═"*55}')
    print('Análisis comparativo entre modalidades')
    print(f'{"═"*55}')

    df_all = pd.concat(todos, ignore_index=True)

    # ── Tabla comparativa ────────────────────────────────────────────────
    comp_rows = []
    for modo in MODALIDADES:
        sub = df_all[df_all['Modalidad'] == modo]
        if sub.empty: continue
        for metric in METRICAS:
            if metric not in sub.columns: continue
            comp_rows.append({
                'Modalidad': modo,
                'Metrica':   metric,
                'Media':     round(sub[metric].mean(), 4),
                'Std':       round(sub[metric].std(),  4),
                'Max':       round(sub[metric].max(),  4),
                'Min':       round(sub[metric].min(),  4),
            })
    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(f'{OUT_DIR}/tablas/resumen_comparativa_modalidades.csv', index=False)
    print('\n  Comparativa (Test_Acc):')
    print(df_comp[df_comp['Metrica']=='Test_Acc'][
        ['Modalidad','Media','Std','Max','Min']].to_string(index=False))

    # ══════════════════════════════════════════════════════════════════════
    # FIGURA 3 — Boxplot comparativa modalidades × algoritmo
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric in zip(axes, ['CV_Acc', 'Test_Acc']):
        sns.boxplot(data=df_all, x='Modelo', y=metric,
                    hue='Modalidad', order=ALGORITMOS,
                    palette=MOD_COLORS, width=0.6,
                    hue_order=MODALIDADES, ax=ax)
        ax.axhline(1/2, color='gray', linestyle=':', lw=1.2, label='Azar')
        ax.set_ylim(0, 1.05)
        ax.set_title(f'{metric} — comparativa modalidades', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel(metric, fontsize=11)
        ax.legend(title='Modalidad', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', labelsize=9)

    plt.suptitle('Comparativa entre modalidades (5 semillas)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/figuras/03_comparativa_modalidades.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✅ Fig 3 guardada: Comparativa modalidades')

    # ══════════════════════════════════════════════════════════════════════
    # FIGURA 8 — Ranking de modalidades por métrica (barras agrupadas)
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 5))
    metricas_plot = ['CV_Acc', 'Test_Acc', 'F1']
    x = np.arange(len(metricas_plot))
    width = 0.25

    for i, modo in enumerate(MODALIDADES):
        sub = df_all[df_all['Modalidad'] == modo]
        means = [sub[m].mean() for m in metricas_plot]
        stds  = [sub[m].std()  for m in metricas_plot]
        ax.bar(x + i*width, means, width, yerr=stds, capsize=4,
               label=modo, color=MOD_COLORS[modo], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metricas_plot, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(1/2, color='gray', linestyle=':', lw=1)
    ax.set_ylabel('Valor medio (todas las semillas y modelos)', fontsize=11)
    ax.set_title('Ranking global por modalidad', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/figuras/08_ranking_modalidades.png', dpi=150)
    plt.close()
    print('  ✅ Fig 8 guardada: Ranking global modalidades')

    # ══════════════════════════════════════════════════════════════════════
    # Tabla — Mejor modelo por modalidad (tabla visual)
    # ══════════════════════════════════════════════════════════════════════
    print('\n  Mejor modelo por modalidad (CV_Acc):')
    best_rows = []
    for modo in MODALIDADES:
        sub = df_all[df_all['Modalidad'] == modo]
        if sub.empty: continue
        best = sub.loc[sub['CV_Acc'].idxmax()]
        best_rows.append({
            'Modalidad': modo,
            'Modelo':    best['Modelo'],
            'Seed':      int(best['Seed']),
            'Selector':  best['Selector'],
            'k':         int(best['k']),
            'Train Acc.':  round(best['CV_Acc'], 4),
            'Accuracy':    round(best['Test_Acc'], 4),
            'Precision':   round(best['Precision'], 4),
            'Sensibilidad':round(best['Recall'], 4),
            'F1':          round(best['F1'], 4),
        })
    df_best = pd.DataFrame(best_rows)
    print(df_best.to_string(index=False))
    df_best.to_csv(f'{OUT_DIR}/tablas/mejor_modelo_por_modalidad.csv', index=False)

figura_selector_k_unificada(OUT_DIR, MODALIDADES, dfs_por_modalidad)

figura_top10_todas_modalidades(OUT_DIR, MODALIDADES)

print(f'\n{"═"*55}')
print(f'  ✅ Análisis completado.')
print(f'  📁 Salidas en: {OUT_DIR}/')
print(f'{"═"*55}')
