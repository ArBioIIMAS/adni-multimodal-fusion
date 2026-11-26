"""
analisis_unimodales.py
======================
Análisis exploratorio de resultados unimodales del Joint Search (Outputs in English).
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

warnings.filterwarnings('ignore')

# ── Configuración y Traducción ────────────────────────────────────────────────
MODALIDADES  = ['Morfológicos', 'Metabólicos', 'DTI']
ALGORITMOS   = ['ElasticNet', 'KNN', 'SVM', 'RF', 'CatBoost', 'MLP']
METRICAS     = ['Train Acc.', 'Accuracy', 'Precision', 'Recall', 'F1']
SEEDS        = [7, 14, 21, 28, 35]
OUT_DIR      = 'resultados_unimodales'

MOD_TRANSLATE = {
    'Morfológicos': 'MRI',
    'Metabólicos':  'PET',
    'DTI':          'DTI'
}

PALETTE      = sns.color_palette('tab10', n_colors=len(ALGORITMOS))
COLOR_MAP    = dict(zip(ALGORITMOS, PALETTE))
MOD_COLORS   = {'MRI': '#DD8452', 'PET': '#55A868', 'DTI': '#4C72B0'}

os.makedirs(f'{OUT_DIR}/tablas',  exist_ok=True)
os.makedirs(f'{OUT_DIR}/figuras', exist_ok=True)

# ── Diccionarios Anatómicos y Función Limpiadora ─────────────────────────────
COLORES_ROI = {
    'amig_der': '#E84545',  
    'amig_izq': '#F4A0A0',  
    'hip_der':  '#2E86AB',  
    'hip_izq':  '#90CAE4',  
    'tal_der':  '#4CAF50',  
    'tal_izq':  '#A8D8A8',  
}

def clean_roi_label(raw_name):
    """
    Extrae la anatomía, asigna acrónimo, limpia guiones y traduce métricas.
    """
    ROI_ACRONYMS = {
        'amig_der': 'RA', 'amig_izq': 'LA',
        'hip_der':  'RH', 'hip_izq':  'LH',
        'tal_der':  'RT', 'tal_izq':  'LT',
    }
    
    TRANSLATIONS = {
        'tortuosidad': 'Tortuosity',
        'compacidad':  'Compacity',
        'volumen':     'Volume',
        'volumenn':    'Norm. Volume',
    }

    UPPERCASE_METRICS = ['FA', 'MD', 'MO']

    clean_name = raw_name.lower()
    acronym = ''
    
    # 1. Extraer la anatomía PRIMERO y asignarle el acrónimo
    for esp_roi, acr in ROI_ACRONYMS.items():
        if esp_roi in clean_name:
            acronym = acr
            clean_name = clean_name.replace(esp_roi, '')
            break
            
    # 2. Limpiamos los guiones bajos sobrantes
    clean_name = clean_name.strip('_')
    
    # Si la variable original era SOLO la anatomía, asumimos que es Volumen
    if not clean_name:
        return f"Volume ({acronym})" if acronym else "Volume"
        
    # 3. Partimos y traducimos lo que sobra
    parts = [p for p in clean_name.split('_') if p]
    clean_parts = []
    
    for p in parts:
        if p in TRANSLATIONS:
            clean_parts.append(TRANSLATIONS[p])
        elif p.upper() in UPPERCASE_METRICS:
            clean_parts.append(p.upper())
        else:
            clean_parts.append(p.title())
            
    final_metric_name = ' '.join(clean_parts).strip()
    
    # 4. Ensamblaje final
    if acronym:
        return f"{final_metric_name} ({acronym})"
    return final_metric_name

def get_color_roi(roi_name):
    for key, color in COLORES_ROI.items():
        if key in roi_name:
            return color
    return '#BBBBBB' 

# ── Helpers Generales ─────────────────────────────────────────────────────────
def cargar_resumen(modo):
    path = f'{modo}_resumen_maestro_joint.csv'
    if not os.path.exists(path):
        print(f'  ⚠️  No encontrado: {path}')
        return None
    df = pd.read_csv(path)
    df['Modalidad'] = modo
    df['Modality_EN'] = MOD_TRANSLATE.get(modo, modo)
    df = df.rename(columns={'CV_Acc': 'Train Acc.', 'Test_Acc': 'Accuracy'})
    return df

def cargar_importancias(modo):
    pattern = f'{modo}/importancia_ROIs_*_s*.csv'
    files = glob.glob(pattern)
    if not files:
        print(f'  ⚠️  Sin archivos de importancia en {modo}/')
        return None

    partes = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        imp_min = df['Importancia'].min()
        imp_max = df['Importancia'].max()
        if imp_max > imp_min:
            df['Importancia'] = (df['Importancia'] - imp_min) / (imp_max - imp_min)
        partes.append(df)

    all_imp = pd.concat(partes, ignore_index=True)
    return all_imp


# ── Funciones de Trazado Agrupado ──────────────────────────────────────────────
def figura_top10_todas_modalidades(out_dir, modalidades):
    fig, axes = plt.subplots(1, len(modalidades), figsize=(16, 6))

    for ax, modo in zip(axes, modalidades):
        all_imp = cargar_importancias(modo)   
        if all_imp is None:
            ax.set_visible(False)
            continue

        imp_media = (all_imp.groupby('ROI')['Importancia']
                     .mean()
                     .sort_values(ascending=False)
                     .head(10))

        colores = [get_color_roi(roi) for roi in imp_media.index[::-1]]
        etiquetas_limpias = [clean_roi_label(roi) for roi in imp_media.index[::-1]]

        ax.barh(etiquetas_limpias, imp_media.values[::-1],
                color=colores, alpha=0.92, edgecolor='white', linewidth=0.5)

        ax.set_xlim(0, 1)
        ax.set_xlabel('Mean Norm. Importance', fontsize=10)
        ax.set_title(MOD_TRANSLATE.get(modo, modo), fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.25)
        ax.tick_params(axis='y', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    legend_elements = [
        mpatches.Patch(color='#E84545', label='Right Amygdala (RA)'),
        mpatches.Patch(color='#F4A0A0', label='Left Amygdala (LA)'),
        mpatches.Patch(color='#2E86AB', label='Right Hippocampus (RH)'),
        mpatches.Patch(color='#90CAE4', label='Left Hippocampus (LH)'),
        mpatches.Patch(color='#4CAF50', label='Right Thalamus (RT)'),
        mpatches.Patch(color='#A8D8A8', label='Left Thalamus (LT)'),
        mpatches.Patch(color='#BBBBBB', label='Other'),
    ]
    fig.legend(handles=legend_elements,
               loc='lower center', ncol=4,
               fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.08))

    plt.suptitle('Top 10 biomarkers per Modality\n(normalized importance [0,1], colored by region and laterality)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/figuras/09_top10_rois_todas_modalidades.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✅ Fig 9 guardada')


def figura_selector_k_unificada(out_dir, modalidades, dfs):
    # Diccionario con el orden exacto deseado
    short_labels = {
        'KBest_f_classif':                  'ANOVA',
        'KBest_mutual_info_classif':        'MI',
        'FromModel_RandomForestClassifier': 'MDI (RF)',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    width = 0.25

    # ── Panel A: Selector ─────────────────────────────────────────────────────
    # Forzar el orden de los selectores basándose en los valores del diccionario
    all_selectors = list(short_labels.values())
    x = np.arange(len(all_selectors))

    for i, modo in enumerate(modalidades):
        df = dfs.get(modo)
        if df is None: continue
        modo_en = MOD_TRANSLATE.get(modo, modo)
        
        sel_mapped = df['Selector'].map(short_labels).fillna(df['Selector'])
        counts = [sel_mapped.value_counts().get(s, 0) for s in all_selectors]
        
        bars = ax1.bar(x + i * width, counts, width,
                       label=modo_en, color=MOD_COLORS[modo_en], alpha=0.87)
        for bar, val in zip(bars, counts):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.1,
                         str(val), ha='center', va='bottom', fontsize=9)

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(all_selectors, fontsize=11)
    ax1.set_xlabel('Feature selector', fontsize=11)
    ax1.set_yticks([3*i for i in range(1,6)])
    ax1.set_ylim(0, 17 * 1.12)
    ax1.set_ylabel('Number of optimized models', fontsize=11)
    ax1.set_title('Optimal feature selector per modality', fontsize=13)
    ax1.legend(title='Modality', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Panel B: k features ───────────────────────────────────────────────────
    all_k = sorted(set(k for df in dfs.values() for k in df['k'].unique()))
    x_k = np.arange(len(all_k))

    for i, modo in enumerate(modalidades):
        df = dfs.get(modo)
        if df is None: continue
        modo_en = MOD_TRANSLATE.get(modo, modo)
        counts = [df['k'].value_counts().get(k, 0) for k in all_k]
        bars = ax2.bar(x_k + i * width, counts, width,
                       label=modo_en, color=MOD_COLORS[modo_en], alpha=0.87)
        for bar, val in zip(bars, counts):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.1,
                         str(val), ha='center', va='bottom', fontsize=9)

    ax2.set_xticks(x_k + width)
    ax2.set_xticklabels([str(k) for k in all_k], fontsize=11)
    ax2.set_xlabel('Number of selected features (k)', fontsize=11)
    ax2.set_yticks([3*i for i in range(1,6)])
    ax2.set_ylim(0, 17 * 1.12)
    ax2.set_ylabel('Number of optimized models', fontsize=11)
    ax2.set_title('Optimal number of selected features (k) per modality', fontsize=13)
    ax2.legend(title='Modality', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/figuras/05_selector_k_unificado.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✅ Fig 5 guardada')

# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS POR MODALIDAD
# ══════════════════════════════════════════════════════════════════════════════

todos = []
dfs_por_modalidad = {}

for modo in MODALIDADES:
    modo_en = MOD_TRANSLATE.get(modo, modo)
    print(f'\n{"─"*55}')
    print(f'  Procesando: {modo} ({modo_en})')
    print(f'{"─"*55}')

    df = cargar_resumen(modo)
    if df is None:
        continue
    dfs_por_modalidad[modo] = df
    todos.append(df)

    # ── 1. Tablas ─────────────────────────────────────────────────────────────
    stats_rows = []
    for algo, grp in df.groupby('Modelo'):
        row = {'Model': algo}
        for m in METRICAS:
            if m in grp.columns:
                row[f'{m} mean'] = round(grp[m].mean(), 4)
                row[f'{m} std']  = round(grp[m].std(),  4)
        row['Selector_top'] = grp['Selector'].mode().iloc[0]
        row['k_top']        = int(grp['k'].mode().iloc[0])
        stats_rows.append(row)

    df_stats = pd.DataFrame(stats_rows).sort_values('Train Acc. mean', ascending=False)
    df_stats.to_csv(f'{OUT_DIR}/tablas/resumen_stats_{modo_en}.csv', index=False)
    print(f'  ✅ Tablas stats guardadas')

    # ══════════════════════════════════════════════════════════════════════
    # FIGURAS SIMPLES (1, 2, 4, 7)
    # ══════════════════════════════════════════════════════════════════════
    algos_presentes = [a for a in ALGORITMOS if a in df['Modelo'].unique()]

    # Fig 1
    fig, ax = plt.subplots(figsize=(9, 5))
    width_bar = 0.35
    cv_means  = [df[df['Modelo']==a]['Train Acc.'].mean()  for a in algos_presentes]
    cv_stds   = [df[df['Modelo']==a]['Train Acc.'].std()   for a in algos_presentes]
    tst_means = [df[df['Modelo']==a]['Accuracy'].mean() for a in algos_presentes]
    tst_stds  = [df[df['Modelo']==a]['Accuracy'].std()  for a in algos_presentes]

    xi = np.arange(len(algos_presentes))
    ax.bar(xi - width_bar/2, cv_means,  width_bar, yerr=cv_stds,
           capsize=4, label='Train Acc.',   color='#4C72B0', alpha=0.85)
    ax.bar(xi + width_bar/2, tst_means, width_bar, yerr=tst_stds,
           capsize=4, label='Accuracy', color='#DD8452', alpha=0.85)

    ax.set_xticks(xi)
    ax.set_xticklabels(algos_presentes, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'CV vs Test Accuracy — {modo_en}\n(mean ± std, 5 seeds)', fontsize=13)
    ax.legend(fontsize=11)
    #ax.axhline(1/2, color='gray', linestyle=':', lw=1, label='Chance (50%)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/figuras/01_cv_test_acc_{modo_en}.png', dpi=150)
    plt.close()

    # Fig 4
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ['Train Acc.', 'Accuracy']):
        sns.boxplot(data=df, x='Modelo', y=metric,
                    order=algos_presentes, palette=COLOR_MAP,
                    width=0.5, ax=ax, fliersize=0)
        sns.stripplot(data=df, x='Modelo', y=metric,
                      order=algos_presentes, palette=COLOR_MAP,
                      size=7, jitter=True, alpha=0.8, ax=ax)
        ax.set_ylim(0, 1.05)
        ax.axhline(1/2, color='gray', linestyle=':', lw=1)
        ax.set_title(f'{metric} per seed — {modo_en}', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/figuras/04_varianza_semillas_{modo_en}.png', dpi=150)
    plt.close()

    # Fig 7
    pivot = df.pivot_table(index='Seed', columns='Modelo', values='Train Acc.', aggfunc='mean')
    if not pivot.empty:
        pivot = pivot[pivot.mean().sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
                    vmin=0.4, vmax=1.0, linewidths=0.5,
                    ax=ax, cbar_kws={'label': 'Train Accuracy'})
        ax.set_title(f'CV Accuracy — seed × algorithm — {modo_en}', fontsize=12)
        ax.set_xlabel('Model')
        ax.set_ylabel('Seed')
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/figuras/07_heatmap_cv_acc_{modo_en}.png', dpi=150)
        plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # FIGURA 6 (Usando la nueva limpieza de ROIs)
    # ══════════════════════════════════════════════════════════════════════
    all_imp = cargar_importancias(modo)
    if all_imp is not None:
        imp_media = (all_imp.groupby('ROI')['Importancia']
                     .mean()
                     .sort_values(ascending=False)
                     .head(25))

        etiquetas_limpias = [clean_roi_label(roi) for roi in imp_media.index[::-1]]

        fig, ax = plt.subplots(figsize=(9, 7))
        colors_imp = plt.cm.viridis(np.linspace(0.2, 0.85, len(imp_media)))
        ax.barh(etiquetas_limpias, imp_media.values[::-1],
                color=colors_imp[::-1], alpha=0.9)
        ax.set_xlabel('Mean Importance', fontsize=11)
        ax.set_title(f'Top 25 ROIs — {modo_en}\n(mean across models and seeds)',
                     fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/figuras/06_importancia_rois_{modo_en}.png', dpi=150)
        plt.close()
        print(f'  ✅ Fig 6 guardada')


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS COMPARATIVO ENTRE MODALIDADES
# ══════════════════════════════════════════════════════════════════════════════

if len(todos) > 1:
    print(f'\n{"═"*55}')
    print('Análisis comparativo entre modalidades')
    print(f'{"═"*55}')

    df_all = pd.concat(todos, ignore_index=True)
    
    # ══════════════════════════════════════════════════════════════════════
    # FIGURA 3 y 8
    # ══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    hue_order_en = [MOD_TRANSLATE[m] for m in MODALIDADES]

    for ax, metric in zip(axes, ['Train Acc.', 'Accuracy']):
        sns.boxplot(data=df_all, x='Modelo', y=metric,
                    hue='Modality_EN', order=ALGORITMOS,
                    palette=MOD_COLORS, width=0.6,
                    hue_order=hue_order_en, ax=ax)
        #ax.axhline(1/2, color='gray', linestyle=':', lw=1.2, label='Chance')
        ax.set_ylim(0.4, 1.05)
        ax.set_title(f'{metric} — Modality Comparison', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel(metric, fontsize=11)
        ax.legend(title='Modality', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', labelsize=9)

    plt.suptitle('Comparison across modalities (5 seeds)', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/figuras/03_comparativa_modalidades.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    metricas_plot = ['Train Acc.', 'Accuracy', 'F1']
    x_rank = np.arange(len(metricas_plot))
    width_rank = 0.25

    for i, modo in enumerate(MODALIDADES):
        sub = df_all[df_all['Modalidad'] == modo]
        modo_en = MOD_TRANSLATE.get(modo, modo)
        means = [sub[m].mean() for m in metricas_plot]
        stds  = [sub[m].std()  for m in metricas_plot]
        ax.bar(x_rank + i*width_rank, means, width_rank, yerr=stds, capsize=4,
               label=modo_en, color=MOD_COLORS[modo_en], alpha=0.85)

    ax.set_xticks(x_rank + width_rank)
    ax.set_xticklabels(metricas_plot, fontsize=12)
    ax.set_ylim(0.4, 1.05)
    ax.axhline(1/2, color='gray', linestyle=':', lw=1)
    ax.set_ylabel('Mean value (all seeds and models)', fontsize=11)
    ax.set_title('Global Ranking per Modality', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/figuras/08_ranking_modalidades.png', dpi=150)
    plt.close()

figura_selector_k_unificada(OUT_DIR, MODALIDADES, dfs_por_modalidad)
figura_top10_todas_modalidades(OUT_DIR, MODALIDADES)

# ══════════════════════════════════════════════════════════════════════
# Tabla — Mejor modelo por modalidad (tabla visual)
# ══════════════════════════════════════════════════════════════════════
print('\n  Mejor modelo por modalidad (Train Acc.):')
best_rows = []
for modo in MODALIDADES:
    sub = df_all[df_all['Modalidad'] == modo]
    if sub.empty: continue
    best = sub.loc[sub['Train Acc.'].idxmax()]
    best_rows.append({
        'Modalidad': modo,
        'Modelo':    best['Modelo'],
        'Seed':      int(best['Seed']),
        'Selector':  best['Selector'],
        'k':         int(best['k']),
        'Train Acc.':  round(best['Train Acc.'], 4),
        'Accuracy':    round(best['Accuracy'], 4),
        'Precision':   round(best['Precision'], 4),
        'Sensibilidad':round(best['Recall'], 4),
        'F1':          round(best['F1'], 4),
    })
df_best = pd.DataFrame(best_rows)
print(df_best.to_string(index=False))
df_best.to_csv(f'{OUT_DIR}/tablas/mejor_modelo_por_modalidad.csv', index=False)

print(f'\n{"═"*55}')
print(f'  ✅ Análisis completado.')
print(f'  📁 Salidas en: {OUT_DIR}/')
print(f'{"═"*55}')

# ══════════════════════════════════════════════════════════════════════
# TABLA RESUMEN GENERAL (Media ± SD por Modelo y Modalidad)
# ══════════════════════════════════════════════════════════════════════
if 'df_all' in locals() and not df_all.empty:
    print(f'\n{"─"*55}')
    print(' Generando tabla de métricas agregadas (Mean ± SD)...')
    print(f'{"─"*55}')

    # 1. Agrupar por Modalidad y Modelo para calcular media y std de cada métrica
    cols_metricas = ['Train Acc.', 'Accuracy', 'Precision', 'Recall', 'F1']
    
    # Calcular media y std
    grouped = df_all.groupby(['Modalidad', 'Modelo'])
    means = grouped[cols_metricas].mean()
    stds = grouped[cols_metricas].std()

    # 2. Formatear como "Mean ± SD"
    df_formatted = pd.DataFrame(index=means.index)
    for col in cols_metricas:
        df_formatted[col] = means[col].map('{:.4f}'.format) + ' ± ' + stds[col].map('{:.4f}'.format)

    # Añadir moda del Selector y k
    df_formatted['Selector_top'] = grouped['Selector'].agg(lambda x: x.mode().iloc[0])
    df_formatted['k_top'] = grouped['k'].agg(lambda x: int(x.mode().iloc[0]))

    # Resetear índice para guardar en CSV
    df_formatted_res = df_formatted.reset_index()
    
    # Ordenar por el orden de modalidades y media de Train Acc.
    df_formatted_res['Mod_Order'] = df_formatted_res['Modalidad'].map({'Morfológicos': 1, 'Metabólicos': 2, 'DTI': 3})
    df_formatted_res['Train_Mean_Num'] = means['Train Acc.'].values
    df_formatted_res = df_formatted_res.sort_values(['Mod_Order', 'Train_Mean_Num'], ascending=[True, False]).drop(columns=['Mod_Order', 'Train_Mean_Num'])

    # 3. Guardar en CSV
    csv_out_path = f'{OUT_DIR}/tablas/metricas_mean_sd_por_modelo.csv'
    df_formatted_res.to_csv(csv_out_path, index=False)
    print(f'  ✅ Tabla CSV guardada en: {csv_out_path}')

    # 4. Generar y mostrar código LaTeX (Booktabs)
    short_selectors = {
        'KBest_f_classif': 'ANOVA',
        'KBest_mutual_info_classif': 'MI',
        'FromModel_RandomForestClassifier': 'MDI (RF)'
    }
    
    latex_str = []
    latex_str.append(r"\begin{table}[htbp]")
    latex_str.append(r"\centering")
    latex_str.append(r"\caption{Rendimiento de clasificación por algoritmo y modalidad (Promedio $\pm$ Desviación Estándar sobre 5 semillas).}")
    latex_str.append(r"\label{tab:metricas_unimodales_mean_sd}")
    latex_str.append(r"\small")
    latex_str.append(r"\begin{tabular}{l l c c c c c}")
    latex_str.append(r"\hline")
    latex_str.append(r"\textbf{Modalidad} & \textbf{Modelo} & \textbf{Selector} & \textbf{$k$} & \textbf{Train Acc.} & \textbf{Accuracy} & \textbf{F1-Score} \\")
    latex_str.append(r"\hline")

    for modo in ['Morfológicos', 'Metabólicos', 'DTI']:
        modo_en = MOD_TRANSLATE.get(modo, modo)
        sub_df = df_formatted_res[df_formatted_res['Modalidad'] == modo]
        
        for i, (_, row) in enumerate(sub_df.iterrows()):
            mod_label = f"\\multirow{{{len(sub_df)}}}{{*}}{{{modo_en}}}" if i == 0 else ""
            sel_clean = short_selectors.get(row['Selector_top'], row['Selector_top'])
            
            line = f"{mod_label} & {row['Modelo']} & {sel_clean} & {row['k_top']} & {row['Train Acc.']} & {row['Accuracy']} & {row['F1']} \\\\"
            latex_str.append(line)
        latex_str.append(r"\hline")

    latex_str.append(r"\end{tabular}")
    latex_str.append(r"\end{table}")

    latex_code = "\n".join(latex_str)
    
    # Guardar LaTeX en archivo .tex
    tex_out_path = f'{OUT_DIR}/tablas/tabla_metricas_mean_sd.tex'
    with open(tex_out_path, 'w') as f:
        f.write(latex_code)
    print(f'  ✅ Código LaTeX guardado en: {tex_out_path}')

    print('\n── Vista previa del código LaTeX generado ──\n')
    print(latex_code)
    
# ══════════════════════════════════════════════════════════════════════
# ANÁLISIS POR FACTOR: (Modalidad × Selector) y (Modalidad × k)
# ══════════════════════════════════════════════════════════════════════
if 'df_all' in locals() and not df_all.empty:
    print(f'\n{"─"*55}')
    print(' Generando estadísticas agregadas por Selector y por k...')
    print(f'{"─"*55}')

    short_selectors = {
        'KBest_f_classif': 'ANOVA',
        'KBest_mutual_info_classif': 'MI',
        'FromModel_RandomForestClassifier': 'MDI (RF)'
    }

    df_calc = df_all.copy()
    df_calc['Selector_Clean'] = df_calc['Selector'].map(short_selectors).fillna(df_calc['Selector'])

    # ── 1. Agrupación: Modalidad × Selector ─────────────────────────────────
    grp_sel = df_calc.groupby(['Modalidad', 'Selector_Clean'])
    res_sel = []
    for (modo, sel), grp in grp_sel:
        res_sel.append({
            'Modalidad': modo,
            'Modality_EN': MOD_TRANSLATE.get(modo, modo),
            'Selector': sel,
            'Count (models × seeds)': len(grp),
            'Accuracy_Mean': grp['Accuracy'].mean(),
            'Accuracy_SD': grp['Accuracy'].std(),
            'F1_Mean': grp['F1'].mean(),
            'F1_SD': grp['F1'].std(),
            'Accuracy': f"{grp['Accuracy'].mean():.4f} ± {grp['Accuracy'].std():.4f}",
            'F1-Score': f"{grp['F1'].mean():.4f} ± {grp['F1'].std():.4f}"
        })
    df_res_sel = pd.DataFrame(res_sel)
    df_res_sel.to_csv(f'{OUT_DIR}/tablas/stats_por_modalidad_y_selector.csv', index=False)

    # ── 2. Agrupación: Modalidad × k ─────────────────────────────────────────
    grp_k = df_calc.groupby(['Modalidad', 'k'])
    res_k = []
    for (modo, k_val), grp in grp_k:
        res_k.append({
            'Modalidad': modo,
            'Modality_EN': MOD_TRANSLATE.get(modo, modo),
            'k': int(k_val),
            'Count (models × seeds)': len(grp),
            'Accuracy_Mean': grp['Accuracy'].mean(),
            'Accuracy_SD': grp['Accuracy'].std(),
            'F1_Mean': grp['F1'].mean(),
            'F1_SD': grp['F1'].std(),
            'Accuracy': f"{grp['Accuracy'].mean():.4f} ± {grp['Accuracy'].std():.4f}",
            'F1-Score': f"{grp['F1'].mean():.4f} ± {grp['F1'].std():.4f}"
        })
    df_res_k = pd.DataFrame(res_k)
    df_res_k.to_csv(f'{OUT_DIR}/tablas/stats_por_modalidad_y_k.csv', index=False)

    # ── 3. Generar LaTeX para Selector ───────────────────────────────────────
    tex_sel = []
    tex_sel.append(r"\begin{table}[htbp]")
    tex_sel.append(r"\centering")
    tex_sel.append(r"\caption{Rendimiento de clasificación por mecanismo de selección de características y modalidad ($\text{Media} \pm \text{DE}$).}")
    tex_sel.append(r"\label{tab:stats_selector}")
    tex_sel.append(r"\small")
    tex_sel.append(r"\begin{tabular}{l l c c c}")
    tex_sel.append(r"\hline")
    tex_sel.append(r"\textbf{Modalidad} & \textbf{Selector} & \textbf{Frecuencia ($N$)} & \textbf{Accuracy} & \textbf{F1-Score} \\")
    tex_sel.append(r"\hline")

    for modo in ['Morfológicos', 'Metabólicos', 'DTI']:
        modo_en = MOD_TRANSLATE.get(modo, modo)
        sub = df_res_sel[df_res_sel['Modalidad'] == modo].sort_values('Accuracy_Mean', ascending=False)
        for i, (_, row) in enumerate(sub.iterrows()):
            mod_col = f"\\multirow{{{len(sub)}}}{{*}}{{{modo_en}}}" if i == 0 else ""
            acc = row['Accuracy'].replace('±', r'\pm')
            f1 = row['F1-Score'].replace('±', r'\pm')
            tex_sel.append(f"{mod_col} & {row['Selector']} & {row['Count (models × seeds)']} & ${acc}$ & ${f1}$ \\\\")
        tex_sel.append(r"\hline")

    tex_sel.append(r"\end{tabular}")
    tex_sel.append(r"\end{table}")

    with open(f'{OUT_DIR}/tablas/tabla_stats_selector.tex', 'w') as f:
        f.write("\n".join(tex_sel))

    # ── 4. Generar LaTeX para k ──────────────────────────────────────────────
    tex_k = []
    tex_k.append(r"\begin{table}[htbp]")
    tex_k.append(r"\centering")
    tex_k.append(r"\caption{Rendimiento de clasificación por número de características ($k$) y modalidad ($\text{Media} \pm \text{DE}$).}")
    tex_k.append(r"\label{tab:stats_k}")
    tex_k.append(r"\small")
    tex_k.append(r"\begin{tabular}{l c c c c}")
    tex_k.append(r"\hline")
    tex_k.append(r"\textbf{Modalidad} & \textbf{$k$} & \textbf{Frecuencia ($N$)} & \textbf{Accuracy} & \textbf{F1-Score} \\")
    tex_k.append(r"\hline")

    for modo in ['Morfológicos', 'Metabólicos', 'DTI']:
        modo_en = MOD_TRANSLATE.get(modo, modo)
        sub = df_res_k[df_res_k['Modalidad'] == modo].sort_values('k', ascending=True)
        for i, (_, row) in enumerate(sub.iterrows()):
            mod_col = f"\\multirow{{{len(sub)}}}{{*}}{{{modo_en}}}" if i == 0 else ""
            acc = row['Accuracy'].replace('±', r'\pm')
            f1 = row['F1-Score'].replace('±', r'\pm')
            tex_k.append(f"{mod_col} & {row['k']} & {row['Count (models × seeds)']} & ${acc}$ & ${f1}$ \\\\")
        tex_k.append(r"\hline")

    tex_k.append(r"\end{tabular}")
    tex_k.append(r"\end{table}")

    with open(f'{OUT_DIR}/tablas/tabla_stats_k.tex', 'w') as f:
        f.write("\n".join(tex_k))

    print('  ✅ Tablas guardadas en CSV y LaTeX.')


# ══════════════════════════════════════════════════════════════════════
# GRÁFICAS DE CURVAS DE RENDIMIENTO Y SENSIBILIDAD POR FACTORES
# ══════════════════════════════════════════════════════════════════════
if 'df_all' in locals() and not df_all.empty:
    print(f'\n{"─"*55}')
    print(' Generando curvas de rendimiento e interacción...')
    print(f'{"─"*55}')

    df_plot_all = df_all.copy()
    
    # ── GRÁFICA: INTERACCIÓN SELECTOR VS k SEPARADA POR MODALIDAD ────────────
    short_selectors = {
        'KBest_f_classif': 'ANOVA',
        'KBest_mutual_info_classif': 'MI',
        'FromModel_RandomForestClassifier': 'MDI (RF)'
    }
    df_plot_all['Selector_Clean'] = df_plot_all['Selector'].map(short_selectors).fillna(df_plot_all['Selector'])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    modalidades_ord = ['Morfológicos', 'Metabólicos', 'DTI']
    selectors_ord = list(short_selectors.values())

    # Paleta y marcadores por selector para mantener uniformidad
    sel_palette = {'ANOVA': '#1f77b4', 'MI': '#ff7f0e', 'MDI (RF)': '#2ca02c'}
    sel_markers = ['o', 's', 'D']
    sel_linestyles = ['-', '--', '-.']

    for ax, modo in zip(axes, modalidades_ord):
        sub_df = df_plot_all[df_plot_all['Modalidad'] == modo]
        modo_en = MOD_TRANSLATE.get(modo, modo)
        
        if sub_df.empty:
            ax.set_visible(False)
            continue

        sns.pointplot(
            data=sub_df, 
            x='k', 
            y='Accuracy', 
            hue='Selector_Clean',
            hue_order=selectors_ord,
            palette=sel_palette,
            markers=sel_markers,
            linestyles=sel_linestyles,
            capsize=0.1, 
            dodge=0.2,
            ax=ax
        )

        ax.set_title(modo_en, fontsize=12)#, fontweight='bold')
        if modo== 'Metabólicos':
            ax.set_xlabel('Number of selected features ($k$)', fontsize=11)
        ax.grid(True, axis='y', linestyle=':', alpha=0.5)
        ax.set_ylim(0.4, 1.0)
        
        if ax != axes[0]:
            ax.set_ylabel('')
            #ax.get_legend().remove()
        else:
            ax.set_ylabel('Test Accuracy', fontsize=11)
        ax.legend(title='Selector', fontsize=9, title_fontsize=10, loc=('upper center' if modo =='Metabólicos' else 'lower center'))

    plt.suptitle('Interaction between feature selector and number of selected features ($k$) per Modality')#, fontsize=13, y=1.02)
    plt.tight_layout()
    fig2_path = f'{OUT_DIR}/figuras/11_interaccion_selector_k_facet.png'
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  ✅ Fig 10 (Interacción Selector vs k) guardada en: {fig2_path}')