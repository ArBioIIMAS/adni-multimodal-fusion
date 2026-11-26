"""
EF_essentials.py
==============================
Exploratory analysis of Early Fusion results of models for biomarker classification.

Expected Inputs (in the working directory):
    resumen_maestro_joint.csv               — Metric results by seed and model
    estadisticas_biomarcadores_todos.csv    — Pre-calculated mean importance of ROIs

Outputs (in resultados_modelos/):
    tablas/
        resumen_stats_modelos.csv           — Mean ± std per algorithm for all metrics
    figuras/
        01_comparativa_modelos.png          — Boxplot of Train_Acc vs Test_Acc across models
        02_selector_k_distribucion.png      — Distribution of k and selectors by algorithm
        03_importancia_rois_top35.png       — Top 35 ROIs according to Mean Importance
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
MODALITIES   = ['Morphological', 'Metabolic', 'DTI']
ALGORITMOS   = ['ElasticNet', 'KNN', 'SVM', 'RF', 'CatBoost', 'MLP']
METRICAS     = ['Train Accuracy', 'Test Accuracy', 'Precision', 'Recall', 'F1-Score']
SEEDS        = [7, 14, 21, 28, 35]
OUT_DIR      = 'resultados_modelos'
PALETTE      = sns.color_palette('tab10', n_colors=len(ALGORITMOS))
COLOR_MAP    = dict(zip(ALGORITMOS, PALETTE))

# Anatomical palette with laterality
COLORES_ROI = {
    'amig_der': '#E84545',  # deep red
    'amig_izq': '#F4A0A0',  # light red
    'hip_der':  '#2E86AB',  # deep blue
    'hip_izq':  '#90CAE4',  # light blue
    'tal_der':  '#4CAF50',  # deep green
    'tal_izq':  '#A8D8A8',  # light green
}

# English labels for the anatomical legend
LEGEND_LABELS = {
    'amig_der': 'Right Amygdala (RA)',
    'amig_izq': 'Left Amygdala (LA)',
    'hip_der':  'Right Hippocampus (RH)',
    'hip_izq':  'Left Hippocampus (LH)',
    'tal_der':  'Right Thalamus (RT)',
    'tal_izq':  'Left Thalamus (LT)',
}

def setup_directories():
    """Creates the necessary directories for the outputs."""
    os.makedirs(f"{OUT_DIR}/tablas", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/figuras", exist_ok=True)

def clean_roi_label(roi_name):
    """Parses and rewrites ROI names avoiding underscores and using abbreviations."""
    translations = {
        'peak': 'Peak',
        'compacidad': 'Compacity',
        'tortuosidad': 'Tortuosity',
        'location': 'Location',
        'mean': 'Mean',
        'volume': 'Volume',
        'volumen': 'Volume',
        'area': 'Area',
    }
    
    anatomy_map = {
        'amig_der': '(RA)',
        'amig_izq': '(LA)',
        'hip_der':  '(RH)',
        'hip_izq':  '(LH)',
        'tal_der':  '(RT)',
        'tal_izq':  '(LT)'
    }
    
    lower_name = roi_name.lower()
    
    # Identify laterality/anatomy and assign suffix abbreviation
    suffix = ""
    found_anatomy = None
    for key, abbr in anatomy_map.items():
        if key in lower_name:
            suffix = f" {abbr}"
            found_anatomy = key
            break
            
    # Remove anatomy substring to clean up the feature description
    remaining_text = lower_name
    if found_anatomy:
        remaining_text = remaining_text.replace(found_anatomy, "")
        
    # Split by underscore and map to clean English terms
    parts = remaining_text.split('_')
    cleaned_parts = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in translations:
            cleaned_parts.append(translations[part])
        elif part.upper() in ['FA', 'MD', 'MO', 'PET']:  # Specific DTI metrics
            cleaned_parts.append(part.upper())
        else:
            cleaned_parts.append(part.capitalize())
            
    return " ".join(cleaned_parts) + suffix

def generar_tabla_estadisticas(df):
    """Calculates Mean ± Std for metrics mapped to English names."""
    stats_list = []
    
    # Internal mapping from original CSV columns to English names
    metric_mapping = {
        'CV_Acc': 'Train Accuracy',
        'Test_Acc': 'Test Accuracy',
        'Precision': 'Precision',
        'Recall': 'Recall',
        'F1': 'F1-Score'
    }
    
    for modelo in ALGORITMOS:
        df_modelo = df[df['Modelo'] == modelo]
        if df_modelo.empty:
            continue
        fila_stats = {'Model': modelo}
        
        for orig_col, eng_col in metric_mapping.items():
            if orig_col in df_modelo.columns:
                media = df_modelo[orig_col].mean()
                std = df_modelo[orig_col].std()
                fila_stats[eng_col] = f"{media:.4f} ± {std:.4f}"
                
        stats_list.append(fila_stats)
        
    df_stats = pd.DataFrame(stats_list)
    out_path = f"{OUT_DIR}/tablas/resumen_stats_modelos.csv"
    df_stats.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[*] Table saved to: {out_path}")
    stats_l = df_stats.to_latex(index=False, float_format="%.4f")
    print(stats_l)

def plot_comparativa_modelos(df):
    """Generates boxplots comparing Train Accuracy and Test Accuracy."""
    plt.figure(figsize=(11, 6))
    
    # Rename columns to match requested English terminology
    df_renamed = df.rename(columns={'Modelo': 'Algorithm', 'CV_Acc': 'Train Accuracy', 'Test_Acc': 'Test Accuracy'})
    
    # Melt data for standard comparative visualization
    df_melt = pd.melt(df_renamed, id_vars=['Algorithm'], value_vars=['Train Accuracy', 'Test Accuracy'], 
                      var_name='Metric', value_name='Accuracy')
    
    sns.boxplot(data=df_melt, x='Algorithm', y='Accuracy', hue='Metric', 
                palette='Set2', order=ALGORITMOS)
    
    plt.title('Model Comparison: Train Accuracy vs Test Accuracy', fontsize=14, pad=15)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.legend(title='Metric')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    out_path = f"{OUT_DIR}/figuras/01_comparativa_modelos.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[*] Figure 1 saved to: {out_path}")

def plot_distribucion_selector_k(df):
    """Generates vertical frequency plots for selectors and k values using short labels."""
    df_plot = df.copy()
    
    # Define short labels dictionary requested
    short_labels = {
        'KBest_f_classif': 'ANOVA',
        'KBest_mutual_info_classif': 'MI',
        'FromModel_RandomForestClassifier': 'MDI (RF)'
    }
    
    # Safely replace matching substrings or values in Selector column
    for raw_label, short_label in short_labels.items():
        df_plot['Selector'] = df_plot['Selector'].astype(str).str.replace(raw_label, short_label, regex=False)
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    # ORDEN EXPLÍCITO PARA LOS SELECTORES
    selector_order = ['ANOVA', 'MI', 'MDI (RF)']
    # Plot 1: Selectors (X-axis = Selector, Y-axis = Frequency, Hue = Algorithm)
    sns.countplot(
        data=df_plot, 
        x='Selector', 
        order=selector_order,  # <--- Se agrega order aquí
        hue='Modelo', 
        hue_order=ALGORITMOS, 
        palette=COLOR_MAP, 
        ax=axes[0]
    )
    axes[0].set_title('Optimal feature selector per model', fontsize=12, pad=10)
    axes[0].set_xlabel('Feature selector', fontsize=11)
    axes[0].set_ylabel('Number of optimized models', fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    axes[0].legend(title='Models', bbox_to_anchor=(0.9, 0.8), loc='upper right', borderaxespad=0)

    #if axes[0].get_legend() is not None:
    #    axes[0].get_legend().remove()
    
    # Plot 2: Distribution of k (X-axis = k value, Y-axis = Frequency, Hue = Algorithm)
    sns.countplot(
        data=df_plot, 
        x='k', 
        hue='Modelo', 
        hue_order=ALGORITMOS, 
        palette=COLOR_MAP, 
        ax=axes[1]
    )
    axes[1].set_title('Optimal number of selected features (k) per model', fontsize=12, pad=10)
    axes[1].set_xlabel('Number of selected features (k)', fontsize=11)
    axes[1].set_ylabel('Number of optimized models', fontsize=11)
    axes[1].set_ylim(0,4)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    axes[1].legend(title='Models', bbox_to_anchor=(0.15, 0.8), loc='upper left', borderaxespad=0)
    
    plt.tight_layout()
    out_path = f"{OUT_DIR}/figuras/02_selector_k_distribucion.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[*] Figure 2 saved to: {out_path}")

def obtener_color_roi(nombre_roi):
    """Returns assigned color based on ROI keyword or a neutral default."""
    for roi_key, color in COLORES_ROI.items():
        if roi_key in nombre_roi.lower():
            return color
    return '#B0B0B0'

def obtener_modalidad(nombre_roi):
    """Determina la modalidad (MRI, PET, DTI) basándose en el nombre de la ROI."""
    nombre_upper = nombre_roi.upper()
    # Si tiene métricas de tractografía, es DTI
    if any(metric in nombre_upper for metric in ['_FA', '_MD', '_MO']):
        return 'DTI'
    # Si tiene PET
    elif 'PET' in nombre_upper:
        return 'PET'
    # De lo contrario (volumen, área, compacidad, tortuosidad), es MRI
    else:
        return 'MRI'
        
def plot_importancia_rois(df_biomarcadores):
    """Plots the top ROIs by Mean Importance using cleaned English abbreviations and modality labels."""
    df_top = df_biomarcadores.sort_values(by='Imp_Media', ascending=False).head(35).copy()
    
    # Apply dynamic label cleaning avoiding underscores and adding abbreviations
    df_top['Clean_ROI'] = df_top['ROI'].apply(clean_roi_label)
    colores = [obtener_color_roi(roi) for roi in df_top['ROI']]
    
    plt.figure(figsize=(14, 8)) # Se aumentó un poco el alto para dar espacio a las etiquetas
    ax = sns.barplot(data=df_top, x='Clean_ROI', y='Imp_Media', palette=colores)
    
    plt.title('Top 35 early fusion neuroimaging biomarkers for classification (AD, MCI, NC)', fontsize=15, pad=15)
    plt.ylabel('Mean of normalized importance [0,1]', fontsize=12)
    plt.xlabel('Biomarkers (Anatomy/Laterality)', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # --- NUEVO: Agregar la modalidad encima de cada barra ---
    # Para ajustar la altura del texto dinámicamente y que no choque con la barra
    max_y = df_top['Imp_Media'].max()
    offset = max_y * 0.02 
    
    for i, (index, row) in enumerate(df_top.iterrows()):
        modalidad = obtener_modalidad(row['ROI'])
        ax.text(i, 
                row['Imp_Media'] + offset, 
                modalidad, 
                ha='center', 
                va='bottom', 
                rotation=90, 
                fontsize=10, 
                color='black',
                fontweight='bold')
    
    # Aumentar un poco el límite superior del eje Y para que no se corte el texto
    ax.set_ylim(0, max_y + (max_y * 0.15))
    
    # Anatomical legend using corrected English syntax and abbreviations
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=LEGEND_LABELS[roi]) for roi, color in COLORES_ROI.items()]
    plt.legend(handles=legend_elements, title='Anatomy / Laterality', loc='upper right')
    
    plt.tight_layout()
    out_path = f"{OUT_DIR}/figuras/03_importancia_rois_top35_.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[*] Figure 3 saved to: {out_path}")
def main():
    print("Starting updated Early Fusion analysis script...")
    setup_directories()
    
    try:
        df_maestro = pd.read_csv('resumen_maestro_joint.csv')
        df_biomarcadores = pd.read_csv('estadisticas_biomarcadores_top60.csv')
    except FileNotFoundError as e:
        print(f"Error: Target file not found - {e}")
        return
        
    generar_tabla_estadisticas(df_maestro)
    plot_comparativa_modelos(df_maestro)
    plot_distribucion_selector_k(df_maestro)
    plot_importancia_rois(df_biomarcadores)
    
    print("Analysis finished successfully with English formatting corrections.")

if __name__ == "__main__":
    main()