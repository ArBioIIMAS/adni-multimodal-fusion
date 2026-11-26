"""
analisis_fusion_tardia.py
=========================
Generación de tablas y figuras de resultados para experimentos de Fusión Tardía,
ordenado estrictamente por familias de métodos y algoritmos base.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ── Configuración de Directorios y Colores ────────────────────────────────────
OUT_DIR = 'resultados_fusion'
os.makedirs(f'{OUT_DIR}/tablas', exist_ok=True)
os.makedirs(f'{OUT_DIR}/figuras', exist_ok=True)

# Algoritmos base y mapeo de colores original de unimodal_analysis_4.py
ALGORITMOS_BASE = ['ElasticNet', 'KNN', 'SVM', 'RF', 'CatBoost']
PALETTE = sns.color_palette('tab10', n_colors=len(ALGORITMOS_BASE))
COLOR_MAP = dict(zip(ALGORITMOS_BASE, PALETTE))
COLOR_GRIS = '#A0A0A0' 

# Paleta para las modalidades en la gráfica de pesos
MOD_COLORS = {
    'Morfológicos': '#D62728', # Rojo
    'Metabólicos':  '#2CA02C', # Verde
    'DTI':          '#1F77B4'  # Azul
}

print("Consolidando resultados en el orden estructurado de métodos...")
data_frames = []

# ── 1A. Cargar Baselines Unimodales ──────────────────────────────────────────
modalidades_nombres = ['Morfológicos', 'Metabólicos', 'DTI']
for modo in modalidades_nombres:
    archivo_unimodal = f'{modo}_modelos_seleccionados.csv'
    if os.path.exists(archivo_unimodal):
        df_uni = pd.read_csv(archivo_unimodal)
        mejor = df_uni.iloc[[0]].copy() 
        
        mejor_fila = {
            'Estrategia_Global': 'Baseline Unimodal',
            'Configuracion': f'Mejor {modo} ({mejor["Modelo"].values[0]})',
            'CV_Acc': mejor['CV_Acc'].values[0],
            'Test_Acc': mejor['Test_Acc'].values[0],
            'Macro_Prec': np.nan,
            'Macro_Rec': np.nan,
            'Macro_F1': np.nan
        }
        data_frames.append(pd.DataFrame([mejor_fila]))

# ── 1B. Cargar Resultados de Fusión ──────────────────────────────────────────
archivos_fusion = {
    'Metricas_Hard_voting_Final.csv': 'Hard Voting',
    'Metricas_Soft_voting_Final.csv': 'Soft Voting',
    'Metricas_Stacking_por_algoritmo.csv': 'Stacking',
    'Metricas_Stacking_mejores_por_modalidad.csv': 'Stacking',
    'Metricas_Stacking_Top2.csv': 'Stacking',
    'Metricas_Stacking_Top3.csv': 'Stacking'
}

for archivo, estrategia in archivos_fusion.items():
    if os.path.exists(archivo):
        df_tmp = pd.read_csv(archivo)
        df_tmp['Estrategia_Global'] = estrategia
        
        if 'Algoritmo' in df_tmp.columns:
            df_tmp['Configuracion'] = df_tmp['Algoritmo']
        elif 'Estrategia' in df_tmp.columns:
            df_tmp['Configuracion'] = df_tmp['Estrategia']
            
        if 'Acc_Global' in df_tmp.columns:
            df_tmp['Test_Acc'] = df_tmp['Acc_Global']
            
        if 'CV_Acc' not in df_tmp.columns:
            df_tmp['CV_Acc'] = np.nan
            
        data_frames.append(df_tmp)

if not data_frames:
    print("❌ No se encontraron archivos para procesar.")
    exit()

df_all = pd.concat(data_frames, ignore_index=True)

# Homologar nombres de las combinaciones especiales
df_all['Configuracion'] = df_all['Configuracion'].replace({
    'mejores_por_modalidad': 'Top 1',
    'Top-2': 'Top 2',
    'Top-3': 'Top 3'
})

# Generar etiqueta única para mapeo y visualización
df_all['Modelo_Fusion'] = df_all['Estrategia_Global'] + ' - ' + df_all['Configuracion']

# ── 2. Definición del Orden Estricto de Filas ─────────────────────────────────
# Mapeo explícito para ordenar la tabla según tu requerimiento
lista_orden_filas = []
colores_grafica = []

# Bloques ordenados de Hard y Soft Voting
for est in ['Hard Voting', 'Soft Voting']:
    for algo in ALGORITMOS_BASE:
        etiqueta = f"{est} - {algo}"
        if etiqueta in df_all['Modelo_Fusion'].values:
            lista_orden_filas.append(etiqueta)
            colores_grafica.append(COLOR_MAP[algo])

# Bloque ordenado de Stacking (Algoritmos base primero, combinaciones después)
orden_stacking_interno = ALGORITMOS_BASE + ['Top 1', 'Top 2', 'Top 3']
for algo in orden_stacking_interno:
    etiqueta = f"Stacking - {algo}"
    if etiqueta in df_all['Modelo_Fusion'].values:
        lista_orden_filas.append(etiqueta)
        if algo in ALGORITMOS_BASE:
            colores_grafica.append(COLOR_MAP[algo])
        else:
            colores_grafica.append(COLOR_GRIS)

# Bloque de Baselines Unimodales al final de la tabla
baselines_presentes = df_all[df_all['Estrategia_Global'] == 'Baseline Unimodal']['Modelo_Fusion'].tolist()
lista_orden_filas.extend(baselines_presentes)

# Reindexar el DataFrame maestro con el nuevo orden estricto
df_all = df_all.set_index('Modelo_Fusion').loc[lista_orden_filas].reset_index()

# ── 3. Guardar la Tabla Maestra Ordenada ──────────────────────────────────────
columnas_tabla = ['Modelo_Fusion', 'Estrategia_Global', 'Configuracion', 'CV_Acc', 'Test_Acc', 'Macro_Prec', 'Macro_Rec', 'Macro_F1']
columnas_existentes = [c for c in columnas_tabla if c in df_all.columns]

tabla_maestra = df_all[columnas_existentes].copy()
tabla_path = f'{OUT_DIR}/tablas/01_Resumen_Mestro_Fusion_vs_Unimodal.csv'
tabla_maestra.to_csv(tabla_path, index=False)
print(f"✅ Tabla maestra ordenada de forma estructurada guardada en: {tabla_path}")


# ── 4. Generación de la Gráfica de Rendimiento Horizontal ─────────────────────
print("\nGenerando gráfica de rendimiento...")
plt.figure(figsize=(11, 8))

# Excluir los baselines unimodales de la gráfica para mantener foco en fusión tardía
df_grafica = df_all[df_all['Estrategia_Global'] != 'Baseline Unimodal'].copy()

# En matplotlib/seaborn, el eje Y se grafica de abajo hacia arriba.
# Para que 'Hard Voting - ElasticNet' aparezca arriba del todo, invertimos la lista de datos y colores.
df_grafica_inv = df_grafica.iloc[::-1].reset_index(drop=True)
colores_grafica_inv = colores_grafica[::-1]

ax = sns.barplot(
    data=df_grafica_inv,
    x='Test_Acc',
    y='Modelo_Fusion',
    palette=colores_grafica_inv,
    edgecolor='black',
    linewidth=0.7
)

# Añadir valores numéricos exactos al final de cada barra
for p in ax.patches:
    width = p.get_width()
    if width > 0:
        ax.text(
            width + 0.01,
            p.get_y() + p.get_height() / 2,
            f'{width:.4f}',
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold'
        )

plt.title('Rendimiento Global (Accuracy) por Estrategia de Fusión Tardía', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('Global Accuracy', fontsize=12, fontweight='bold')
plt.ylabel('Estrategia y Configuración', fontsize=12, fontweight='bold')
plt.xlim(0, 1.05)
plt.grid(axis='x', linestyle='--', alpha=0.5)

# Añadir leyenda descriptiva de los algoritmos
legend_patches = [plt.Rectangle((0,0),1,1, color=COLOR_MAP[algo], edgecolor='black') for algo in ALGORITMOS_BASE]
legend_patches.append(plt.Rectangle((0,0),1,1, color=COLOR_GRIS, edgecolor='black'))
legend_labels = ALGORITMOS_BASE + ['Comb. Especiales (Tops)']
plt.legend(legend_patches, legend_labels, title='Algoritmos / Modelos Base', loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figuras/01_Rendimiento_Estrategias.png', dpi=150)
plt.close()
print("✅ Gráfica de rendimiento generada respetando la jerarquía metodológica.")


# ── 5. Figura de Estabilidad de Pesos (Grouped Boxplot) ──────────────────────
print("\nAnalizando varianza de pesos del Meta-Modelo (Stacking)...")
archivos_pesos = glob.glob('Pesos_Comite_*.csv')

if archivos_pesos:
    df_pesos_list = []
    for f in archivos_pesos:
        df_p = pd.read_csv(f)
        nombre_estrategia = os.path.basename(f).replace('Pesos_Comite_', '').replace('.csv', '')
        df_p['Estrategia_Stacking'] = nombre_estrategia
        df_pesos_list.append(df_p)
        
    df_pesos = pd.concat(df_pesos_list, ignore_index=True)
    
    id_vars = ['Estrategia_Stacking']
    value_vars = [c for c in df_pesos.columns if c != 'Estrategia_Stacking']
    df_melt = df_pesos.melt(id_vars=id_vars, value_vars=value_vars, 
                            var_name='Modelo_Modalidad', value_name='Peso_Asignado')
    
    def extraer_modalidad(nombre_col):
        for mod in MOD_COLORS.keys():
            if mod in nombre_col:
                return mod
        return 'Otra'
        
    df_melt['Modalidad'] = df_melt['Modelo_Modalidad'].apply(extraer_modalidad)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_melt, 
        x='Estrategia_Stacking', 
        y='Peso_Asignado', 
        hue='Modalidad',
        palette=MOD_COLORS,
        width=0.7,
        boxprops=dict(alpha=0.9)
    )
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    plt.title('Distribución de Pesos del Meta-Modelo por Estrategia y Modalidad', fontsize=14, pad=15)
    plt.xlabel('Estrategia de Stacking', fontsize=12)
    plt.ylabel('Valor del Peso (Regresión Logística)', fontsize=12)
    plt.legend(title='Modalidad', loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/figuras/02_Pesos_Boxplot_Agrupado.png', dpi=150)
    plt.close()
    print("✅ Figura de varianza de weights guardada con éxito.")

print("\n🚀 Proceso completado.")