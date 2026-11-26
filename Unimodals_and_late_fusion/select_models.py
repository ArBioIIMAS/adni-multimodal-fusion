#!/usr/bin/env python
# coding: utf-8

import pandas as pd

def seleccionar_mejor_modelo(group):
    if len(group) < 2:
        return group
    # Solo CV: criterio que nunca toca el test set
    group = group.sort_values(['CV_Acc', 'Seed'], ascending=[False, True])
    return group.head(1)

modelos_seleccionados = {}

for modo in ['DTI', 'Metabólicos', 'Morfológicos']:
    df = pd.read_csv(f'{modo}_resumen_maestro_joint.csv')

    # Iteramos explícitamente: evita el bug de apply/include_groups en pandas >= 2.2
    grupos = []
    for modelo, group in df.groupby('Modelo'):
        grupos.append(seleccionar_mejor_modelo(group.copy()))

    resultado = pd.concat(grupos).reset_index(drop=True)

    resultado.to_csv(f'{modo}_modelos_seleccionados.csv', index=False)
    modelos_seleccionados[modo] = resultado

    print(f"\n--- Modelos Ganadores: {modo} ---")
    print(resultado[['Modelo', 'Seed', 'Selector', 'CV_Acc', 'Test_Acc']].to_string(index=False))