import os
import pandas as pd

def consolidar_dataframes(ruta_raiz):
    """
    Consolida todos los archivos CSV encontrados en una ruta raíz y sus subcarpetas en un solo DataFrame.
    Añade columnas con el nombre de la carpeta y el nombre de la carpeta padre.

    Parámetros:
    ruta_raiz (str): Ruta raíz donde empezar a buscar archivos CSV.

    Retorna:
    pd.DataFrame: DataFrame consolidado con todos los datos y las columnas adicionales.
    """
    dataframes = []

    # Recorrer todas las carpetas y subcarpetas
    for root, dirs, files in os.walk(ruta_raiz):
        for file in files:
            if file.endswith('.csv') and 'resumen_maestro' in file:
                # Ruta completa del archivo CSV
                ruta_completa = os.path.join(root, file)

                # Leer el archivo CSV en un DataFrame
                df = pd.read_csv(ruta_completa)

                # Obtener el nombre de la carpeta y el nombre de la carpeta padre
                carpeta = os.path.basename(root)
                carpeta_padre = os.path.basename(os.path.dirname(root))

                # Añadir columnas con el nombre de la carpeta y el nombre de la carpeta padre
                df['carpeta'] = carpeta
                df['carpeta_padre'] = carpeta_padre

                # Añadir el DataFrame a la lista de DataFrames
                dataframes.append(df)

    # Consolidar todos los DataFrames en uno solo
    dataframe_consolidado = pd.concat(dataframes, ignore_index=True)

    return dataframe_consolidado

if __name__ == '__main__':

    ruta_raiz = './'  # Reemplaza con la ruta raíz de tus archivos CSV
    dataframe_final = consolidar_dataframes(ruta_raiz)
    print(dataframe_final.head())
    dataframe_final.to_csv('consolidado.csv')

