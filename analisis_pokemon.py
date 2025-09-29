# -*- coding: utf-8 -*-
# Análisis de Datos con Pokémon (Primera Generación)
# Ejecutar en Visual Studio Code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =======================
# 1. Lectura de datos
# =======================
df = pd.read_csv("pokemon_primera_gen.csv")
print("Primeras filas del DataFrame:")
print(df.head())

# =======================
# 2. Filtrado y selección
# =======================
tipo_Fuego = df.loc[df["Tipo 1"] == "Fuego", ["Nombre", "Tipo 1", "Ataque", "Velocidad"]]
print("\nPokémon de tipo Fuego (Nombre, Tipo 1, Ataque, Velocidad):")
print(tipo_Fuego)

# =======================
# 3. Estadística descriptiva básica
# =======================
promedio_Ataques = round(df["Ataque"].mean(), 1)
mediana_Ataques = df["Ataque"].median()
moda_Ataques = df["Ataque"].mode()[0]

print("\n--- Estadísticas de Ataque ---")
print("Promedio de Ataque:", promedio_Ataques)
print("Mediana de Ataque:", mediana_Ataques)
print("Moda de Ataque:", moda_Ataques)

# Pokémon con mayor defensa
idx_max_defensa = df["Defensa"].idxmax()
pokemon_defensa_alta = df.loc[idx_max_defensa, "Nombre"]

# Pokémon con menor velocidad
idx_min_velocidad = df["Velocidad"].idxmin()
pokemon_menor_velocidad = df.loc[idx_min_velocidad, "Nombre"]

print("\nEl Pokémon con mayor defensa es:", pokemon_defensa_alta)
print("El Pokémon con menor velocidad es:", pokemon_menor_velocidad)

# Pokémon con dos tipos
dos_tipos = df["Tipo 2"].notnull().sum()
print("\nCantidad de Pokémon con dos tipos:", dos_tipos)

# Rango y desviación estándar de PS
rango_PS = df["PS"].max() - df["PS"].min()
desv_std_PS = round(df["PS"].std(), 2)

print("\n--- Estadísticas de PS ---")
print("Rango de PS:", rango_PS)
print("Desviación estándar de PS:", desv_std_PS)

# =======================
# 4. Visualización de datos
# =======================
sns.set(style="whitegrid")

# Histograma de Ataque
plt.figure(figsize=(8, 5))
sns.histplot(df["Ataque"], kde=True, bins=15, color="red")
plt.title("Histograma de Ataque")
plt.xlabel("Ataque")
plt.ylabel("Frecuencia")
plt.show()

# Dispersión Ataque vs Velocidad
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Ataque", y="Velocidad", data=df, hue="Tipo 1", legend=False)
plt.title("Dispersión Ataque vs Velocidad")
plt.show()

# Boxplot de PS por tipo principal
plt.figure(figsize=(12, 6))
sns.boxplot(x="Tipo 1", y="PS", data=df)
plt.title("PS por Tipo Principal")
plt.xticks(rotation=45)
plt.show()

# Diagrama de violín de Defensa por tipo
plt.figure(figsize=(12, 6))
sns.violinplot(x="Tipo 1", y="Defensa", data=df)
plt.title("Distribución de Defensa por Tipo Principal")
plt.xticks(rotation=45)
plt.show()

# =======================
# 5. Manipulación de datos
# =======================
df["Poder Total"] = df["Ataque"] + df["Defensa"] + df["Velocidad"] + df["PS"]
df_ordenado = df.sort_values("Poder Total", ascending=False)

print("\nTop 5 Pokémon con mayor Poder Total:")
print(df_ordenado[["Nombre", "Tipo 1", "Poder Total"]].head())

# =======================
# 6. Agrupamiento y análisis por grupo
# =======================
# Estadísticas de ataque por Tipo 1
ataque_stats = df.groupby("Tipo 1")["Ataque"].agg(["mean", "median", "std"])
print("\nEstadísticas de Ataque por Tipo 1:")
print(ataque_stats)

# Tipo con mayor velocidad promedio
velocidad_media = df.groupby("Tipo 1")["Velocidad"].mean().sort_values(ascending=False)
print("\nPromedio de Velocidad por Tipo 1:")
print(velocidad_media)
print("\nTipo con mayor promedio de Velocidad:", velocidad_media.index[0])

# Pokémon con mayor y menor PS por tipo
ps_extremos = df.groupby("Tipo 1").apply(
    lambda x: pd.Series({
        "Pokemon Mayor PS": x.loc[x["PS"].idxmax(), "Nombre"],
        "Max PS": x["PS"].max(),
        "Pokemon Menor PS": x.loc[x["PS"].idxmin(), "Nombre"],
        "Min PS": x["PS"].min()
    })
)
print("\nPokémon con mayor y menor PS por tipo:")
print(ps_extremos)

# =======================
# 7. Análisis exploratorio (EDA)
# =======================
# Tipos con mayor ataque y defensa
print("\nTop 3 tipos con mayor ataque promedio:")
print(ataque_stats.sort_values("mean", ascending=False).head(3))

defensa_stats = df.groupby("Tipo 1")["Defensa"].mean().sort_values(ascending=False)
print("\nTop 3 tipos con mayor defensa promedio:")
print(defensa_stats.head(3))

# Correlación ataque-velocidad
corr = df["Ataque"].corr(df["Velocidad"])
print("\nCoeficiente de correlación Ataque-Velocidad:", round(corr, 3))

# Dispersión de PS (std por tipo)
ps_dispersion = df.groupby("Tipo 1")["PS"].std().sort_values(ascending=False)
print("\nDesviación estándar de PS por Tipo:")
print(ps_dispersion)

# Boxplots para detectar outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Ataque"])
plt.title("Outliers en Ataque")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["PS"])
plt.title("Outliers en PS")
plt.show()

# =======================
# 8. Interpretación (texto orientativo)
# =======================
print("\n--- Interpretación ---")
print("Algunos tipos destacan en ataque (ej. Lucha, Dragón), otros en defensa (Roca, Acero).")
print("La correlación entre Ataque y Velocidad indica si los fuertes también suelen ser rápidos.")
print("El tipo con mayor dispersión en PS es el menos homogéneo en cuanto a vida.")
print("Los outliers corresponden a Pokémon legendarios o con stats muy elevados.")