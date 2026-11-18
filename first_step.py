import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Настройки отображения
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

FILE_PATH = "Circulation_2021_500k.csv"

# 1. Загрузка данных (учитываем возможные скрытые пропуски)
df = pd.read_csv(FILE_PATH, na_values=['?', 'None', 'null', ''], low_memory=False)

# Оценка веса и размерности (Пункты 2, 4)
file_size_mb = os.path.getsize(FILE_PATH) / (1024 ** 2)
ram_size_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

print("=" * 60)
print(f"Размер файла на диске: {file_size_mb:.2f} МБ")
print(f"Объём в оперативной памяти (RAM): {ram_size_mb:.2f} МБ")
print(f"Размерность выборки: {df.shape[0]} строк, {df.shape[1]} столбцов")
print("=" * 60)

# 2. Первичная статистика и типы данных (Пункты 2, 6)
print("\n--- Список столбцов и их типы ---")
print(df.dtypes)
print("\n--- Сводка по типам ---")
print(df.dtypes.value_counts())

# Выделяем числовые и категориальные признаки (исключая технические ID)
id_cols = [col for col in df.columns if 'id' in col.lower()]
num_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in id_cols]
cat_cols = [col for col in df.select_dtypes(include=['object', 'string', 'category', 'bool']).columns if col not in id_cols]


if num_cols:
    print("\n--- Описательная статистика (Числовые признаки) ---")
    print(df[num_cols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']])

if cat_cols:
    print("\n--- Описательная статистика (Категориальные признаки) ---")
    print(df[cat_cols].describe().T[['count', 'unique', 'top', 'freq']])

# 3. Анализ пропусков и дубликатов (Пункт 7)
print("\n--- Пропуски в данных ---")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Пропусков (шт)': missing, 'Доля (%)': missing_pct})
print(missing_df[missing_df['Пропусков (шт)'] > 0].sort_values(by='Доля (%)', ascending=False))
print(f"\nПолных дубликатов строк: {df.duplicated().sum()}")

# 4. Визуализация корреляций без ID (Пункт 5)
if len(num_cols) > 1:
    plot_num_cols = num_cols
    if len(num_cols) > 15:
        # Выбираем топ-15 признаков с наибольшей дисперсией (уже без id)
        top_var_cols = df[num_cols].var().nlargest(15).index.tolist()
        plot_num_cols = top_var_cols
        print(f"\n[Инфо] Признаков > 15. Для графиков взяты топ-15 по дисперсии: {plot_num_cols}")

    # Heatmap
    plt.figure(figsize=(12, 10))
    corr_matrix = df[plot_num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Корреляционная матрица числовых признаков (без ID)")
    plt.tight_layout()
    plt.savefig("heatmap_numeric.png", dpi=150)
    plt.close()
    print("График 'heatmap_numeric.png' сохранён.")

    # Pairplot (берём топ-5 признаков и подвыборку 2000 строк для читаемости)
    pair_cols = plot_num_cols[:5]
    sample_size = min(2000, len(df))
    sample_df = df[pair_cols].dropna().sample(n=sample_size, random_state=42)
    sns.pairplot(sample_df, diag_kind='kde', plot_kws={'alpha': 0.4})
    plt.savefig("pairplot_key_features.png", dpi=150)
    plt.close()
    print("График 'pairplot_key_features.png' сохранён.")

# 5. Очистка данных: пропуски и выбросы (Пункт 7)
df_clean = df.copy()

# Удаляем технические ID и полные дубликаты
df_clean = df_clean.drop(columns=id_cols, errors='ignore')
df_clean = df_clean.drop_duplicates()

# Удаляем столбцы, где пропусков больше 50% или всего 1 уникальное значение (константы)
cols_to_drop = [col for col in df_clean.columns if df_clean[col].isnull().mean() > 0.5 or df_clean[col].nunique() <= 1]
if cols_to_drop:
    print(f"\nУдалены неинформативные/пустые столбцы: {cols_to_drop}")
    df_clean = df_clean.drop(columns=cols_to_drop)

# Обновляем списки признаков после удаления
num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df_clean.select_dtypes(include=['object', 'string', 'category', 'bool']).columns.tolist()


# Заполнение пропусков:
# - Числовые заполняем медианой (устойчива к асимметрии и выбросам)
for col in num_cols:
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# - Категориальные заполняем отдельной категорией 'Unknown'
for col in cat_cols:
    if df_clean[col].isnull().any():
        df_clean[col] = df_clean[col].fillna('Unknown')

# Мягкая обработка выбросов (Винзоризация / Clipping по 1-му и 99-му перцентилю)
outliers_report = {}
for col in num_cols:
    if df_clean[col].nunique() > 10 and not col.startswith('time_'):
        q01 = df_clean[col].quantile(0.01)
        q99 = df_clean[col].quantile(0.99)
        outliers_count = ((df_clean[col] < q01) | (df_clean[col] > q99)).sum()
        if outliers_count > 0:
            outliers_report[col] = outliers_count
            df_clean[col] = np.clip(df_clean[col], q01, q99)

print(f"\nОбработаны выбросы (1%–99% квантили) в столбцах: {outliers_report}")

# 6. Feature Engineering и кодирование категорий (Пункты 3, 8)
# Извлекаем признаки из дат, если есть столбцы с датами
for col in cat_cols[:]:
    if 'date' in col.lower() or 'time' in col.lower():
        parsed_date = pd.to_datetime(df_clean[col], errors='coerce')
        if parsed_date.notnull().mean() > 0.8:
            df_clean[f'{col}_month'] = parsed_date.dt.month.fillna(0).astype(int)
            df_clean[f'{col}_dayofweek'] = parsed_date.dt.dayofweek.fillna(0).astype(int)
            df_clean = df_clean.drop(columns=[col])
            cat_cols.remove(col)
            print(f"Из даты '{col}' извлечены месяц и день недели.")

# Разделяем оставшиеся категориальные признаки по кардинальности
low_card_cols = [col for col in cat_cols if df_clean[col].nunique() <= 15]
high_card_cols = [col for col in cat_cols if df_clean[col].nunique() > 15]

# One-Hot Encoding для категорий с малым числом уникальных значений (<= 15)
if low_card_cols:
    df_clean = pd.get_dummies(df_clean, columns=low_card_cols, drop_first=True, dtype=int)
    print(f"Применён One-Hot Encoding к: {low_card_cols}")

# Frequency Encoding для категорий с большим числом значений (> 15)
for col in high_card_cols:
    freq_map = df_clean[col].value_counts(normalize=True)
    df_clean[f'{col}_freq'] = df_clean[col].map(freq_map)
    df_clean = df_clean.drop(columns=[col])
print(f"Применён Frequency Encoding к высококардинальным признакам: {high_card_cols}")

print(f"\nИтоговая размерность после обработки: {df_clean.shape}")
df_clean.to_csv("circulation_preprocessed.csv", index=False)
print("Файл 'circulation_preprocessed.csv' успешно сохранён!")
