from pathlib import Path
import pandas as pd


# ============================================================
# 0. 路径设置
# ============================================================

# 当前文件是 src/01_data_cleaning.py
# parents[1] 就是项目根目录 financial-transaction-risk-monitoring
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW_DIR / "synthetic_fraud_dataset.csv"
OUTPUT_FILE = PROCESSED_DIR / "fraud_data_cleaned.csv"


# ============================================================
# 1. 读取数据
# ============================================================

df = pd.read_csv(RAW_FILE)

print("\n========== RAW DATA LOADED ==========")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())


# ============================================================
# 2. 基础数据检查
# ============================================================

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATED ROWS ==========")
print("Duplicated rows:", df.duplicated().sum())


# ============================================================
# 3. 统一列名格式
# ============================================================

# 把列名变成小写，并把空格替换成下划线
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
)

print("\n========== STANDARDIZED COLUMN NAMES ==========")
print(df.columns.tolist())


# ============================================================
# 4. 时间字段处理
# ============================================================

# 假设原始列名叫 Timestamp
# 标准化后会变成 timestamp
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # 拆出一些常用时间特征
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

print("\n========== TIMESTAMP CHECK ==========")
if "timestamp" in df.columns:
    print("Min timestamp:", df["timestamp"].min())
    print("Max timestamp:", df["timestamp"].max())
else:
    print("No timestamp column found.")


# ============================================================
# 5. 目标变量检查
# ============================================================

# 这里的标签列是 Fraud_Label
# 标准化后会变成 fraud_label
if "fraud_label" in df.columns:
    print("\n========== TARGET VARIABLE DISTRIBUTION ==========")
    print(df["fraud_label"].value_counts(dropna=False))
    print("\nFraud rate:")
    print(df["fraud_label"].value_counts(normalize=True, dropna=False))
else:
    print("\nWARNING: fraud_label column not found.")


# ============================================================
# 6. 数值字段简单检查
# ============================================================

numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

print("\n========== NUMERIC COLUMNS ==========")
print(numeric_cols)

print("\n========== NUMERIC SUMMARY ==========")
print(df[numeric_cols].describe().T)


# ============================================================
# 7. 类别字段简单检查
# ============================================================

categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

print("\n========== CATEGORICAL COLUMNS ==========")
print(categorical_cols)

for col in categorical_cols:
    print(f"\n--- Top values in {col} ---")
    print(df[col].value_counts(dropna=False).head(10))


# ============================================================
# 8. 缺失值简单处理
# ============================================================

# 这一步我们先做保守处理：
# - 数值列：如果有缺失，先用中位数填充
# - 类别列：如果有缺失，先填 Unknown

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ["float64", "int64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Unknown")

print("\n========== MISSING VALUES AFTER FILL ==========")
print(df.isnull().sum())


# ============================================================
# 9. 删重
# ============================================================

before_drop = df.shape[0]
df = df.drop_duplicates()
after_drop = df.shape[0]

print("\n========== DUPLICATE REMOVAL ==========")
print("Rows before:", before_drop)
print("Rows after :", after_drop)
print("Dropped    :", before_drop - after_drop)


# ============================================================
# 10. 保存清洗后的数据
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\n========== CLEANED DATA SAVED ==========")
print("Saved to:", OUTPUT_FILE)
print("Final shape:", df.shape)

print("\nData cleaning completed successfully.")