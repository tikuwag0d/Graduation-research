# File: check_data.py
import os
import pandas as pd

# ------------------------------------------------------------
# このスクリプト自身が置かれているフォルダの絶対パスを取得する。
# これを基準にすることで、どの場所から実行しても
# パスがずれず、確実にファイルを見つけられる。
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# parquetファイルの保存先フォルダを指定する。
# main.py で "output/xxx.parquet" に保存した場合は "output"、
# main.py と同じ階層に保存した場合は "" （空文字）にする。
# ------------------------------------------------------------
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # outフォルダに保存した場合
# OUTPUT_DIR = BASE_DIR  # 同じ階層に保存した場合はこちらを有効にする

names = [
    "train_exclude", "valid_exclude", "test_exclude",
    "train_include", "valid_include", "test_include",
]

for name in names:
    # --- 各ファイルの絶対パスを組み立てる ---
    path = os.path.join(OUTPUT_DIR, f"{name}.parquet")

    # --- ファイルが存在するか確認してから読み込む ---
    if not os.path.exists(path):
        print(f"[見つかりません] {path}")
        continue

    df = pd.read_parquet(path)
    print(f"{name}: {df.shape[0]:,} 件 × {df.shape[1]} 項目, "
          f"{df['race_date'].min().date()} 〜 {df['race_date'].max().date()}")