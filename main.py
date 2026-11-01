# File: main.py
"""前処理の全工程を実行し、include/exclude の2系統のデータを出力する。"""
import os
import numpy as np

from preprocess import (
    load_csv, clean_dtypes, add_track_features,
    classify_abnormal, filter_records,
    add_race_features, add_class_features, handle_missing,
)
from build_dataset import build_dataset
from split_encode import time_based_split, encode_categoricals, target_encode


# ------------------------------------------------------------
# スクリプト自身の場所を基準に、入出力フォルダの絶対パスを決める。
# これにより、どこから実行してもパスがずれない。
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def main():
    np.random.seed(42)

    # --- output フォルダが無ければ作成する ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 【共通前処理】Step1〜7
    df = load_csv(os.path.join(BASE_DIR, "861京.CSV"))  # ←CSVの場所に合わせて調整
    df = clean_dtypes(df)
    df = add_track_features(df)
    df = classify_abnormal(df)
    df = filter_records(df)
    df = add_race_features(df)
    df = add_class_features(df)
    df = handle_missing(df)

    # 【2パターンのデータセットを生成・保存】
    for mode in ["exclude", "include"]:
        ds = build_dataset(df, mode=mode)

        train, valid, test = time_based_split(
            ds, train_end="1986-1-8", valid_end="1986-1-12"
        )
        train, valid, test = encode_categoricals(train, valid, test)
        for col in ["jockey_name", "trainer_name"]:
            train, valid, test = target_encode(train, valid, test, col)

        # --- output フォルダに保存（絶対パスで指定）---
        train.to_parquet(os.path.join(OUTPUT_DIR, f"train_{mode}.parquet"))
        valid.to_parquet(os.path.join(OUTPUT_DIR, f"valid_{mode}.parquet"))
        test.to_parquet(os.path.join(OUTPUT_DIR, f"test_{mode}.parquet"))

        print(f"[{mode}] train:{len(train)}, "
              f"valid:{len(valid)}, test:{len(test)}")


if __name__ == "__main__":
    main()