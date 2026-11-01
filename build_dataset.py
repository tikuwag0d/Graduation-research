#2つのデータセットの生成
"""異常コードを「含める／含めない」の2パターンのデータセットを構築する。"""
import pandas as pd
from step09_entity_stats import add_entity_stats
from step10_make_target import make_target
from step08_past_performance import add_past_performance

def build_dataset(df_base: pd.DataFrame, mode: str) -> pd.DataFrame:
    """mode に応じてデータセットを構築する。

    この関数は、Step1〜7まで共通処理したデータ（df_base）を受け取り、
    異常コードの扱いだけが異なる2種類のデータセットを作り分ける。
    比較実験の中核となる関数。

    Parameters
    ----------
    df_base : pd.DataFrame
        Step1〜7（読み込み〜欠損処理）まで済ませた共通データ。
    mode : str
        "exclude" : 異常記録を学習対象から除外（完走馬のみの理想的設定）
        "include" : 異常記録も含める（現実に近い設定）
    """
    # ------------------------------------------------------------
    # 元データを破壊しないようコピーして処理する。
    # df_base は include/exclude の両方で使い回すため、片方の処理が元データを変更しないようにする必要がある。
    # ------------------------------------------------------------
    df = df_base.copy()

    # ============================================================
    # excludeパターン：完走馬のみを対象とする構成
    # 「無事に走り切った馬だけ」で学習する理想的・単純な設定。
    # ============================================================
    if mode == "exclude":
        # --- 過去成績：異常履歴は特徴量化しない ---
        # 完走前提なので、過去の中止率・取消率は使わない。
        df = add_past_performance(df, n_recent=3,
                                  include_abnormal_history=False)

        # --- 騎手・調教師の直近勝率を付与 ---
        df = add_entity_stats(df, n_recent=50)

        # --- 目的変数：完走馬の着順のみで作成 ---
        # abnormal_as_loss=False で単純に着順からラベルを作る。
        df = make_target(df, top_k=3, abnormal_as_loss=False)

        # --- 完走した有効記録だけを残す ---
        # 特徴量生成（過去集計）が終わった後に除外するのがポイント。
        # 先に除外すると過去成績の集計対象から異常記録が抜けてしまい、include側と集計条件がずれてしまうため、ここで初めて絞る。
        df = df[df["has_valid_finish"] == 1].reset_index(drop=True)

    # ============================================================
    # includeパターン：異常記録も含める構成
    # 出走取消・中止・失格も起こる現実に近い設定で、
    # 異常＝「好走できなかった」として学習させる。
    # ============================================================
    elif mode == "include":
        # --- 過去成績：異常履歴も特徴量化する ---
        # 「過去に中止・取消が多い馬」という情報を使えるようにする。
        # これは走る前に分かる情報なのでリークにはならない。
        df = add_past_performance(df, n_recent=3,
                                  include_abnormal_history=True)

        # --- 騎手・調教師の直近勝率を付与 ---
        df = add_entity_stats(df, n_recent=50)

        # --- 目的変数：異常も「非好走(0)」として作成 ---
        # abnormal_as_loss=True で、中止・失格も勝てなかった扱いにする。
        df = make_target(df, top_k=3, abnormal_as_loss=True)

        # --- 実際に走った馬のみ残す（出走取消・発走除外は除く）---
        # 取消・除外はそもそもゲートを出ておらず馬券も返還されるため、
        # 分析対象から外す。残すのは「走った馬（完走＋中止＋失格）」。
        df = df[df["did_run"] == 1].reset_index(drop=True)

    # ------------------------------------------------------------
    # 想定外のmode指定はバグの温床になるため、明示的にエラーを出す。
    # ------------------------------------------------------------
    else:
        raise ValueError(f"unknown mode: {mode}")

    return df