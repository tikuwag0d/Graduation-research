# File: step08_past_performance.py
"""Step 8: 馬ごとの過去成績特徴量を生成するモジュール。

【最重要】当該レースの結果を一切含めず、必ず「それより前」の
データだけで集計する（データリーク防止の要となる処理）。
"""
import numpy as np
import pandas as pd


def add_past_performance(df: pd.DataFrame, n_recent: int = 3,
                         include_abnormal_history: bool = False
                         ) -> pd.DataFrame:
    """馬ごとの過去成績特徴量を生成する。

    Parameters
    ----------
    df : pd.DataFrame
        前段までの前処理済みデータ。
    n_recent : int
        直近何走分を集計対象とするか。
    include_abnormal_history : bool
        True の場合、過去の異常発生率（競走中止率・取消率）も特徴量化する。
        （当該レースの異常ではなく過去の履歴なのでリークにはならない）

    Returns
    -------
    pd.DataFrame
        過去成績特徴量を追加したデータ。
    """
    # ------------------------------------------------------------
    # 馬ごと・時系列順に並べ替える。
    # 過去成績を「時間の流れに沿って」集計するため、
    # ped_id（馬）→ race_date（開催日）→ race_no（レース番号）
    # の順でソートし、同じ馬のレースが古い順に並ぶようにする。
    # ------------------------------------------------------------
    df = df.sort_values(
        ["ped_id", "race_date", "race_no"]
    ).reset_index(drop=True)
    grp = df.groupby("ped_id")

    # ------------------------------------------------------------
    # 集計に使う着順は「完走して着順が有効なもの」に限定する。
    # 異常（中止・失格等）の行は着順をNaNにして集計対象から外す。
    #   .where(条件) : 条件を満たさない値をNaNに置き換える。
    # これにより過去成績が異常記録で歪むのを防ぐ。
    # ------------------------------------------------------------
    valid_finish = df["finish_pos"].where(df["has_valid_finish"] == 1)

    # ------------------------------------------------------------
    # 直近n走の平均着順を計算する。リーク防止の核心部分。
    #   shift(1)          : 1つ前にずらし「当該レース」を除外
    #   rolling(n_recent) : 直近n走の範囲で移動平均をとる
    #   min_periods=1     : 過去が1走でもあれば計算する
    # ------------------------------------------------------------
    df["past_avg_finish"] = (
        valid_finish.groupby(df["ped_id"]).apply(
            lambda s: s.shift(1).rolling(n_recent, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
    )

    # ------------------------------------------------------------
    # 通算出走数（キャリア数）を数える。
    #   cumcount() : グループ内で0,1,2,…と連番を振る。
    # 「当該レースより前に何回走ったか」を表す（0なら初出走）。
    # ------------------------------------------------------------
    df["career_count"] = grp.cumcount()

    # ------------------------------------------------------------
    # 過去の勝利数を計算する。
    #   is_win   : 完走して1着だった行を1、それ以外を0とする
    #   cumsum() : 累積和（これだと当該レースまで含む）
    #   - is_win : 当該レース分を引いて「当該より前」までにする。
    # ------------------------------------------------------------
    is_win = (valid_finish == 1).astype(float)
    df["past_win_count"] = (
        is_win.groupby(df["ped_id"]).cumsum() - is_win.fillna(0)
    )

    # ------------------------------------------------------------
    # 過去の勝率＝過去勝利数 ÷ 過去出走数。
    #   replace(0, np.nan) : 出走数0での0除算を避ける（初出走はNaN）
    # ------------------------------------------------------------
    df["past_win_rate"] = (
        df["past_win_count"] / df["career_count"].replace(0, np.nan)
    )

    # ------------------------------------------------------------
    # 前走からの経過日数（レース間隔）を計算する。
    #   diff()   : 同じ馬の前のレース日との差分をとる
    #   .dt.days : 差分（Timedelta）を日数に変換
    # 休養明けや連闘など、間隔はパフォーマンスに影響する重要な情報。
    # ------------------------------------------------------------
    df["days_since_last"] = grp["race_date"].diff().dt.days

    # ------------------------------------------------------------
    # 初出走（デビュー戦）かどうかのフラグ。
    # career_count==0 は過去に走った記録が無い＝デビュー戦を意味する。
    # ------------------------------------------------------------
    df["is_debut"] = df["career_count"].eq(0).astype(int)

    # ------------------------------------------------------------
    # 過去レースのPCI（ペース指数）の平均を特徴量化する。
    # PCIはその馬の脚質・展開傾向（前傾/後傾ラップでの走り）を表す。
    #   valid_pci        : 完走記録のPCIのみを対象（異常記録は除外）
    #   shift(1)         : 当該レースを除外（当該PCIはレース結果なので禁止）
    #   rolling(n_recent): 直近n走の移動平均
    # 「この馬は後半型（PCI高め）／前半型（PCI低め）」という
    # 展開適性を、走る前から分かる過去情報として利用できる。
    # ------------------------------------------------------------
    valid_pci = df["pci"].where(df["has_valid_finish"] == 1)
    df["past_avg_pci"] = (
        valid_pci.groupby(df["ped_id"]).apply(
            lambda s: s.shift(1).rolling(n_recent, min_periods=1).mean()
        ).reset_index(level=0, drop=True)
    )

    # ------------------------------------------------------------
    # 【includeパターン専用】過去の異常発生履歴を特徴量化する。
    # 「この馬は過去に競走中止や取消が多い」という情報は、
    # 走る前から分かるためリークではない。
    # ------------------------------------------------------------
    if include_abnormal_history:
        # --- 過去の競走中止率（当該レースを除く）---
        dnf = df["is_dnf"].astype(float)
        df["past_dnf_count"] = (
            dnf.groupby(df["ped_id"]).cumsum() - dnf
        )
        df["past_dnf_rate"] = (
            df["past_dnf_count"] / df["career_count"].replace(0, np.nan)
        ).fillna(0)

        # --- 過去の出走取消率（当該レースを除く）---
        scr = df["is_scratched"].astype(float)
        df["past_scratch_count"] = (
            scr.groupby(df["ped_id"]).cumsum() - scr
        )
        df["past_scratch_rate"] = (
            df["past_scratch_count"] / df["career_count"].replace(0, np.nan)
        ).fillna(0)

    return df