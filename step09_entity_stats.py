#騎手・調教師集計
"""Step 9: 騎手・調教師の直近成績を特徴量化するモジュール。

馬の過去成績と同様に、当該レースを含めず過去データのみで集計する（データリーク防止）。
"""
import pandas as pd

def add_entity_stats(df: pd.DataFrame, n_recent: int = 50) -> pd.DataFrame:
    """騎手・調教師の直近勝率を特徴量化する。

    Parameters
    ----------
    df : pd.DataFrame
        前段までの前処理済みデータ。
    n_recent : int
        直近何走分で勝率を計算するか。

    Returns
    -------
    pd.DataFrame
        騎手・調教師の直近勝率を追加したデータ。
    """
    # ------------------------------------------------------------
    # 全体を時系列順に並べ替える。
    # 騎手・調教師は複数の馬に関わるため、馬単位ではなくレース全体の時間順（日付→レース番号→馬番）で並べて集計する。
    # ------------------------------------------------------------
    df = df.sort_values(
        ["race_date", "race_no", "horse_no"]
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # 勝利フラグを作る。完走して1着だった場合のみ1、それ以外は0。
    #   .where(has_valid_finish==1) : 異常記録は集計対象外（NaN）にする
    # ------------------------------------------------------------
    win = (
        (df["finish_pos"] == 1)
        .where(df["has_valid_finish"] == 1)
        .astype(float)
    )

    # ------------------------------------------------------------
    # 騎手の直近n走の勝率を計算する。
    #   groupby("jockey_code") : 騎手ごとに集計
    #   shift(1)               : 当該レースを除外（リーク防止）
    #   rolling(n_recent)      : 直近n走の移動平均で勝率を算出
    #   min_periods=5          : 最低5走の実績が無いと計算しない（少試行での不安定な勝率を避ける）
    # 「今の調子（直近の勝率）」を表す特徴量になる。
    # ------------------------------------------------------------
    df["jockey_recent_winrate"] = (
        win.groupby(df["jockey_code"]).apply(
            lambda s: s.shift(1).rolling(n_recent, min_periods=5).mean()
        ).reset_index(level=0, drop=True)
    )

    # ------------------------------------------------------------
    # 調教師の直近n走の勝率も同様に計算する。
    # 騎手と同じく、当該レースを除いた過去データのみを使う。
    # ------------------------------------------------------------
    df["trainer_recent_winrate"] = (
        win.groupby(df["trainer_code"]).apply(
            lambda s: s.shift(1).rolling(n_recent, min_periods=5).mean()
        ).reset_index(level=0, drop=True)
    )

    return df