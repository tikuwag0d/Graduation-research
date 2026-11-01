#  時系列分割・エンコーディング・保存
"""データの時系列分割、カテゴリ変数のエンコーディングを行う。

いずれもデータリークを防ぐことを最優先に設計している。
"""
import pandas as pd

def time_based_split(df: pd.DataFrame, train_end: str, valid_end: str):
    """開催日を基準に学習・検証・テストへ時系列分割する。

    【重要】競馬データは時系列性を持つため、ランダム分割は厳禁。
    「過去で学習し、未来で評価する」という実運用に近い形にする。

    Parameters
    ----------
    train_end : str
        学習データの終端日（例 "2023-12-31"）。これ以前が学習用。
    valid_end : str
        検証データの終端日。train_end より後〜valid_end が検証用、
        それ以降がテスト用。
    """
    # ------------------------------------------------------------
    # train_end 以前のレースを学習データとする（最も過去の期間）。
    # ------------------------------------------------------------
    train = df[df["race_date"] <= train_end]

    # ------------------------------------------------------------
    # train_end の翌日〜valid_end を検証データとする（中間の期間）。
    # ハイパーパラメータ調整や早期終了の判断に使う。
    # ------------------------------------------------------------
    valid = df[(df["race_date"] > train_end)
               & (df["race_date"] <= valid_end)]

    # ------------------------------------------------------------
    # valid_end より後をテストデータとする（最も未来の期間）。
    # 最終評価のみに使い、学習・調整には一切使わない。
    # ------------------------------------------------------------
    test = df[df["race_date"] > valid_end]

    return train, valid, test


def encode_categoricals(train: pd.DataFrame, valid: pd.DataFrame,
                        test: pd.DataFrame):
    """カーディナリティの低いカテゴリ変数をワンホット化する。

    性別・路面・馬場状態など、取りうる値が少ない変数が対象。
    """
    # ------------------------------------------------------------
    # ワンホット化する列（取りうる値の種類が少ないもの）。
    # 性別・路面・馬場状態・所属・競馬場・距離カテゴリなど。
    # ------------------------------------------------------------
    onehot_cols = ["sex", "surface", "track_condition", "affiliation",
                   "place", "dist_category"]

    # ------------------------------------------------------------
    # 各データを個別にワンホット化する。
    #   dummy_na=False : 欠損に対する列は作らない
    # 注意：train/valid/test を別々に変換すると、データごとに出現するカテゴリが異なり列数がずれる可能性がある。
    # → 次のreindexで列を揃えることで対処する。
    # ------------------------------------------------------------
    train = pd.get_dummies(train, columns=onehot_cols, dummy_na=False)
    valid = pd.get_dummies(valid, columns=onehot_cols, dummy_na=False)
    test = pd.get_dummies(test, columns=onehot_cols, dummy_na=False)

    # ------------------------------------------------------------
    # valid/test の列を train の列構成に合わせる（列揃え）。
    # reindex(columns=train.columns) : train に存在する列に統一
    # fill_value=0 : train にあって valid/test に無い列は0で埋める
    # これにより、学習時と推論時の特徴量の形が必ず一致する。
    # ------------------------------------------------------------
    valid = valid.reindex(columns=train.columns, fill_value=0)
    test = test.reindex(columns=train.columns, fill_value=0)

    return train, valid, test


def target_encode(train: pd.DataFrame, valid: pd.DataFrame,
                  test: pd.DataFrame, col: str,
                  target: str = "target_win", smoothing: int = 10):
    """高カーディナリティ変数（騎手名・調教師名など）を
    ターゲットエンコーディングする。

    【重要】統計量は学習データからのみ算出し、valid/test に適用する。
    valid/test の目的変数を使うとリークになるため絶対に使わない。
    """
    # ------------------------------------------------------------
    # 全体の平均勝率を計算する（スムージングの基準値・未知カテゴリ用）。
    # ------------------------------------------------------------
    global_mean = train[target].mean()

    # ------------------------------------------------------------
    # 学習データで、カテゴリごとの勝率（mean）と件数（count）を集計する。
    # 例：騎手ごとの勝率と騎乗回数。
    # ------------------------------------------------------------
    stats = train.groupby(col)[target].agg(["mean", "count"])

    # ------------------------------------------------------------
    # スムージング付きの平均を計算する。
    # 件数が少ないカテゴリ（例：数回しか乗っていない騎手）は、その勝率が偶然に左右されやすい。そこで全体平均へ寄せることで
    # 過学習を防ぐ。
    #   件数が多い → そのカテゴリの実勝率に近づく
    #   件数が少ない → 全体平均に近づく
    # ------------------------------------------------------------
    stats["enc"] = (
        (stats["mean"] * stats["count"] + global_mean * smoothing)
        / (stats["count"] + smoothing)
    )

    # ------------------------------------------------------------
    # カテゴリ名 → エンコード値 の対応表を辞書にする。
    # ------------------------------------------------------------
    mapping = stats["enc"].to_dict()

    # ------------------------------------------------------------
    # train/valid/test それぞれにエンコード値を付与する。
    #   .map(mapping)       : カテゴリをエンコード値に変換
    #   .fillna(global_mean): 学習データに無かった未知カテゴリは全体平均
    # ------------------------------------------------------------
    new_col = f"{col}_te"
    for d in (train, valid, test):
        d[new_col] = d[col].map(mapping).fillna(global_mean)

    return train, valid, test