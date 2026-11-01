#目的変数作成
"""Step 10: 予測対象となる目的変数（正解ラベル）を作成するモジュール。

異常記録を「含める／含めない」の比較実験に対応するため、
2つのパターンを引数で切り替えられるようにする。
"""
import pandas as pd


def make_target(df: pd.DataFrame, top_k: int = 3,
                abnormal_as_loss: bool = False) -> pd.DataFrame:
    """目的変数（単勝・複勝の的中ラベル）を作成する。

    Parameters
    ----------
    df : pd.DataFrame
        前段までの前処理済みデータ。
    top_k : int
        複勝的中とみなす着順の上限（通常は3着以内）。
    abnormal_as_loss : bool
        True （includeパターン）: 異常記録も「好走しなかった(=0)」とする
        False（excludeパターン）: 完走馬のみを対象とし、異常は後で除外する

    Returns
    -------
    pd.DataFrame
        target_win（単勝）・target_place（複勝）を追加したデータ。
    """
    # ------------------------------------------------------------
    # includeパターン：異常を「好走できなかった」として0扱いする。
    # 現実には出走しても中止・失格で結果を残せないことがあり、それも「勝てなかった」ケースとして学習させる考え方。
    #   勝ち(=1)の条件 : 1着 かつ 完走して着順が有効
    #   それ以外（異常含む）は自動的に0になる
    # ------------------------------------------------------------
    if abnormal_as_loss:
        df["target_win"] = (
            (df["finish_pos"] == 1) & (df["has_valid_finish"] == 1)
        ).astype(int)
        df["target_place"] = (
            (df["finish_pos"] <= top_k) & (df["has_valid_finish"] == 1)
        ).astype(int)

    # ------------------------------------------------------------
    # excludeパターン：完走馬のみを対象とする前提なので、単純に着順だけでラベルを作る（異常記録は後の工程で除外される）。
    #   target_win   : 1着なら1
    #   target_place : top_k着以内なら1（複勝相当）
    # ------------------------------------------------------------
    else:
        df["target_win"] = (df["finish_pos"] == 1).astype(int)
        df["target_place"] = (df["finish_pos"] <= top_k).astype(int)

    return df