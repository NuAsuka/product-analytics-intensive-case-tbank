import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from itertools import combinations


def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))


def pairwise_cramers(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    results = []
    for c1, c2 in combinations(cat_cols, 2):
        ct = pd.crosstab(df[c1], df[c2])
        v = cramers_v(ct)
        results.append({'col1': c1, 'col2': c2, 'cramers_v': round(v, 3)})
    return pd.DataFrame(results).sort_values('cramers_v', ascending=False)


def neg_share_by_group(df: pd.DataFrame, group_col: str, min_count: int = 30) -> pd.DataFrame:
    return (
        df.groupby(group_col)
        .agg(
            total=('review_emotion', 'size'),
            neg_count=('review_emotion', lambda x: (x == 'Негативный').sum()),
            neg_share=('review_emotion', lambda x: (x == 'Негативный').mean()),
            avg_mark=('review_mark_num', 'mean'),
        )
        .query(f'total >= {min_count}')
        .sort_values('neg_count', ascending=False)
        .reset_index()
    )
