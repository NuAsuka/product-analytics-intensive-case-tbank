import pandas as pd
from pathlib import Path
from src.utils.helpers import DATA_RAW


def load_data() -> pd.DataFrame:
    xlsx_path = DATA_RAW / 'dataset-sots-seti.xlsx'
    return pd.read_excel(xlsx_path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['review_dttm'] = pd.to_datetime(df['review_dttm'])
    df['finish_dttm'] = pd.to_datetime(df['finish_dttm'])

    df['review_mark_num'] = pd.to_numeric(df['review_mark'], errors='coerce')
    df['has_numeric_mark'] = df['review_mark_num'].notna()

    df['response_hours'] = (
        df['finish_dttm'] - df['review_dttm']
    ).dt.total_seconds() / 3600

    df['text_len'] = df['review_text'].str.len()
    df['word_cnt'] = df['review_text'].str.split().str.len()

    df['month'] = df['review_dttm'].dt.month

    df['solution_flg_num'] = df['solution_flg'].map(
        {'проблема решена': 1, 'не указано': 0}
    ).fillna(0)

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['is_detractor'] = (
        df['review_mark_num'].between(1, 3) |
        ((df['review_emotion'] == 'Негативный') & ~df['has_numeric_mark'])
    ).astype(int)

    return df


def fill_missing_segments(df: pd.DataFrame) -> pd.DataFrame:
    fill_cols = [
        'segment_name', 'age_segment', 'gender_cd',
        'education_level_cd', 'marital_status_cd',
        'new_flg', 'influencer_flg', 'subscription_important_flg',
        'children_cnt',
    ]
    for col in fill_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(-1)
    return df
