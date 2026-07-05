import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_satisfaction_heatmap(cross: pd.DataFrame, title: str, fmt: str = 'd'):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(cross, annot=True, fmt=fmt, cmap='RdYlGn', center=None, ax=ax)
    ax.set_title(title)
    plt.tight_layout()


def plot_product_ranking(df: pd.DataFrame, metric_col: str, top_n: int = 10, min_count: int = 10):
    product_stats = df.groupby('product')[metric_col].agg(['mean', 'count'])
    ranking = (
        product_stats[product_stats['count'] >= min_count]['mean']
        .sort_values()
        .head(top_n)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ranking.plot(kind='barh', ax=ax, title=f'{top_n} худших продуктов по среднему {metric_col} (мин. {min_count} отзывов)')
    ax.set_xlabel(metric_col)
    plt.tight_layout()


def plot_negativity_share(df: pd.DataFrame, group_col: str, top_n: int = 10):
    neg_share = (
        df.groupby(group_col)['review_emotion']
        .apply(lambda x: (x == 'Негативный').mean())
        .sort_values(ascending=False)
        .head(top_n)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    neg_share.plot(kind='barh', ax=ax, title=f'Top-{top_n} по доле негатива ({group_col})', color='coral')
    ax.set_xlabel('Доля негативных отзывов')
    plt.tight_layout()


def plot_segment_comparison(df: pd.DataFrame, segment_col: str, value_col: str = 'review_mark_num'):
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby(segment_col)[value_col].median().sort_values().index
    sns.boxplot(data=df, x=segment_col, y=value_col, order=order, ax=ax)
    ax.set_title(f'{value_col} по сегменту {segment_col}')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()


def plot_temporal(df: pd.DataFrame, freq: str = 'ME'):
    monthly = (
        df.set_index('review_dttm')
        .resample(freq)
        .agg(
            reviews=('id_client', 'size'),
            avg_mark=('review_mark_num', 'mean'),
            neg_share=('review_emotion', lambda x: (x == 'Негативный').mean()),
        )
    )
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    monthly['reviews'].plot(ax=axes[0], marker='o', title='Количество отзывов')
    axes[0].set_ylabel('Отзывов')
    monthly['avg_mark'].plot(ax=axes[1], marker='o', color='orange', title='Средняя оценка')
    axes[1].set_ylabel('Средний review_mark')
    monthly['neg_share'].plot(ax=axes[2], marker='o', color='red', title='Доля негативных отзывов')
    axes[2].set_ylabel('Доля негатива')
    plt.tight_layout()


def plot_solution_impact(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    plot_df = df.dropna(subset=['review_mark_num']).copy()
    sol_effect = plot_df.groupby('solution_flg_num')['review_mark_num'].mean()
    sol_effect = sol_effect.reindex([0.0, 1.0], fill_value=0)

    sol_effect.plot(kind='bar', ax=axes[0], title='Средняя оценка по решённости', color=['coral', 'green'], legend=False)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Не решено', 'Решено'], rotation=0)
    axes[0].set_ylabel('Средний review_mark')
    axes[0].set_ylim(0, 5)

    csat_df = df.dropna(subset=['csat_score']).copy()
    sol_csat = csat_df.groupby('solution_flg_num')['csat_score'].mean()
    sol_csat = sol_csat.reindex([0.0, 1.0], fill_value=0)

    sol_csat.plot(kind='bar', ax=axes[1], title='Средний CSAT по решённости', color=['coral', 'green'], legend=False)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Не решено', 'Решено'], rotation=0)
    axes[1].set_ylabel('Средний csat_score')
    axes[1].set_ylim(0, 5)

    plt.tight_layout()


def plot_reason_heatmap(df: pd.DataFrame, group_col: str, top_n: int = 10):
    top_groups = df[group_col].value_counts().head(top_n).index
    cross = (
        df[df[group_col].isin(top_groups)]
        .groupby([group_col, 'review_emotion'])
        .size()
        .unstack(fill_value=0)
    )
    cross = cross.div(cross.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
    sns.heatmap(cross, annot=True, fmt='.0%', cmap='RdYlGn', ax=ax)
    ax.set_title(f'Доля тональности по топ-{top_n} значений {group_col}')
    plt.tight_layout()
