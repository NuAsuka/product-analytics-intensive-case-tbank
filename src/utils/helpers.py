import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
REPORTS_FIGURES = PROJECT_ROOT / 'reports' / 'figures'


def configure_plots():
    sns.set_theme(style='whitegrid')
    plt.rcParams.update({
        'figure.dpi': 100,
        'figure.figsize': (10, 5),
        'font.size': 11,
    })


def save_figure(name: str, dpi: int = 150):
    REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
    path = REPORTS_FIGURES / name
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f'Saved: {path}')


def top_n_encoder(series: str, n: int = 10, label: str = 'Other'):
    counts = series.value_counts()
    top = counts.index[:n]
    return series.where(series.isin(top), other=label)
