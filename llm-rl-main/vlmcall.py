import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

neighbors = ['3 neighbor', '5 neighbor', '7 neighbor']

data = {
    'Swimmer': {
        'means': [215.35, 309.36, 275.64],
        'stds':  [90.08,  51.83,  87.53],
    },
    'Mountain Car (discrete)': {
        'means': [-140.93, -112.01, -119.71],
        'stds':  [36.88,   2.85,    20.23],
    },
}

palette = ['#1D9E75', '#D85A30', '#7F77DD']

sns.set_theme(style='whitegrid', font_scale=1.0)

for title, vals in data.items():
    df = pd.DataFrame({
        'neighbor': neighbors,
        'mean':     vals['means'],
        'std':      vals['stds'],
    })

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.barplot(
        data=df,
        x='neighbor',
        y='mean',
        hue='neighbor',
        palette=palette,
        legend=False,
        ax=ax,
    )

    ax.errorbar(
        x=np.arange(len(neighbors)),
        y=vals['means'],
        yerr=vals['stds'],
        fmt='none',
        color='#444441',
        capsize=5,
        elinewidth=1.2,
        capthick=1.2,
    )

    ax.set_title(title, fontsize=11, style='italic', fontweight='normal')
    ax.set_xlabel('')
    ax.set_ylabel('Return', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_locator(ticker.AutoLocator())

    filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.png'
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()