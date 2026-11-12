"""Engineer 1: explore the training data and create summary charts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import read_users, ensure_dirs, FIGURES, TABLES, TARGET, CLASSES


def main():
    ensure_dirs()
    df = read_users()
    print(f'Training records: {len(df):,}')
    print(f'Number of columns: {len(df.columns)}')

    # Save a summary of missing values for the report.
    missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isna().sum().to_numpy(),
        'missing_percent': (df.isna().mean() * 100).round(2).to_numpy(),
    })
    missing.to_csv(TABLES / 'missing_values.csv', index=False)

    # Count users in all 12 destination categories.
    counts = df[TARGET].value_counts().reindex(CLASSES, fill_value=0)
    shares = (counts / len(df) * 100).round(2)
    pd.DataFrame({'destination': counts.index,
                  'users': counts.values,
                  'percent': shares.values}).to_csv(
        TABLES / 'destination_counts.csv', index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_counts = counts.sort_values(ascending=True)
    ax.barh(sorted_counts.index, sorted_counts.values)
    ax.set(title='First booking destination in training data', xlabel='Users')
    for i, value in enumerate(sorted_counts.values):
        ax.text(value + 500, i, f'{value:,}', va='center', fontsize=8)
    ax.set_xlim(0, max(counts) * 1.15)
    fig.tight_layout()
    fig.savefig(FIGURES / '01_destinations.png', dpi=180)
    plt.close(fig)

    # Clean ages for the graph without removing the user from the dataset.
    ages = pd.to_numeric(df['age'], errors='coerce')
    ages = ages.where(ages.between(18, 100))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(ages.dropna(), bins=np.arange(18, 101, 4))
    ax.set(title='Age distribution (valid ages: 18–100)', xlabel='Age', ylabel='Users')
    fig.tight_layout()
    fig.savefig(FIGURES / '02_ages.png', dpi=180)
    plt.close(fig)

    # Display destination proportions within each signup method.
    methods = df['signup_method'].fillna('Missing')
    cross = pd.crosstab(methods, df[TARGET], normalize='index') * 100
    cross = cross.reindex(columns=CLASSES, fill_value=0)
    cross.to_csv(TABLES / 'signup_method_destination_percent.csv')
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cross[['NDF', 'US', 'other', 'FR']].plot.bar(stacked=True, ax=ax)
    ax.set(title='Selected destination shares within signup methods',
           xlabel='Signup method', ylabel='Percent of users')
    ax.legend(title='Destination', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(FIGURES / '03_signup_method.png', dpi=180)
    plt.close(fig)

    # Compare NDF rates in the eight most common first-device categories.
    top_devices = df['first_device_type'].value_counts().head(8).index
    device_rows = df[df['first_device_type'].isin(top_devices)].copy()
    device_ndf = device_rows.groupby('first_device_type')[TARGET].apply(
        lambda s: (s == 'NDF').mean() * 100
    ).sort_values()
    device_ndf.rename('ndf_percent').to_csv(TABLES / 'device_ndf_rates.csv')
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(device_ndf.index, device_ndf.values)
    ax.set(title='NDF rate by first device type (8 most common)',
           xlabel='Percentage with NDF', xlim=(0, 100))
    fig.tight_layout()
    fig.savefig(FIGURES / '04_device_ndf.png', dpi=180)
    plt.close(fig)

    # Compare two dominant destinations across account-creation months.
    dates = pd.to_datetime(df['date_account_created'], errors='coerce')
    monthly = df.assign(month=dates.dt.to_period('M').astype(str)).groupby('month')[TARGET]
    month_rates = monthly.apply(lambda s: (s == 'NDF').mean() * 100)
    month_counts = monthly.size()
    month_rates[month_counts >= 100].rename('ndf_percent').to_csv(
        TABLES / 'monthly_ndf_rates.csv')
    fig, ax = plt.subplots(figsize=(10, 4.7))
    filtered = month_rates[month_counts >= 100]
    ax.plot(range(len(filtered)), filtered.to_numpy())
    positions = np.arange(0, len(filtered), max(1, len(filtered) // 9))
    ax.set_xticks(positions, [filtered.index[i] for i in positions], rotation=35)
    ax.set(title='NDF share by account creation month (100+ users)',
           ylabel='Percent NDF', xlabel='Account creation month')
    fig.tight_layout()
    fig.savefig(FIGURES / '05_monthly_ndf.png', dpi=180)
    plt.close(fig)

    summary = {
        'train_users': len(df),
        'columns': len(df.columns),
        'destination_classes': df[TARGET].nunique(),
        'missing_age_percent': round(float(df['age'].isna().mean() * 100), 2),
        'invalid_age_count': int((~pd.to_numeric(df['age'], errors='coerce').between(18, 100)
                                  & df['age'].notna()).sum()),
        'ndf_percent': round(float((df[TARGET] == 'NDF').mean() * 100), 2),
    }
    pd.Series(summary, name='value').to_csv(TABLES / 'dataset_summary.csv')
    print('EDA charts and tables saved to results/.')
    for k, v in summary.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
