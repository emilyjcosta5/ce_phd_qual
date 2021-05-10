import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

def plot_sparsity(df):
    mask = df['N'] == 20000
    pos = np.flatnonzero(mask)
    df = df.iloc[pos].reset_index()
    fig, ax = plt.subplots(1,1, figsize=(5,2.5))
    # Plot Naive
    mask = df['Matrix Representation'] == 'Naive'
    pos = np.flatnonzero(mask)
    sns.regplot(data=df.iloc[pos], x='Sparsity (%)', y='Performance (ms)', color='b', ax=ax, ci=0, order=3, label='Naive')
    # Plot CSR
    mask = df['Matrix Representation'] == 'CSR'
    pos = np.flatnonzero(mask)
    sns.regplot(data=df.iloc[pos], x='Sparsity (%)', y='Performance (ms)', color='g', ax=ax, ci=0, order=3, label='CSR')
    # Plot COO
    mask = df['Matrix Representation'] == 'COO'
    pos = np.flatnonzero(mask)
    sns.regplot(data=df.iloc[pos], x='Sparsity (%)', y='Performance (ms)', color='r', ax=ax, ci=0, order=3, label='COO')
    # Add plot aesthetics 
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_xlim(60,100)
    ax.set_ylim(0,14000)
    ax.set_yticks(range(0,15000,2000))
    ax.set_yticklabels(range(0,15000,2000))
    ax.set_xlabel('Matrix Sparsity')
    ax.set_ylabel('Runtime (ms)')
    vals = ax.get_xticks()
    ax.set_xticklabels(['{:,.0%}'.format(x/100) for x in vals])
    plt.tight_layout()
    fig.savefig('sparsity.pdf')
    plt.clf()
    plt.close()
    return 0

def plot_matrix_size(df):
    mask = df['Sparsity (%)'] == 90
    pos = np.flatnonzero(mask)
    df = df.iloc[pos].reset_index()
    fig, ax = plt.subplots(1,1, figsize=(5,2.5))
    # Plot Naive
    mask = df['Matrix Representation'] == 'Naive'
    pos = np.flatnonzero(mask)
    sns.regplot(data=df.iloc[pos], x='N', y='Performance (ms)', color='b', ax=ax, ci=0, order=3, label='Naive')
    # Plot CSR
    mask = df['Matrix Representation'] == 'CSR'
    pos = np.flatnonzero(mask)
    sns.regplot(data=df.iloc[pos], x='N', y='Performance (ms)', color='g', ax=ax, ci=0, order=3, label='CSR')
    # Plot COO
    mask = df['Matrix Representation'] == 'COO'
    pos = np.flatnonzero(mask)
    sns.regplot(data=df.iloc[pos], x='N', y='Performance (ms)', color='r', ax=ax, ci=0, order=3, label='COO')
    # Add plot aesthetics 
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_xlim(5000,30000)
    ax.set_ylim(0,30000)
    ax.set_yticks(range(5000,31000,5000))
    ax.set_yticklabels(range(5000,31000,5000))
    ax.set_xlabel('N, such that\nMatrix Size = N*N')
    ax.set_ylabel('Runtime (ms)')
    plt.tight_layout()
    fig.savefig('size.pdf')
    return 0

def plot_matrix_size_set_nnz(df, original_df):
    mask = df['Sparsity (#)'] == 1000
    pos = np.flatnonzero(mask)
    df = df.iloc[pos].reset_index()
    fig, ax = plt.subplots(1,1, figsize=(5,2.5))
    # Plot Naive
    mask = df['Matrix Representation'] == 'Naive'
    pos = np.flatnonzero(mask)
    sns.regplot(data=df.iloc[pos], x='N', y='Performance (ms)', color='b', ax=ax, ci=0, order=3, label='Naive')
    # Plot CSR
    mask = df['Matrix Representation'] == 'CSR'
    pos = np.flatnonzero(mask)
    sns.regplot(data=df.iloc[pos], x='N', y='Performance (ms)', color='g', ax=ax, ci=0, order=3, label='CSR')
    # Plot COO
    mask = df['Matrix Representation'] == 'COO'
    pos = np.flatnonzero(mask)
    sns.regplot(data=df.iloc[pos], x='N', y='Performance (ms)', color='r', ax=ax, ci=0, order=3, label='COO')
    # Add original
    mask = original_df['Sparsity (%)'] == 90
    pos = np.flatnonzero(mask)
    df = original_df.iloc[pos].reset_index()
    # Plot Naive
    mask = df['Matrix Representation'] == 'Naive'
    pos = np.flatnonzero(mask)
    sns.regplot(data=df.iloc[pos], x='N', y='Performance (ms)', color='b', ax=ax, ci=0, order=3, line_kws={'linestyle':'--', 'alpha':0.5})
    # Plot CSR
    mask = df['Matrix Representation'] == 'CSR'
    pos = np.flatnonzero(mask)
    sns.regplot(data=df.iloc[pos], x='N', y='Performance (ms)', color='g', ax=ax, ci=0, order=3, line_kws={'linestyle':'--', 'alpha':0.5})
    # Plot COO
    mask = df['Matrix Representation'] == 'COO'
    pos = np.flatnonzero(mask)
    sns.regplot(data=df.iloc[pos], x='N', y='Performance (ms)', color='r', ax=ax, ci=0, order=3, line_kws={'linestyle':'--', 'alpha':0.5})
    # Add plot aesthetics 
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_xlim(5000,30000)
    ax.set_ylim(0,30000)
    ax.set_yticks(range(5000,31000,5000))
    ax.set_yticklabels(range(5000,31000,5000))
    ax.set_xlabel('N, such that\nMatrix Size = N*N')
    ax.set_ylabel('Runtime (ms)')
    plt.tight_layout()
    fig.savefig('size_fixed_nnz.pdf')
    return 0

if __name__=='__main__':
    df0 = pd.read_csv('SpMV_performance_results.csv')
    plot_sparsity(df0)
    plot_matrix_size(df0)
    df1 = pd.read_csv('SpMV_performance_results1.csv')
    plot_matrix_size_set_nnz(df1, df0)