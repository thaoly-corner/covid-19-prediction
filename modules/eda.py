import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
from modules.config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
import numpy as np

def _apply_mapping(df, column, hue, mapping):
    """Helper function to apply categorical mapping safely."""
    temp = df.copy()
    if mapping:
        # Check for nested or flat mapping
        if isinstance(mapping, dict) and column in mapping:
            temp[column] = temp[column].astype(object).replace(mapping[column])
        elif not isinstance(list(mapping.values())[0], dict):
            temp[column] = temp[column].astype(object).replace(mapping)
            
        if hue:
            if isinstance(mapping, dict) and hue in mapping:
                temp[hue] = temp[hue].astype(object).replace(mapping[hue])
            elif not isinstance(list(mapping.values())[0], dict):
                temp[hue] = temp[hue].astype(object).replace(mapping)
    return temp

def plot_categorical_distribution(df, column, title, hue=None, mapping=None, palette='Blues'):
    """
    Unified function for categorical analysis.
    - If hue is None: Plots general distribution (% of total).
    - If hue is provided: Plots relationship (% within each X-category).
    """
    # 1. Apply mapping helper
    temp = _apply_mapping(df, column, hue, mapping)
    total_samples = len(temp)

    # 2. Initialize Plot
    plt.figure(figsize=(12, 7))
    ax = sns.countplot(data=temp, x=column, hue=hue, palette=palette)
    
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.ylabel('Number of Patients')
    plt.xlabel(column.replace('_', ' ').title())

    # 3. Dynamic Bar Labeling
    for container in ax.containers:
        raw_values = container.datavalues
        labels = []
        for i, v in enumerate(raw_values):
            if v > 0:
                if hue:
                    # Relationship logic: % relative to the specific X-axis group
                    curr_x_label = ax.get_xticklabels()[i % len(ax.get_xticks())].get_text()
                    category_total = temp[temp[column] == curr_x_label].shape[0]
                    percentage = (v / category_total) * 100
                else:
                    # Distribution logic: % relative to the entire population
                    percentage = (v / total_samples) * 100
                
                labels.append(f'{int(v):,} ({percentage:.1f}%)')
            else:
                labels.append('')
        
        ax.bar_label(container, labels=labels, label_type='edge', padding=3, fontsize=9)

    # Adjust Y-limit so labels don't hit the top of the box
    plt.ylim(0, ax.get_ylim()[1] * 1.15)
    
    if hue:
        plt.legend(title=hue.replace('_', ' ').title(), loc='best', frameon=True)
    
    plt.tight_layout()
    plt.show()

    # 4. Summary Tables
    if hue:
        print(f"\n[SUMMARY TABLE: `{column}` vs `{hue}`]")
        count_table = pd.crosstab(temp[column], temp[hue], margins=True, margins_name="Total")
        perc_table = (pd.crosstab(temp[column], temp[hue], normalize='index') * 100).round(2).astype(str) + '%'
        print("\n1. Raw Counts:\n", count_table)
        print("\n2. Row-wise Percentage (Risk Analysis):\n", perc_table)
    else:
        print(f"\n[SUMMARY TABLE: `{column}`]")
        counts = temp[column].value_counts()
        pcts = (temp[column].value_counts(normalize=True) * 100).round(2).astype(str) + '%'
        summary_df = pd.DataFrame({'Count': counts, 'Percentage (%)': pcts})
        print(summary_df)
    
    print("-" * 60 + "\n")


def plot_numerical_distribution(df, column, title, hue=None, mapping=None, palette='Blues'):
    """
    Unified Numerical Plot with Dynamic Layouts:
    - If hue is None: Standard Top-Down (Horizontal Boxplot + Histogram).
    - If hue is provided: Side-by-Side (Vertical Boxplots + KDE Comparison).
    """
    # 1. Apply mapping helper
    temp = _apply_mapping(df, column, hue, mapping)

    if hue is None:
        # --- CASE 1: STANDARD TOP-DOWN LAYOUT ---
        fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
                                              gridspec_kw={"height_ratios": (.15, .85)}, 
                                              figsize=(12, 8))
        
        sns.boxplot(data=temp, x=column, ax=ax_box, palette=palette)
        ax_box.set(xlabel='')
        ax_box.set_title(title, fontsize=15, fontweight='bold', pad=20)

        sns.histplot(data=temp, x=column, kde=True, palette=palette, ax=ax_hist)
        
        mean_val = temp[column].mean()
        ax_hist.axvline(mean_val, color='black', linestyle='--', label=f'Mean: {mean_val:.1f}')
        
        ax_hist.set_xlabel(column.replace('_', ' ').title())
        ax_hist.set_ylabel('Number of Patients')
        ax_hist.legend()

    else:
        # --- CASE 2: SIDE-BY-SIDE COMPARISON LAYOUT ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 2.5]})
       
        sns.boxplot(data=temp, x=hue, y=column, ax=axes[0], palette=palette)
        axes[0].set_title('Quartile Analysis', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(hue.replace('_', ' ').title())
        axes[0].set_ylabel(column.replace('_', ' ').title())

        sns.kdeplot(data=temp, x=column, hue=hue, fill=True, ax=axes[1], palette=palette, common_norm=False)
        
        mean_val = temp[column].mean()
        axes[1].axvline(mean_val, color='black', linestyle='--', alpha=0.6, label=f'Overall Mean: {mean_val:.1f}')
        
        axes[1].set_title('Density Distribution (KDE)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(column.replace('_', ' ').title())
        axes[1].legend(title=hue.replace('_', ' ').title())

        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.show()

    # 3. Summary Statistics
    print(f"\n[SUMMARY STATISTICS: `{column.upper()}`]")
    if hue:
        stats = temp.groupby(hue)[column].describe().round(2)
    else:
        stats = temp[column].describe().to_frame().T.round(2)
    print(stats)
    print("-" * 60 + "\n")

def plot_correlation(df, columns, title='Correlation Matrix', palette='RdBu_r'):
    """
    Plots a Heatmap to visualize the correlation between numerical features.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of numerical columns to include in the correlation.
        title (str): Title of the plot.
    """
    # 1. Calculate the correlation matrix
    # Note: Only numerical data can be correlated
    corr_matrix = df[columns].corr()

    # 2. Setup the plot
    plt.figure(figsize=(10, 8))
    
    # 3. Draw the Heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap=palette, 
                fmt=".2f", 
                vmin=-1, 
                vmax=1, 
                center=0,
                linewidths=.5,
                cbar_kws={"shrink": .8})
    
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.xticks(rotation=45)
    plt.show()

    # 4. Display the raw coefficients for precise reference
    print(f"\n[CORRELATION COEFFICIENTS: {', '.join(columns)}]")
    print(corr_matrix)
    print("-" * 60 + "\n")

def cramers_v(x, y):
    """Calculates Cramér's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    # Check division by zero
    min_dim = min((kcorr-1), (rcorr-1))
    if min_dim == 0:
        return 0.0
    return np.sqrt(phi2corr / min_dim)

def plot_categorical_heatmap(df, categorical_cols, title="Categorical Correlation (Cramér's V)", palette='Blues', threshold=0.7):
    """
    Calculates the Cramér's V statistic for categorical-categorical association.
    """
    # 1. Filter columns that exist in the DataFrame
    cols = [c for c in categorical_cols if c in df.columns]
    
    # 2. Create an empty correlation matrix
    corr_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    
    # 3. Calculate Cramér's V for each pair of variables
    for col1 in cols:
        for col2 in cols:
            corr_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
            
    # 4. Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap=palette,
                fmt=".2f", 
                vmin=0, vmax=1, 
                linewidths=.5,
                cbar_kws={"shrink": .8})
    
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.show()

    # 5. Extract highly correlated pairs (ignoring the diagonal and lower triangle to avoid duplicates)
    high_corr_pairs = []
    
    # np.triu with k=1 gets the upper triangle above the main diagonal
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    for col in upper_triangle.columns:
        for row in upper_triangle.index:
            score = upper_triangle.loc[row, col]
            if pd.notna(score) and score >= threshold:
                high_corr_pairs.append((row, col, score))
                
    # Sort pairs by correlation score (highest first)
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)


    # 6. Print Summary
    print(f"\n[CRAMÉR'S V STATISTICAL MATRIX]")
    print("-" * 60)
    
    print(f"\n[MULTICOLLINEARITY ALERT (Threshold >= {threshold})]")
    if high_corr_pairs:
        for feat1, feat2, score in high_corr_pairs:
            print(f" {feat1} & {feat2} : {score:.3f}")
    print("-" * 60 + "\n")

def plot_categorical_importance(df, categorical_cols, target='is_dead', title="Categorical Feature Importance", palette='magma'):
    """
    Plots the importance of categorical features based on Cramér's V with respect to the target variable.
    """
    # 1. Calculate importance scores for each feature
    importance_scores = []
    cols_to_check = [c for c in categorical_cols if c in df.columns and c != target]
    
    for col in cols_to_check:
        score = cramers_v(df[col], df[target])
        importance_scores.append({'Feature': col.replace('_', ' ').title(), 'Score': score})
        
    # 2. Create DataFrame and sort from high to low
    importance_df = pd.DataFrame(importance_scores).sort_values(by='Score', ascending=False)
    
    # 3. Plot bar chart
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x='Score', y='Feature', data=importance_df, palette=palette)
    
    # 4. Add value labels on each bar
    for p in ax.patches:
        width = p.get_width()
        ax.text(width + 0.01, p.get_y() + p.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')
        
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(f"Importance Score (Cramér's V vs {target})", fontsize=12)
    plt.ylabel('')
    
    plt.xlim(0, importance_df['Score'].max() * 1.2)
    sns.despine()
    plt.tight_layout()
    plt.show()

    # Print the importance ranking in a clear table format
    print(f"\n[FEATURE IMPORTANCE RANKING vs `{target.upper()}`]")
    print(importance_df.to_string(index=False))
    print("-" * 60 + "\n")