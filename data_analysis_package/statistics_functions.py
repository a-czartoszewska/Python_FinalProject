""" A module for data analysis.

Functions:
    - statistics_table: Generates a summary table of min, mean, median, and max values.
    - barplots: Plots barplots for selected columns.
    - normalised_barplot: Plots normalised barplot to compare variables.
    - corr_plot: Plots a correlation plot of selected columns.
    - corr_test: Performs a Pearson correlation test and saves the result.
"""

import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import numpy as np

def statistics_table(df, cols=None):
    """ Function returning a table with basic statistics: min, average, median (50%), max.

    Args:
        df (pd.DataFrame): Dataframe to summarize.
        cols (list[str], optional): Columns to include. If None, includes all.

    Returns:
        pd.DataFrame: Table with basic statistics.

    """

    if df.empty:
        raise ValueError('The dataframe is empty. Please provide a valid dataframe.')

    if cols is None:
        cols = df.columns

    df = df[cols]
    stat_table = df.describe()
    stat_table = stat_table.round(2)
    stat_table.drop(labels=['count', '25%', '75%'], inplace=True)

    return stat_table

def barplots(df, x=None, cols=None,
             output_folder='', filename='barplots.png', save=True,
             show=True):
    """ Function creating a set of barplots.

    Args:
        df (pd.DataFrame): Input data.
        x (str, optional): Column to use for x-axis. If None, index is used.
        cols (List[str], optional): Columns to plot. If None, plots all.
        save (bool): Whether to save the plot to file. Default is True.
        output_folder (str): Folder path for saving the plot. Default is ''.
        filename (str): Name of the output file. Default is 'barplots.png'.
        show (bool): Whether to display the plot. Default is True.

    Returns:
        None

    Raises:
        ValueError: If some of the cols are not in df.columns.
    """

    if cols is None:
        cols = df.columns
    if x is None:
        x = df.index
        xlabel = df.index.name
    else:
        xlabel = x
        x = df[x]

    if len(x) > 30:
        raise ValueError(f'There are more than 30 values in the {xlabel} column. The plot would not be meaningful')

    # creating the plot
    _, axs = plt.subplots(len(cols), 1, figsize=(8, 12), sharex=True)

    # drawing each barplot
    if len(cols) == 1:
        axs = [axs]

    i = 0
    for col in cols:
        if col not in df.columns:
            raise ValueError('Some of the columns not in dataframe.')
        axs[i].bar(x, df[col], color=sns.color_palette()[i])
        axs[i].set_ylabel(col, fontweight='bold')
        i = i + 1

    # setting one x label
    axs[i-1].set_xlabel(xlabel, fontweight='bold')

    # displaying, saving
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
    if show:
        plt.show()


def normalised_barplot(df, x=None, cols=None,
                       output_folder='', filename='normalized_barplot.png', save=True,
                       show=True):
    """ Function creating a comparison normalised barplot.

    Args:
        df (pd.DataFrame): Input data.
        x (str, optional): Column to use for x-axis. If None, index is used.
        cols (List[str], optional): Columns to compare. If None, uses all numeric.
        save (bool): Whether to save the plot. Default is True.
        output_folder (str): Folder path for saving the plot. Default is ''.
        filename (str): Output filename. Default is 'normalized_barplot.png'.
        show (bool): Whether to display the plot. Default is True.

    Returns:
        None

    Raises:
        ValueError: If any of the columns' maximal value is zero.
        ValueError: If any of the cols are not in df.columns.
    """

    # specifying columns and x if not provided
    if cols is None:
        cols = df.select_dtypes(include='number').columns
    if x is None:
        x = df.index
        xlabel = df.index.name
    else:
        xlabel = x
        x = df[x]

    if len(x) > 30:
        raise ValueError(f'There are more than 30 values in the {xlabel} column. The plot would not be meaningful')

    # x-axis positions and bar width
    bar_width = 1/(len(cols)+1)
    x_pos = np.arange(len(x))
    bar_pos = np.arange(-0.5*(len(cols)-1), 0.5*(len(cols)-1)+1, 1)

    # plot
    _, ax = plt.subplots(figsize=(10, 6))

    # normalize the values (divide by max) and draw the bars
    i = 0
    for col in cols:
        if col not in df.columns:
            raise ValueError('Some of the columns not in dataframe.')
        if df[col].max() == 0:
            raise ValueError(f'The maximal value in {col} column is zero. Please provide valid data.')
        values = df[col]/df[col].max()
        ax.bar(x_pos + bar_pos[i] * bar_width, values,
               width=bar_width, color=sns.color_palette()[i], label=col)
        i = i + 1

    # Labels and ticks
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel('Normalized Value (0-1)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=90)
    ax.legend()

    # display and print
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
    if show:
        plt.show()


def corr_plot(df, cols, output_folder='', filename='corr_plot.png', save=True, show=True):
    """ Function creating a Pearson correlation plot.

    Args:
        df (pd.DataFrame): Input data.
        cols (List[str]): Columns to include in correlation.
        save (bool): Whether to save the plot. Default is True.
        output_folder (str): Folder path to save the plot.
        filename (str): Filename for the output image.
        show (bool): Whether to display the plot. Default is True.

    Returns:
        None

    Raises:
        ValueError: If cols contains only one column.
        ValueError: If any of the columns are not in df.columns.
    """

    if len(cols) == 1:
        raise ValueError('Only one column provided. Please indicate two or more columns.')

    for col in cols:
        if col not in df.columns:
            raise ValueError('Some of the columns not in dataframe.')

    # calculate correlations
    corr_voi = df[cols].corr(method='pearson')

    # plot
    plt.figure(figsize=(8, 6), dpi=500)
    plt.title('Correlation of the variables of interest')
    sns.heatmap(corr_voi, annot=True, fmt=".2f", linewidth=.5, cmap="coolwarm")

    if save:
        plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
    if show:
        plt.show()

def density_corr_plot(df, cols, area, output_folder='', filename='corr_plot.png', save=True, show=True):
    """ Function creating a Pearson correlation plot.

    Args:
        df (pd.DataFrame): Input data.
        cols (List[str]): Columns to include in correlation.
        area (string): Column with the area data.
        save (bool): Whether to save the plot. Default is True.
        output_folder (str): Folder path to save the plot. Default is ''
        filename (str): Filename for the output image. Default is 'corr_plot.png'.
        show (bool): Whether to display the plot. Default is True.

    Returns:
        None

    Raises:
        ValueError: If cols contains only one column.
    """

    if len(cols) == 1:
        raise ValueError('Only one column provided. Please indicate two or more columns.')

    for i, col in enumerate(cols):
        df[col] = df[col] / df[area]
        df.rename(columns={col: col+'/Area [km2]'}, inplace=True)
        cols[i] = col+'/Area [km2]'

    # calculate correlations
    density_corr_voi = df[cols].corr(method='pearson')

    # plot
    plt.figure(figsize=(8, 6), dpi=500)
    plt.title('Correlation of the variables\' densities')
    sns.heatmap(density_corr_voi, annot=True, fmt=".2f", linewidth=.5, cmap="coolwarm")

    if save:
        plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
    if show:
        plt.show()


def corr_test(x, y, description, output_folder='', output_file='corr_test.txt', save=True, overwrite=False, printer=False):
    """ Function performing a correlation test.

    Args:
        x (pd.Series): First variable.
        y (pd.Series): Second variable.
        description (str): Description of the test to print/save.
        save (bool): Whether to save the results of the test. Default is True.
        output_folder (str): Folder to save the results. Default is ''.
        output_file (str): Name of the output text file. Default is 'corr_test.txt'.
        overwrite (bool): If True, overwrites existing file. Otherwise appends. Default is False.
        printer (bool): If True, prints the result. Default is False.

    Returns:
        None
    """

    if overwrite:
        option = 'w'
    else:
        option = 'a'

    corr = pearsonr(x, y)
    if printer:
        print(description+': \n', round(corr.statistic, 2), ' ' , corr.pvalue, sep='')
    if save:
        with open(os.path.join(output_folder, output_file), option, encoding="utf-8") as f:
            f.write(description + ': \n'
                    + str(round(corr.statistic, 2)) + ' '
                    + str(corr.pvalue) + '\n')
