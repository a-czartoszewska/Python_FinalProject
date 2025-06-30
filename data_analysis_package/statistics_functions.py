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
from .data_prep import columns_present

def statistics_table(df, cols=None):
    """ Returns a table with basic statistics: min, average, median (50%), max.

    Args:
        df (pd.DataFrame): Dataframe to summarize.
        cols (list[str], optional): Columns to include. If None, includes all.

    Returns:
        pd.DataFrame: Table with basic statistics.

    Raises:
        ValueError: If the DataFrame is empty.
    """

    # Raise Error if the df is empty
    if df.empty:
        raise ValueError('The DataFrame is empty. Please provide a valid DataFrame.')

    # handle cols=None
    if cols is None:
        cols = df.columns

    # prepare the summary table
    df = df[cols]
    stat_table = df.describe()
    stat_table = stat_table.round(2)
    stat_table.drop(labels=['count', '25%', '75%'], inplace=True)

    return stat_table

def barplots(df, x=None, cols=None,
             output_folder='', filename='barplots.png', save=True,
             show=True):
    """ Creates a set of barplots.

    Args:
        df (pd.DataFrame): Input data.
        x (str, optional): Column to use for x-axis. If None, index is used.
        cols (List[str], optional): Columns to plot.
            If None, plots all numeric (apart from x)
        output_folder (str): Folder path for saving the plot. Default is ''.
        filename (str): Name of the output file. Default is 'barplots.png'.
        save (bool): Whether to save the plot to file. Default is True.
        show (bool): Whether to display the plot. Default is True.

    Returns:
        None

    Raises:
        ValueError: If some of the cols are not in the df DataFrame.
        ValueError: If there are more than 30 bars to plot.
    """

    # handle the arguments
    if cols is None:
        cols = [col for col in df.select_dtypes(include='number').columns if col != 'x']
    if x is None:
        x = df.index
        xlabel = df.index.name
    else:
        xlabel = x
        x = df[x]

    # Raise Error if the columns are not in the DataFrame
    columns_present(df, cols)

    # Raise Error if there are too many bars to plot
    if len(x) > 30:
        raise ValueError(f'There are more than 30 values in the {xlabel} column. '
                         f'The plot would not be meaningful')

    # create the plot
    _, axs = plt.subplots(len(cols), 1, figsize=(8, 12), sharex=True)

    # draw each barplot
    if len(cols) == 1:
        axs = [axs]

    i = 0
    for col in cols:
        axs[i].bar(x, df[col], color=sns.color_palette()[i])
        axs[i].set_ylabel(col, fontweight='bold')
        i = i + 1

    # set one x label
    axs[i-1].set_xlabel(xlabel, fontweight='bold')

    # save and display
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
    if show:
        plt.show()


def normalised_barplot(df, x=None, cols=None,
                       output_folder='', filename='normalised_barplot.png', save=True,
                       show=True):
    """ Creates a normalised, comparison barplot.

    Args:
        df (pd.DataFrame): Input data.
        x (str, optional): Column to use for x-axis. If None, index is used.
        cols (List[str], optional): Columns to compare.
            If None, uses all numeric, apart from x.
        output_folder (str): Folder path for saving the plot. Default is ''.
        filename (str): Output filename. Default is 'normalized_barplot.png'.
        save (bool): Whether to save the plot. Default is True.
        show (bool): Whether to display the plot. Default is True.

    Returns:
        None

    Raises:
        ValueError: If any of the columns' maximal value is zero,
            if any of the cols are not in the df DataFrame,
            or if there are more than 30 bars to plot, or more than 6 cols to compare.
    """

    # specify columns and x if not provided, set xlabel
    if cols is None:
        cols = [col for col in df.select_dtypes(include='number').columns if col != 'x']
    if x is None:
        x = df.index
        xlabel = df.index.name
    else:
        xlabel = x
        x = df[x]

    # Raise ValueError if the columns are not in the DataFrame
    columns_present(df, cols)

    # Raise ValueError if too much data
    if len(x) > 30:
        raise ValueError(f'There are more than 30 values in the {xlabel} column. The plot would not be meaningful')
    if len(cols) > 6:
        raise ValueError(f'There are more than 6 columns to compare. The plot would not be meaningful')

    # set x-axis positions and bar width
    bar_width = 1/(len(cols)+1)
    x_pos = np.arange(len(x))
    bar_pos = np.arange(-0.5*(len(cols)-1), 0.5*(len(cols)-1)+1, 1)

    # create plot
    _, ax = plt.subplots(figsize=(10, 6))

    # normalise the values (divide by max) and draw the bars
    i = 0
    for col in cols:
        if df[col].max() == 0:
            raise ValueError(f'The maximal value in {col} column is zero. Please provide valid data for normalization.')
        values = df[col]/df[col].max()
        ax.bar(x_pos + bar_pos[i] * bar_width, values,
               width=bar_width, color=sns.color_palette()[i], label=col)
        i = i + 1

    # set labels and ticks
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel('Normalized Value (0-1)', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=90)
    ax.legend()

    # save and display
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
    if show:
        plt.show()


def corr_plot(df, cols, output_folder='', filename='corr_plot.png', save=True, show=True):
    """ Creates a Pearson correlation plot.

    Args:
        df (pd.DataFrame): Input data.
        cols (list[str]): Columns to include in correlation plot.
        output_folder (str): Folder path to save the plot. Default is ''.
        filename (str): Filename for the output image. Default is 'corr_plot.png'.
        save (bool): Whether to save the plot. Default is True.
        show (bool): Whether to display the plot. Default is True.

    Returns:
        None

    Raises:
        ValueError: If cols contains only one column,
            or if any of the columns are not in the DataFrame.
    """

    # Raise ValueError
    if len(cols) == 1:
        raise ValueError('Only one column provided. Please indicate two or more columns.')

    columns_present(df, cols)

    # calculate correlations
    corr_voi = df[cols].corr(method='pearson')

    # plot
    plt.figure(figsize=(8, 6))
    plt.title('Correlation of the variables of interest')
    sns.heatmap(corr_voi, annot=True, fmt='.2f', cmap='coolwarm')

    # save and print
    if save:
        plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
    if show:
        plt.show()

def density_corr_plot(df, cols, area, output_folder='', filename='corr_plot.png', save=True, show=True):
    """ Create a Pearson correlation plot on density data.

    Args:
        df (pd.DataFrame): Input data.
        cols (list[str]): Columns to include in correlation plot.
        area (string): Column with the area data.
        output_folder (str): Folder path to save the plot. Default is ''
        filename (str): Filename for the output image. Default is 'corr_plot.png'.
        save (bool): Whether to save the plot. Default is True.
        show (bool): Whether to display the plot. Default is True.

    Returns:
        None

    Raises:
        ValueError: If cols contains only one column,
            or if any of the columns are not in the DataFrame.
    """

    # Raise ValueError
    if len(cols) == 1:
        raise ValueError('Only one column provided. Please indicate two or more columns.')
    columns_present(df, cols)

    # calculate densities
    for i, col in enumerate(cols):
        df[col] = df[col] / df[area]
        df.rename(columns={col: col+'/Area [km2]'}, inplace=True)
        cols[i] = col+'/Area [km2]'

    # calculate correlations
    density_corr_voi = df[cols].corr(method='pearson')

    # plot
    plt.figure(figsize=(8, 6))
    plt.title('Correlation of the variables\' densities')
    sns.heatmap(density_corr_voi, annot=True, fmt=".2f", cmap="coolwarm")

    if save:
        plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight')
    if show:
        plt.show()


def corr_test(x, y, description,
              output_folder='', output_file='corr_test.txt',
              save=True, overwrite=False, printer=False):
    """ Performs a correlation test.

    Args:
        x (pd.Series): First variable.
        y (pd.Series): Second variable.
        description (str): Description of the test to print/save.
        output_folder (str): Folder to save the results. Default is ''.
        output_file (str): Name of the output text file. Default is 'corr_test.txt'.
        save (bool): Whether to save the results of the test. Default is True.
        overwrite (bool): If True, overwrites existing file.
            Otherwise, appends new information. Default is False.
        printer (bool): If True, prints the result. Default is False.

    Returns:
        None
    """

    # set overwriting
    option = 'w' if overwrite else 'a'

    # correlation test
    corr = pearsonr(x, y)

    # print and save the results
    if printer:
        print(description+': \n', round(corr.statistic, 2), ' ' , corr.pvalue, sep='')
    if save:
        with open(os.path.join(output_folder, output_file), option, encoding="utf-8") as f:
            f.write(description + ': \n'
                    + str(round(corr.statistic, 2)) + ' '
                    + str(corr.pvalue) + '\n')
