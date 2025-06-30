import pytest
import pandas as pd
import matplotlib

matplotlib.use('Agg')
from data_analysis_package import statistics_functions as stf

# statistics_table()

def test_statistics_table():
    """ Tests if the function returns proper data """
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = stf.statistics_table(df)

    assert 'mean' in result.index
    assert result.loc['mean', 'A'] == 2.0
    assert '25%' not in result.index

def test_statistics_table_cols():
    """ Tests if the function returns proper data on only chosen columns"""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = stf.statistics_table(df, cols=['A'])

    assert 'B' not in result.columns

def test_statistics_table_empty_df():
    """ Tests if function raises appropriate error if DataFrame is empty """
    df = pd.DataFrame()

    with pytest.raises(ValueError, match='The DataFrame is empty.'):
        stf.statistics_table(df)

# barplots()

def test_barplots_creates_file(tmp_path):
    """ Tests if the function works properly with valid data - if it creates a file """
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200],
        'Fires': [3, 5]
    })

    output = tmp_path / 'bars.png'
    stf.barplots(df, x='Region', cols=['Population', 'Fires'], save=True,
                output_folder=str(tmp_path), filename='bars.png', show=False)

    assert output.exists()

def test_barplots_one_column(tmp_path):
    """ Tests if the function works with only one column """
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200]
    })

    output = tmp_path / 'bars_one_col.png'
    stf.barplots(df, x='Region', cols=['Population'], save=True,
                 output_folder=str(tmp_path), filename='bars_one_col.png', show=False)

    assert output.exists()

def test_barplots_x_none(tmp_path):
    """ Tests if the function works if x is None"""
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200]
    })
    df.set_index('Region', inplace=True)

    output = tmp_path / 'bars_x_none.png'
    stf.barplots(df, cols=['Population'], save=True,
                 output_folder=str(tmp_path), filename='bars_x_none.png', show=False)

    assert output.exists()

def test_barplots_cols_none(tmp_path):
    """ Tests if the function works if cols is None"""
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200]
    })

    output = tmp_path / 'bars_cols_none.png'
    stf.barplots(df, x='Region', save=True,
                 output_folder=str(tmp_path), filename='bars_cols_none.png', show=False)

    assert output.exists()

def test_barplots_cols_x_none(tmp_path):
    """ Tests if the function works if cols and x are None"""
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200]
    })
    df.set_index('Region', inplace=True)

    output = tmp_path / 'bars_cols_x_none.png'
    stf.barplots(df, save=True,
                 output_folder=str(tmp_path), filename='bars_cols_x_none.png', show=False)

    assert output.exists()

def test_barplots_too_many_bars(tmp_path):
    """ Tests if the function raises an appropriate error when there are too many bars to plot """
    df = pd.DataFrame({
        'Region': [f'Region {i}' for i in range(31)],
        'Population': range(31),
        'Fires': range(31)
    })

    with pytest.raises(ValueError, match='There are more than 30 values'):
        stf.barplots(df, x='Region', cols=['Population', 'Fires'], save=False, show=False)

def test_barplots_wrong_columns(tmp_path):
    """ Tests if the function raises and appropriate error when columns are not in the DataFrame """
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200]
    })

    with pytest.raises(ValueError, match='Missing required column'):
        stf.barplots(df, x='Region', cols=['Population', 'Fires'], save=False, show=False)

# normalised_barplot()

def test_normalised_barplot_creates_file(tmp_path):
    """ Tests if the function works properly with valid data - if it creates a file """
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200],
        'Fires': [3, 5]
    })

    output = tmp_path / 'norm.png'
    stf.normalised_barplot(df, x='Region', cols=['Population', 'Fires'], save=True,
                          output_folder=str(tmp_path), filename='norm.png', show=False)

    assert output.exists()

def test_normalised_barplot_unchanged_original(tmp_path):
    """ Test if the function leaves the original DataFrame unchanged """
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200],
        'Fires': [3, 5]
    })

    output = tmp_path / 'norm.png'
    stf.normalised_barplot(df, x='Region', cols=['Population', 'Fires'], save=True,
                          output_folder=str(tmp_path), filename='norm.png', show=False)

    # check if the df is not changed
    assert df['Region'].tolist() == ['A', 'B']
    assert df['Population'].tolist() == [100, 200]
    assert df['Fires'].tolist() == [3, 5]

def test_normalised_barplot_one_column(tmp_path):
    """ Tests if the function works with only one column """
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200]
    })
    output = tmp_path / 'norm_one_col.png'
    stf.normalised_barplot(df, x='Region', cols=['Population'], save=True,
                           output_folder=str(tmp_path), filename='norm_one_col.png', show=False)

    assert output.exists()

def test_normalised_barplot_wrong_columns(tmp_path):
    """ Tests if the function raises and appropriate error when columns are not in the DataFrame """
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200]
    })

    with pytest.raises(ValueError, match='Missing required column'):
        stf.normalised_barplot(df, x='Region', cols=['Population', 'Fires'], save=False, show=False)

def test_normalised_barplot_too_many_bars(tmp_path):
    """ Tests if the function raises an appropriate error when there are too many bars to plot """
    df = pd.DataFrame({
        'Region': [f'Region {i}' for i in range(31)],
        'Population': range(31),
        'Fires': range(31)
    })

    with pytest.raises(ValueError, match='There are more than 30 values'):
        stf.normalised_barplot(df, x='Region', cols=['Population', 'Fires'], save=False, show=False)

def test_normalised_barplot_too_many_cols(tmp_path):
    """ Tests if the function raises an appropriate error
    when there are too many columns to compare """
    df = pd.DataFrame({
        'Region': [f'Region {i}' for i in range(25)],
        'A': range(25),
        'B': range(25),
        'C': range(25),
        'D': range(25),
        'E': range(25),
        'F': range(25),
        'G': range(25)
    })

    with pytest.raises(ValueError, match='There are more than 6 columns to compare'):
        stf.normalised_barplot(df, x='Region', cols=['A', 'B', 'C', 'D', 'E', 'F', 'G'], save=False, show=False)

def test_normalised_barplot_zero_max(tmp_path):
    """ Tests if function raises an appropriate error when max value of a column is zero """
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200],
        'Fires': [0, 0]
    })

    with pytest.raises(ValueError, match='column is zero'):
        stf.normalised_barplot(df, x='Region', cols=['Population', 'Fires'], save=False, show=False)

# corr_plot()

def test_corr_plot_creates_file(tmp_path):
    """ Tests if the function works properly with valid data - if it creates a file """
    df = pd.DataFrame({
        'Region': ['A', 'B', 'C'],
        'Population': [100, 200, 300],
        'Fires': [3, 5, 7],
        'Alcohol': [2, 6, 8]
    })

    output = tmp_path / 'corr.png'
    stf.corr_plot(df, cols=['Population', 'Fires', 'Alcohol'], output_folder=str(tmp_path),
                 filename='corr.png', show=False)

    assert output.exists()

def test_corr_plot_wrong_columns(tmp_path):
    """ Tests if the function raises and appropriate error when columns are not in the DataFrame """
    df = pd.DataFrame({
        'Region': ['A', 'B', 'C'],
        'Population': [100, 200, 300],
        'Fires': [3, 5, 7]
    })

    with pytest.raises(ValueError, match='Missing required column'):
        stf.corr_plot(df, cols=['Population', 'Fires', 'Alcohol'], save=False, show=False)

# corr_test()

def test_corr_test_saves_output(tmp_path):
    """ Tests if the function saves output in a file """
    x = pd.Series([1, 2, 3])
    y = pd.Series([2, 4, 6])

    output = tmp_path / 'corr.txt'
    stf.corr_test(x, y, description='Test Corr', output_folder=str(tmp_path),
                 output_file='corr.txt', overwrite=True, printer=False)

    content = output.read_text()
    assert 'Test Corr' in content
    assert str(round(x.corr(y), 2)) in content

