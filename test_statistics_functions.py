import pytest
import pandas as pd
import matplotlib

matplotlib.use('Agg')
from data_analysis_package import statistics_functions as stf

# statistics_table()

def test_statistics_table():

    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    result = stf.statistics_table(df)

    assert 'mean' in result.index
    assert result.loc['mean', 'A'] == 2.0
    assert '25%' not in result.index

def test_statistics_table_empty_df():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match='The dataframe is empty.'):
        stf.statistics_table(df)

# barplots()

def test_barplots_creates_file(tmp_path):
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
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200]
    })

    output = tmp_path / 'bars_one_col.png'
    stf.barplots(df, x='Region', cols=['Population'], save=True,
                 output_folder=str(tmp_path), filename='bars_one_col.png', show=False)

    assert output.exists()

def test_barplots_wrong_columns(tmp_path):
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200]
    })

    with pytest.raises(ValueError, match='Some of the columns not in dataframe.'):
        stf.barplots(df, x='Region', cols=['Population', 'Fires'], save=False, show=False)

# normalised_barplot()

def test_normalised_barplot_creates_file(tmp_path):
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
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200]
    })
    output = tmp_path / 'norm_one_col.png'
    stf.normalised_barplot(df, x='Region', cols=['Population'], save=True,
                           output_folder=str(tmp_path), filename='norm_one_col.png', show=False)

    assert output.exists()

def test_normalised_barplot_wrong_columns(tmp_path):
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200]
    })

    with pytest.raises(ValueError, match='Some of the columns not in dataframe.'):
        stf.normalised_barplot(df, x='Region', cols=['Population', 'Fires'], save=False, show=False)

def test_normalised_barplot_zero_max(tmp_path):
    df = pd.DataFrame({
        'Region': ['A', 'B'],
        'Population': [100, 200],
        'Fires': [0, 0]
    })

    with pytest.raises(ValueError, match='column is zero'):
        stf.normalised_barplot(df, x='Region', cols=['Population', 'Fires'], save=False, show=False)

# corr_plot()

def test_corr_plot_creates_file(tmp_path):
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
    df = pd.DataFrame({
        'Region': ['A', 'B', 'C'],
        'Population': [100, 200, 300],
        'Fires': [3, 5, 7]
    })

    with pytest.raises(ValueError, match='Some of the columns not in dataframe.'):
        stf.corr_plot(df, cols=['Population', 'Fires', 'Alcohol'], save=False, show=False)

# corr_test()

def test_corr_test_saves_output(tmp_path):
    x = pd.Series([1, 2, 3])
    y = pd.Series([2, 4, 6])

    output = tmp_path / 'corr.txt'
    stf.corr_test(x, y, description='Test Corr', output_folder=str(tmp_path),
                 output_file='corr.txt', overwrite=True, printer=False)

    content = output.read_text()
    assert 'Test Corr' in content
    assert str(round(x.corr(y), 2)) in content

