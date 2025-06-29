import pytest
import pandas as pd
from data_analysis_package import data_prep as dp


# dp.load_data()

def test_load_data_correct(tmp_path, monkeypatch):

    def mock_read(filename, skiprows=None, usecols=None):
        return pd.DataFrame({'dummy': [1]})

    monkeypatch.setattr('pandas.read_csv', mock_read)
    monkeypatch.setattr('pandas.read_excel', mock_read)

    # Create fake files
    files = ['population.xls', 'area.xlsx', 'alcohol.csv', 'fire.csv']

    for f in files:
        (tmp_path / f).write_text("fake content")


    df1, df2, df3, df4 = dp.load_data(str(tmp_path), files)

    assert not df1.empty
    assert not df2.empty
    assert not df3.empty
    assert not df4.empty

def test_load_data_incorrect_file_list_length(tmp_path):

    files= ['population.xls', 'area.xlsx', 'alcohol.csv']

    # Create fake files with correct extensions
    for f in files:
        (tmp_path / f).write_text("fake content")

    with pytest.raises(ValueError, match='file_list must contain exactly 4 filenames'):
        dp.load_data(str(tmp_path), files)

def test_load_data_missing_file(tmp_path):

    files = ['population.xls', 'area.xlsx', 'alcohol.csv', 'fire.csv']

    # Create fake files with correct extensions
    for f in files:
        (tmp_path / f).write_text("fake content")

    with pytest.raises(FileNotFoundError):
        dp.load_data(str(tmp_path), ['missing_population.xls', 'area.xlsx', 'alcohol.csv', 'fire.csv'])

@pytest.mark.parametrize('wrong_ext_file', ['population.txt', 'area.json', 'alcohol.xls', 'fire.xlsx'])
def test_load_data_wrong_extension(tmp_path, wrong_ext_file):

    correct_ext = ['population.xls', 'area.xlsx', 'alcohol.csv', 'fire.csv']

    # Create fake files with correct extensions
    for f in correct_ext:
        (tmp_path / f).write_text("fake content")

    (tmp_path / wrong_ext_file).write_text("fake content")

    for i, file in enumerate(correct_ext):
        if file.startswith(wrong_ext_file.split('.')[0]):
            correct_ext[i] = wrong_ext_file

    with pytest.raises(ValueError, match='File format not supported'):
        dp.load_data(str(tmp_path), correct_ext)

@pytest.mark.parametrize('empty_file', ['population_empty.xls', 'area_empty.xlsx', 'alcohol_empty.csv', 'fire_empty.csv'])
def test_load_data_empty_file(tmp_path, empty_file, monkeypatch):

    def mock_read(filename, skiprows=None, usecols=None):
        if filename.endswith(empty_file):
            return pd.DataFrame()
        return pd.DataFrame({'dummy': [1]})

    monkeypatch.setattr('pandas.read_csv', mock_read)
    monkeypatch.setattr('pandas.read_excel', mock_read)

    files = ['population.xls', 'area.xlsx', 'alcohol.csv', 'fire.csv']

    # Create fake files with correct extensions
    for f in files:
        (tmp_path / f).write_text("fake content")

    (tmp_path / empty_file).write_text("")

    for i, file in enumerate(files):
        if file.startswith(empty_file.split('_')[0]):
            files[i] = empty_file

    with pytest.raises(ValueError, match='Please provide a valid file'):
        dp.load_data(str(tmp_path), files)


# dp.data_inspection()

def test_data_inspection():
    df = pd.DataFrame()
    with pytest.raises(ValueError, match='DataFrame is empty'):
        dp.data_inspection(df)

def test_data_inspection_non_empty(capsys):
    df = pd.DataFrame({
        'A': [1, 2, None],
        'B': [3, None, 5]
    })

    dp.data_inspection(df)

    captured = capsys.readouterr()
    assert 'First 10 rows' in captured.out
    assert '(3, 2)' in captured.out
    assert 'Columns' in captured.out
    assert 'Number of rows with missing data' in captured.out

def test_data_inspection_missing_data(capsys):
    df = pd.DataFrame({
        'A': [1, 2, None],
        'B': [3, 4, 5]
    })
    dp.data_inspection(df)
    captured = capsys.readouterr()

    assert 'Rows with missing data' in captured.out
    assert '0' in captured.out


# dp.relevant_data_prep_and_rename()

@pytest.mark.parametrize(
    'missing_df_index, missing_column',
    [
        (0, 'Województwa\nVoivodships'),
        (0, 'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)'),
        (1, 'TERYT'),
        (1, 'Nazwa jednostki'),
        (1, 'Powierzchnia [km2]'),
        (2, 'Numer zezwolenia'),
        (2, 'Województwo'),
        (3, 'TERYT'),
        (3, 'Województwo'),
        (3, 'Powiat'),
        (3, 'Gmina'),
        (3, 'OGÓŁEM Liczba zdarzeń'),
    ]
)
def test_relevant_data_prep_and_rename_missing_columns(missing_df_index, missing_column):

    # fake valid data
    dfs = [
        pd.DataFrame({
            'Województwa\nVoivodships': ['Mazowieckie'],
            'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000]
        }),
        pd.DataFrame({
            'TERYT': [1],
            'Nazwa jednostki': ['WOJ. MAZOWIECKIE'],
            'Powierzchnia [ha]': [1000000],
            'Powierzchnia [km2]': [10000]
        }),
        pd.DataFrame({
            'Numer zezwolenia': [123],
            'Województwo': ['WOJ. MAZOWIECKIE']
        }),
        pd.DataFrame({
            'TERYT': [1],
            'Województwo': ['mazowieckie'],
            'Powiat': ['X'],
            'Gmina': ['A'],
            'OGÓŁEM Liczba zdarzeń': [10]
        })
    ]

    # drop one of the relevant columns
    df = dfs[missing_df_index].drop(columns=[missing_column])
    dfs[missing_df_index] = df

    with pytest.raises(ValueError, match='relevant column'):
        dp.relevant_data_prep_and_rename(*dfs)


def test_relevant_data_prep_and_rename_correct():

    dfs = [
        pd.DataFrame({
            'Województwa\nVoivodships': ['Mazowieckie', 'Lubelskie', None],
            'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, 2000000, None],
            'Irrelevant_pop': [25235, 23425345, None]
        }),
        pd.DataFrame({
            'TERYT': [1, 2],
            'Nazwa jednostki': ['WOJ. MAZOWIECKIE', 'WOJ. LUBELSKIE'],
            'Powierzchnia [ha]': [1000000, 500000],
            'Powierzchnia [km2]': [10000, 5000],
            'Irrelevant_are': [25235, 23425345]
        }),
        pd.DataFrame({
            'Numer zezwolenia': [123, 456],
            'Województwo': ['WOJ. MAZOWIECKIE', 'WOJ. LUBELSKIE'],
            'Irrelevant_alc': [25235, 23425345]
        }),
        pd.DataFrame({
            'TERYT': [1, 2],
            'Województwo': ['mazowieckie', 'lubelskie'],
            'Powiat': ['X', 'Y'],
            'Gmina': ['A', 'B'],
            'OGÓŁEM Liczba zdarzeń': [10, 20],
            'Irrelevant_fir': [25235, 23425345]
        })
    ]

    df_p, df_a, df_alc, df_f = dp.relevant_data_prep_and_rename(*dfs)

    # check that renaming occurred
    assert 'Voivodship' in df_p.columns
    assert 'Population' in df_p.columns
    assert 'Territory code' in df_a.columns
    assert 'Unit name' in df_a.columns
    assert 'Permit number' in df_alc.columns
    assert 'Total number of fires' in df_f.columns

    # check that the irrelevant columns were not copied / were dropped
    assert 'Irrelevant_pop' not in df_p.columns
    assert 'Irrelevant_are' not in df_a.columns
    assert 'Area [ha]' not in df_a.columns
    assert 'Irrelevant_alc' not in df_alc.columns
    assert 'Irrelevant_fir' not in df_f.columns

    # check that the NaNs were dropped in population data
    assert len(df_p) == 2

# by_voivodship()

def test_by_voivodship_correct_columns():
    dfs = [
        pd.DataFrame({
            'Voivodship': ['Mazowieckie', 'Lubelskie'],
            'Population': [1000000, 20000]
        }),
        pd.DataFrame({
            'Territory code': [1, 2],
            'Unit name': ['WOJ. MAZOWIECKIE', 'WOJ. LUBELSKIE'],
            'Area [km2]': [10000, 5000]
        }),
        pd.DataFrame({
            'Permit number': [123, 456, 789],
            'Voivodship': ['WOJ. MAZOWIECKIE', 'WOJ. LUBELSKIE', 'WOJ. LUBELSKIE']
        }),
        pd.DataFrame({
            'Territory code': [1, 2, 3, 4],
            'Voivodship': ['mazowieckie', 'mazowieckie', 'lubelskie', 'lubelskie'],
            'County': ['X', 'Y', 'Z', 'A'],
            'Municipality': ['A', 'B', 'C', 'D'],
            'Total number of fires': [10, 20, 30, 40]
        })
    ]

    df_voi, df_pop, df_are, df_alc, df_fir = dp.by_voivodship(*dfs)

    assert 'Voivodship' == df_voi.index.name
    assert 'Voivodship' == df_are.index.name
    assert 'Voivodship' == df_alc.index.name
    assert 'Voivodship' == df_fir.index.name
    assert 'Territory code' in df_voi.columns
    assert 'Population' in df_voi.columns
    assert 'Area [km2]' in df_voi.columns
    assert 'Total number of fires' in df_voi.columns
    assert 'Total number of alcohol permits' in df_voi.columns
    assert 'Territory code' in df_are.columns
    assert 'Area [km2]' in df_are.columns
    assert 'Total number of alcohol permits' in df_alc.columns
    assert 'Total number of fires' in df_fir.columns

def test_by_voivodship_correct_values():
    dfs = [
        pd.DataFrame({
            'Voivodship': ['Mazowieckie', 'Lubelskie'],
            'Population': [1000000, 20000]
        }),
        pd.DataFrame({
            'Territory code': [1, 2],
            'Unit name': ['WOJ. MAZOWIECKIE', 'WOJ. LUBELSKIE'],
            'Area [km2]': [10000, 5000]
        }),
        pd.DataFrame({
            'Permit number': [123, 456, 789],
            'Voivodship': ['WOJ. MAZOWIECKIE', 'WOJ. LUBELSKIE', 'WOJ. LUBELSKIE']
        }),
        pd.DataFrame({
            'Territory code': [1, 2, 3, 4],
            'Voivodship': ['mazowieckie', 'mazowieckie', 'lubelskie', 'lubelskie'],
            'County': ['X', 'Y', 'Z', 'A'],
            'Municipality': ['A', 'B', 'C', 'D'],
            'Total number of fires': [10, 20, 30, 40]
        })
    ]

    df_voi, df_pop, df_are, df_alc, df_fir = dp.by_voivodship(*dfs)

    assert df_voi.shape[0] == 2
    assert df_voi.shape[1] == 5
    assert df_voi[df_voi.index == 'WOJ. MAZOWIECKIE']['Total number of fires'].iloc[0] == 30
    assert df_voi[df_voi.index == 'WOJ. LUBELSKIE']['Total number of fires'].iloc[0] == 70
    assert df_voi[df_voi.index == 'WOJ. MAZOWIECKIE']['Total number of alcohol permits'].iloc[0] == 1
    assert df_voi[df_voi.index == 'WOJ. LUBELSKIE']['Total number of alcohol permits'].iloc[0] == 2

@pytest.mark.parametrize(
    'missing_df_index, missing_column',
    [
        (0, 'Voivodship'),
        (0, 'Population'),
        (1, 'Territory code'),
        (1, 'Unit name'),
        (1, 'Area [km2]'),
        (2, 'Permit number'),
        (2, 'Voivodship'),
        (3, 'Territory code'),
        (3, 'Voivodship'),
        (3, 'County'),
        (3, 'Municipality'),
        (3, 'Total number of fires'),
    ]
)
def test_by_voivodship_missing_columns(missing_df_index, missing_column):

    dfs = [
        pd.DataFrame({
            'Voivodship': ['Mazowieckie', 'Lubelskie'],
            'Population': [1000000, 20000]
        }),
        pd.DataFrame({
            'Territory code': [1, 2],
            'Unit name': ['WOJ. MAZOWIECKIE', 'WOJ. LUBELSKIE'],
            'Area [km2]': [10000, 5000]
        }),
        pd.DataFrame({
            'Permit number': [123, 456, 789],
            'Voivodship': ['WOJ. MAZOWIECKIE', 'WOJ. LUBELSKIE', 'WOJ. LUBELSKIE']
        }),
        pd.DataFrame({
            'Territory code': [1, 2, 3, 4],
            'Voivodship': ['mazowieckie', 'mazowieckie', 'lubelskie', 'lubelskie'],
            'County': ['X', 'Y', 'Z', 'A'],
            'Municipality': ['A', 'B', 'C', 'D'],
            'Total number of fires': [10, 20, 30, 40]
        })
    ]

    df = dfs[missing_df_index].drop(columns=[missing_column])
    dfs[missing_df_index] = df

    with pytest.raises(ValueError, match='Please provide a valid dataframe'):
        dp.by_voivodship(*dfs)