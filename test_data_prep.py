import pytest
import pandas as pd
import re
from data_analysis_package import data_prep as dp

# columns_present()
def test_columns_present_correct():
    """ Tests if the function do not raises an error when the columns are present """
    df = pd.DataFrame({
        'A': [1, 2, None],
        'B': [3, None, 5]
    })

    dp.columns_present(df, ['A', 'B'], 'df')

def test_columns_present_missing_one():
    """ Tests if the function raises an error with appropriate message
    when one column is missing """
    df = pd.DataFrame({})

    with pytest.raises(ValueError, match=re.escape('Missing required column(s) in df: A')):
        dp.columns_present(df, ['A'], 'df')

def test_columns_present_missing_two():
    """ Tests if the function raises an error with appropriate message
    when two columns are missing"""
    df = pd.DataFrame({})

    with pytest.raises(ValueError, match=re.escape('Missing required column(s) in df: A, B')):
        dp.columns_present(df, ['A', 'B'], 'df')

def test_columns_present_column_is_index():
    """ Tests if the function do not raises an error when one of the columns is the index """
    df = pd.DataFrame({
        'A': [1, 2, None],
        'B': [3, None, 5]
    })
    df = df.set_index('A')
    dp.columns_present(df, ['A', 'B'], 'df')

# load_data()
def test_load_data_correct(tmp_path, monkeypatch):
    """ Tests if the function works properly (returns non-empty DataFrames)
    when data is loaded correctly """
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
    """ Tests if the function raises an appropriate error when the file list length is wrong """
    files= ['population.xls', 'area.xlsx', 'alcohol.csv']

    # Create fake files with correct extensions
    for f in files:
        (tmp_path / f).write_text("fake content")

    with pytest.raises(ValueError, match='file_list must contain exactly 4 filenames'):
        dp.load_data(str(tmp_path), files)

def test_load_data_missing_file(tmp_path):
    """ Tests if the function raises an appropriate error when a file is missing """
    files = ['population.xls', 'area.xlsx', 'alcohol.csv', 'fire.csv']

    # Create fake files with correct extensions
    for f in files:
        (tmp_path / f).write_text("fake content")

    with pytest.raises(FileNotFoundError):
        # add one non-existent file
        dp.load_data(str(tmp_path), ['missing_population.xls', 'area.xlsx', 'alcohol.csv', 'fire.csv'])

@pytest.mark.parametrize('wrong_ext_file', ['population.txt', 'area.json', 'alcohol.xls', 'fire.xlsx'])
def test_load_data_wrong_extension(tmp_path, wrong_ext_file):
    """ Tests if the function raises an appropriate error
    when any of the files is in a wrong format """
    # file list with right extensions
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

@pytest.mark.parametrize('empty_file', ['population_empty.xls', 'area_empty.xlsx',
                                        'alcohol_empty.csv', 'fire_empty.csv'])
def test_load_data_empty_file(tmp_path, empty_file, monkeypatch):
    """ Tests if the function raises an appropriate error if any of the files is empty """
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


# data_inspection()

def test_data_inspection_empty_df():
    """ Tests if the function raises an appropriate error when the DataFrame is empty """
    df = pd.DataFrame()
    with pytest.raises(ValueError, match='DataFrame is empty'):
        dp.data_inspection(df)

def test_data_inspection_correct(capsys):
    """ Tests if the function works properly when the DataFrame is non-empty """
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
    """ Tests if the function is working properly when there is missing data in the DataFrame """
    df = pd.DataFrame({
        'A': [1, 2, None],
        'B': [3, 4, 5]
    })
    dp.data_inspection(df)
    captured = capsys.readouterr()

    assert 'Rows with missing data' in captured.out
    assert '0' in captured.out

# relevant_data_prep_and_rename_pop_voi()

@pytest.mark.parametrize('missing_col',
                         ['Województwa\nVoivodships',
                          'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)'])
def test_relevant_dpar_pop_voi_missing_cols(missing_col):
    """ Tests if function raises an appropriate error when the DataFrame is missing a required column """

    # fake df
    df = pd.DataFrame({
        'Województwa\nVoivodships': ['Mazowieckie'],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000]
    })

    df = df.drop(columns=[missing_col])

    with pytest.raises(ValueError, match='Missing required column'):
        dp.relevant_data_prep_and_rename_pop_voi(df)

def test_relevant_dpar_pop_voi_rename():
    """ Tests if function renames the columns properly """
    # fake df
    df = pd.DataFrame({
        'Województwa\nVoivodships': ['Mazowieckie'],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000]
    })

    df = dp.relevant_data_prep_and_rename_pop_voi(df)

    assert 'Voivodship' in df.columns
    assert 'Population' in df.columns


def test_relevant_dpar_pop_voi_missing_data():
    """ Tests if function drops rows with missing data """
    # fake df
    df = pd.DataFrame({
        'Województwa\nVoivodships': ['Mazowieckie', 'Łódzkie'],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, None]
    })

    df = dp.relevant_data_prep_and_rename_pop_voi(df)

    assert df.shape[0] == 1

def test_relevant_dpar_pop_voi_irrelevant_columns():
    """ Tests if function drops irrelevant columns """
    # fake df
    df = pd.DataFrame({
        'Województwa\nVoivodships': ['Mazowieckie', 'Łódzkie'],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, None],
        'X': [1, 2]
    })

    df = dp.relevant_data_prep_and_rename_pop_voi(df)

    assert 'X' not in df.columns

# relevant_data_prep_and_rename_pop_pow()

@pytest.mark.parametrize('missing_col',
                         ['Województwa \nVoivodships\nPowiaty\nPowiats',
                          'Identyfikator terytorialny\nCode',
                          'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)'])
def test_relevant_dpar_pop_pow_missing_cols(missing_col):
    """ Tests if function raises an appropriate error when the DataFrame is missing a required column """

    # fake df
    df = pd.DataFrame({
        'Województwa \nVoivodships\nPowiaty\nPowiats': ['Otwocki'],
        'Identyfikator terytorialny\nCode': [123],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000]
    })

    df = df.drop(columns=[missing_col])

    with pytest.raises(ValueError, match='Missing required column'):
        dp.relevant_data_prep_and_rename_pop_pow(df)

def test_relevant_dpar_pop_pow_rename():
    """ Tests if function renames the columns properly """
    # fake df
    df = pd.DataFrame({
        'Województwa \nVoivodships\nPowiaty\nPowiats': ['Otwocki'],
        'Identyfikator terytorialny\nCode': [123],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000]
    })

    df = dp.relevant_data_prep_and_rename_pop_pow(df)

    assert 'Powiat' in df.columns
    assert 'Population' in df.columns
    assert 'Territory code' in df.columns


def test_relevant_dpar_pop_pow_missing_data_powiat():
    """ Tests if function drops rows with missing data in powiat column """
    # fake df
    df = pd.DataFrame({
        'Województwa \nVoivodships\nPowiaty\nPowiats': ['Otwocki', None],
        'Identyfikator terytorialny\nCode': [123, 456],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, 2345]
    })

    df = dp.relevant_data_prep_and_rename_pop_pow(df)

    assert df.shape[0] == 1

def test_relevant_dpar_pop_pow_missing_data_code():
    """ Tests if function drops rows with missing data in code column """
    # fake df
    df = pd.DataFrame({
        'Województwa \nVoivodships\nPowiaty\nPowiats': ['Otwocki', 'Xxx'],
        'Identyfikator terytorialny\nCode': [123, None],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, 2345]
    })

    df = dp.relevant_data_prep_and_rename_pop_pow(df)

    assert df.shape[0] == 1

def test_relevant_dpar_pop_pow_not_powiat():
    """ Tests if function drops rows with voivodship name in powiat column """
    # fake df
    df = pd.DataFrame({
        'Województwa \nVoivodships\nPowiaty\nPowiats': ['Otwocki', 'WOJ. MAZOWIECKIE'],
        'Identyfikator terytorialny\nCode': [123, 7652],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, 2345]
    })

    df = dp.relevant_data_prep_and_rename_pop_pow(df)

    assert df.shape[0] == 1

def test_relevant_dpar_pop_pow_irrelevant_columns():
    """ Tests if function drops irrelevant columns """
    # fake df
    df = pd.DataFrame({
        'Województwa \nVoivodships\nPowiaty\nPowiats': ['Otwocki'],
        'Identyfikator terytorialny\nCode': [123],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000],
        'X': [1]
    })

    df = dp.relevant_data_prep_and_rename_pop_pow(df)

    assert 'X' not in df.columns

# relevant_data_prep_and_rename_pop_gm()

@pytest.mark.parametrize('missing_col',
                         ['Województwa\nVoivodships\nGminy\nGminas',
                          'Identyfikator terytorialny\nCode',
                          'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)'])
def test_relevant_dpar_pop_gm_missing_cols(missing_col):
    """ Tests if function raises an appropriate error when the DataFrame is missing a required column """

    # fake df
    df = pd.DataFrame({
        'Województwa\nVoivodships\nGminy\nGminas': ['Otwock'],
        'Identyfikator terytorialny\nCode': [123],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000]
    })

    df = df.drop(columns=[missing_col])

    with pytest.raises(ValueError, match='Missing required column'):
        dp.relevant_data_prep_and_rename_pop_gm(df)

def test_relevant_dpar_pop_gm_rename():
    """ Tests if function renames the columns properly """
    # fake df
    df = pd.DataFrame({
        'Województwa\nVoivodships\nGminy\nGminas': ['Otwock'],
        'Identyfikator terytorialny\nCode': [123],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000]
    })

    df = dp.relevant_data_prep_and_rename_pop_gm(df)

    assert 'Gmina' in df.columns
    assert 'Population' in df.columns
    assert 'Territory code' in df.columns


def test_relevant_dpar_pop_gm_missing_data():
    """ Tests if function drops rows with missing data in code column """
    # fake df
    df = pd.DataFrame({
        'Województwa\nVoivodships\nGminy\nGminas': ['Otwock', 'X'],
        'Identyfikator terytorialny\nCode': [123, None],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, 4444444]
    })

    df = dp.relevant_data_prep_and_rename_pop_gm(df)

    assert df.shape[0] == 1

def test_relevant_dpar_pop_gm_not_gmina():
    """ Tests if function drops rows with voivodship name in gmina column """
    # fake df
    df = pd.DataFrame({
        'Województwa\nVoivodships\nGminy\nGminas': ['Otwock', 'WOJ. MAZOWIECKIE'],
        'Identyfikator terytorialny\nCode': [123, 2345],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, 4444444]
    })

    df = dp.relevant_data_prep_and_rename_pop_gm(df)

    assert df.shape[0] == 1

def test_relevant_dpar_pop_gm_irrelevant_columns():
    """ Tests if function drops irrelevant columns """
    # fake df
    df = pd.DataFrame({
        'Województwa\nVoivodships\nGminy\nGminas': ['Otwock', 'WOJ. MAZOWIECKIE'],
        'Identyfikator terytorialny\nCode': [123, 2345],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, 4444444],
        'X': [1, 2]
    })

    df = dp.relevant_data_prep_and_rename_pop_gm(df)

    assert 'X' not in df.columns

def test_relevant_dpar_pop_gm_code_reformat():
    """ Tests if function reformats codes properly """
    df = pd.DataFrame({
        'Województwa\nVoivodships\nGminy\nGminas': ['Otwock', 'Xxx'],
        'Identyfikator terytorialny\nCode': ['1234', '5678'],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, 4444444]
    })

    df = dp.relevant_data_prep_and_rename_pop_gm(df)

    assert df['Territory code'].tolist() == [123, 567]

def test_relevant_dpar_pop_gm_aggregate():
    """ Tests if function aggregates data on gminas properly """

    df = pd.DataFrame({
        'Województwa\nVoivodships\nGminy\nGminas': ['Otwock', 'Otwock'],
        'Identyfikator terytorialny\nCode': ['1234', '1234'],
        'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, 4444444]
    })

    df = dp.relevant_data_prep_and_rename_pop_gm(df)

    assert df.shape[0] == 1
    assert df['Population'][0] == 4444444

# relevant_data_prep_and_rename()

@pytest.mark.parametrize(
    'missing_df_index, missing_column',
    [
        (0, 'TERYT'),
        (0, 'Nazwa jednostki'),
        (0, 'Powierzchnia [km2]'),
        (1, 'Numer zezwolenia'),
        (1, 'Województwo'),
        (2, 'TERYT'),
        (2, 'Województwo'),
        (2, 'Powiat'),
        (2, 'Gmina'),
        (2, 'OGÓŁEM Liczba zdarzeń'),
    ]
)
def test_relevant_data_prep_and_rename_missing_columns(missing_df_index, missing_column):

    # fake valid data on area, alcohol permits and fire events
    dfs = [
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
    # fake data on population
    df_pop = pd.DataFrame({
             'Województwa\nVoivodships': ['Mazowieckie', 'Lubelskie', None],
             'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, 2000000, None],
             'Irrelevant_pop': [25235, 23425345, None]
         })

    # drop one of the relevant columns
    df = dfs[missing_df_index].drop(columns=[missing_column])
    dfs[missing_df_index] = df

    with pytest.raises(ValueError, match='Missing required column'):
        dp.relevant_data_prep_and_rename(df_pop, *dfs, territory_level='v')

@pytest.mark.parametrize(
    'territory_lvl, df_pop',
    [
        ('v',
         pd.DataFrame({
             'Województwa\nVoivodships': ['Mazowieckie', 'Lubelskie', None],
             'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, 2000000, None],
             'Irrelevant_pop': [25235, 23425345, None]
         })
         ),
        ('p',
         pd.DataFrame({
            'Województwa \nVoivodships\nPowiaty\nPowiats':['Otwocki', 'WOJ. MAZOWIECKIE'],
            'Identyfikator terytorialny\nCode':[123, 7652],
            'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)':[1000000, 2345]
        })
         ),
        ('g',
         pd.DataFrame({
            'Województwa\nVoivodships\nGminy\nGminas':['Otwock', 'Otwock'],
            'Identyfikator terytorialny\nCode':['1234', '1234'],
            'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)':[1000000, 4444444]
        })
         )
    ]
)
def test_relevant_data_prep_and_rename_correct_renaming(territory_lvl, df_pop):
    """ Tests if function renames the columns properly """
    dfs = [
        df_pop,
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

    df_p, df_a, df_alc, df_f = dp.relevant_data_prep_and_rename(*dfs, territory_lvl)

    # check that renaming occurred
    assert 'Population' in df_p.columns
    assert 'Territory code' in df_a.columns
    assert 'Unit name' in df_a.columns
    assert 'Total number of fires' in df_f.columns
    if territory_lvl == 'v':
        assert 'Permit number' in df_alc.columns

@pytest.mark.parametrize(
    'territory_lvl, df_pop',
    [
        ('v',
         pd.DataFrame({
             'Województwa\nVoivodships': ['Mazowieckie', 'Lubelskie', None],
             'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': [1000000, 2000000, None],
             'Irrelevant_pop': [25235, 23425345, None]
         })
         ),
        ('p',
         pd.DataFrame({
            'Województwa \nVoivodships\nPowiaty\nPowiats':['Otwocki', 'WOJ. MAZOWIECKIE'],
            'Identyfikator terytorialny\nCode':[123, 7652],
            'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)':[1000000, 2345],
            'Irrelevant_pop': [25235, 23425345]
        })
         ),
        ('g',
         pd.DataFrame({
            'Województwa\nVoivodships\nGminy\nGminas':['Otwock', 'Otwock'],
            'Identyfikator terytorialny\nCode':['1234', '1234'],
            'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)':[1000000, 4444444],
            'Irrelevant_pop': [25235, 23425345]
        })
         )
    ]
)
def test_relevant_data_prep_and_rename_irrelevant_columns(territory_lvl, df_pop):
    """ Tests if function drops the irrelevant columns """
    dfs = [
        df_pop,
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

    df_p, df_a, df_alc, df_f = dp.relevant_data_prep_and_rename(*dfs, territory_lvl)

    # check that the irrelevant columns were dropped
    assert 'Irrelevant_pop' not in df_p.columns
    assert 'Irrelevant_are' not in df_a.columns
    assert 'Area [ha]' not in df_a.columns
    assert 'Irrelevant_alc' not in df_alc.columns
    assert 'Irrelevant_fir' not in df_f.columns

@pytest.mark.parametrize(
    'territory_lvl, df_pop',
    [
        ('p',
         pd.DataFrame({
            'Województwa \nVoivodships\nPowiaty\nPowiats':['Otwocki', 'WOJ. MAZOWIECKIE'],
            'Identyfikator terytorialny\nCode':[123, 7652],
            'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)':[1000000, 2345]
        })
         ),
        ('g',
         pd.DataFrame({
            'Województwa\nVoivodships\nGminy\nGminas':['Otwock', 'Otwock'],
            'Identyfikator terytorialny\nCode':['1234', '1234'],
            'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)':[1000000, 4444444]
        })
         )
    ]
)
def test_relevant_data_prep_and_rename_empty_alc(territory_lvl, df_pop):
    """ Tests if alcohol DataFrame is returned empty for p and g territory levels """
    dfs = [
        df_pop,
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

    df_p, df_a, df_alc, df_f = dp.relevant_data_prep_and_rename(*dfs, territory_lvl)

    assert df_alc.empty


# by_voivodship()

def test_by_voivodship_correct_columns():
    """ Tests if the returned DataFrames have proper columns and indexes """
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
            'Powiat': ['X', 'Y', 'Z', 'A'],
            'Gmina': ['A', 'B', 'C', 'D'],
            'Total number of fires': [10, 20, 30, 40]
        })
    ]

    df_voi, df_pop, df_are, df_alc, df_fir, nulls = dp.by_voivodship(*dfs)

    # correct index
    assert 'Voivodship' == df_voi.index.name
    assert 'Voivodship' == df_are.index.name
    assert 'Voivodship' == df_alc.index.name
    assert 'Voivodship' == df_fir.index.name
    # correct columns in merged df
    assert 'Territory code' in df_voi.columns
    assert 'Population' in df_voi.columns
    assert 'Area [km2]' in df_voi.columns
    assert 'Total number of fires' in df_voi.columns
    assert 'Total number of alcohol permits' in df_voi.columns
    # correct columns in the changed dfs
    assert 'Territory code' in df_are.columns
    assert 'Area [km2]' in df_are.columns
    assert 'Total number of alcohol permits' in df_alc.columns
    assert 'Total number of fires' in df_fir.columns

def test_by_voivodship_correct_values():
    """ Tests if the merged DataFrame has properly aggregated values """
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
            'Gmina': ['A', 'B', 'C', 'D'],
            'Total number of fires': [10, 20, 30, 40]
        })
    ]

    df_voi, df_pop, df_are, df_alc, df_fir, nulls = dp.by_voivodship(*dfs)

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
        (2, 'Permit number'),
        (2, 'Voivodship'),
    ]
)
def test_by_voivodship_missing_columns(missing_df_index, missing_column):
    """ Tests if the function raises and appropriate error when required columns
    are missing in population or alcohol DataFrames """

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
            'Gmina': ['A', 'B', 'C', 'D'],
            'Total number of fires': [10, 20, 30, 40]
        })
    ]

    df = dfs[missing_df_index].drop(columns=[missing_column])
    dfs[missing_df_index] = df

    with pytest.raises(ValueError, match='Missing required column'):
        dp.by_voivodship(*dfs)


# by_powiat()

def test_by_powiat_correct_columns():
    """ Tests if the returned DataFrames have proper columns and indexes """
    dfs = [
        pd.DataFrame({
            'Powiat': ['Xxxx', 'Yyyy'],
            'Territory code': [1, 2],
            'Population': [1000, 2000]
        }),
        pd.DataFrame({
            'Territory code': [1, 2],
            'Unit name': ['Powiat Xxxx', 'Powiat Yyyy'],
            'Area [km2]': [10000, 5000]
        }),
        pd.DataFrame({
            'Territory code': [1, 2],
            'Voivodship': ['X', 'Y'],
            'Powiat': ['Xxxx', 'Yyyy'],
            'Gmina': ['A', 'B'],
            'Total number of fires': [10, 20]
        })
    ]

    df_pow, df_pop, df_are, df_fir, nulls = dp.by_powiat(*dfs)

    # correct columns in merged df
    assert 'Territory code' in df_pow.columns
    assert 'Powiat' in df_pow.columns
    assert 'Population' in df_pow.columns
    assert 'Area [km2]' in df_pow.columns
    assert 'Total number of fires' in df_pow.columns
    # correct columns in the changed dfs
    assert 'Powiat' in df_are.columns
    assert 'Area [km2]' in df_are.columns
    assert 'Total number of fires' in df_fir.columns
    assert 'Gmina' not in df_fir.columns
    assert 'Territory code' in df_pop.columns
    assert 'Territory code' in df_are.columns
    assert 'Territory code' in df_fir.columns

def test_by_powiat_correct_values():
    """ Tests if the merged DataFrame has properly aggregated fires number values """
    dfs = [
        pd.DataFrame({
            'Powiat': ['Xxxx', 'Yyyy'],
            'Territory code': [123, 234],
            'Population': [1000, 2000]
        }),
        pd.DataFrame({
            'Territory code': [123, 234],
            'Unit name': ['Powiat Xxxx', 'Powiat Yyyy'],
            'Area [km2]': [10000, 5000]
        }),
        pd.DataFrame({
            'Territory code': [12300, 23400, 23400],
            'Voivodship': ['X', 'Y', 'Y'],
            'Powiat': ['Xxxx', 'Yyyy', 'Yyyy'],
            'Gmina': ['A', 'B', 'C'],
            'Total number of fires': [10, 20, 30]
        })
    ]

    df_pow, df_pop, df_are, df_fir, nulls = dp.by_powiat(*dfs)

    print(df_pow[df_pow['Territory code'] == 123])

    assert df_pow.shape[0] == 2
    assert df_pow.shape[1] == 5
    assert df_pow.loc[df_pow['Territory code'] == 123, 'Total number of fires'].iloc[0] == 10
    assert df_pow.loc[df_pow['Territory code'] == 234, 'Total number of fires'].iloc[0] == 50

@pytest.mark.parametrize('missing_column', ['Powiat', 'Territory code', 'Population'])
def test_by_powiat_missing_columns(missing_column):
    """ Tests if the function raises and appropriate error when required columns
    are missing in population or alcohol DataFrames """

    dfs = [
        pd.DataFrame({
            'Powiat': ['Otwock', 'Xxxx'],
            'Territory code': [1, 2],
            'Population': [10000, 20000]
        }),
        pd.DataFrame({
            'Territory code': [1, 2],
            'Unit name': ['WOJ. MAZOWIECKIE', 'WOJ. LUBELSKIE'],
            'Area [km2]': [10000, 5000]
        }),
        pd.DataFrame({
            'Territory code': [1, 2, 3, 4],
            'Voivodship': ['mazowieckie', 'mazowieckie', 'lubelskie', 'lubelskie'],
            'Powiat': ['X', 'Y', 'Z', 'A'],
            'Gmina': ['A', 'B', 'C', 'D'],
            'Total number of fires': [10, 20, 30, 40]
        })
    ]

    dfs[0] = dfs[0].drop(columns=[missing_column])

    with pytest.raises(ValueError, match='Missing required column'):
        dp.by_powiat(*dfs)


# by_gmina()

def test_by_gmina_correct_columns():
    """ Tests if the returned DataFrames have proper columns and indexes """
    dfs = [
        pd.DataFrame({
            'Gmina': ['Xxxx', 'Yyyy'],
            'Territory code': [1, 2],
            'Population': [1000, 2000]
        }),
        pd.DataFrame({
            'Territory code': [10, 20, 21],
            'Unit name': ['Xxxx', 'Yyyy', 'Yyyy2'],
            'Area [km2]': [10000, 5000, 4000]
        }),
        pd.DataFrame({
            'Territory code': [1, 2],
            'Voivodship': ['X', 'Y'],
            'Powiat': ['Xxxx', 'Yyyy'],
            'Gmina': ['A', 'B'],
            'Total number of fires': [10, 20]
        })
    ]

    df_gm, df_pop, df_are, df_fir, nulls = dp.by_gmina(*dfs)

    # correct columns in merged df
    assert 'Territory code' in df_gm.columns
    assert 'Gmina' in df_gm.columns
    assert 'Population' in df_gm.columns
    assert 'Area [km2]' in df_gm.columns
    assert 'Total number of fires' in df_gm.columns
    # correct columns in the changed dfs
    assert 'Gmina' in df_are.columns
    assert 'Area [km2]' in df_are.columns
    assert 'Total number of fires' in df_fir.columns
    assert 'Gmina' in df_fir.columns
    assert 'Territory code' in df_pop.columns
    assert 'Territory code' in df_are.columns
    assert 'Territory code' in df_fir.columns

def test_by_gmina_correct_values():
    """ Tests if the merged DataFrame has properly aggregated area data """
    dfs = [
        pd.DataFrame({
            'Gmina': ['Xxxx', 'Yyyy'],
            'Territory code': [1000, 2000],
            'Population': [1000, 2000]
        }),
        pd.DataFrame({
            'Territory code': [10005, 20005, 20006],
            'Unit name': ['Xxxx', 'Yyyy', 'Yyyy2'],
            'Area [km2]': [10000, 5000, 4000]
        }),
        pd.DataFrame({
            'Territory code': [1000, 2000],
            'Voivodship': ['X', 'Y'],
            'Powiat': ['Xxxx', 'Yyyy'],
            'Gmina': ['A', 'B'],
            'Total number of fires': [20, 30]
        })
    ]

    df_gm, df_pop, df_are, df_fir, nulls = dp.by_gmina(*dfs)

    assert df_gm.shape[0] == 2
    assert df_gm.shape[1] == 5
    assert df_gm.loc[df_gm['Territory code'] == 1000, 'Area [km2]'].iloc[0] == 10000
    assert df_gm.loc[df_gm['Territory code'] == 2000, 'Area [km2]'].iloc[0] == 5000

@pytest.mark.parametrize('missing_column', ['Gmina', 'Territory code', 'Population'])
def test_by_gmina_missing_columns(missing_column):
    """ Tests if the function raises and appropriate error when required columns
    are missing in population or alcohol DataFrames """

    dfs = [
        pd.DataFrame({
            'Gmina': ['Xxxx', 'Yyyy'],
            'Territory code': [1, 2],
            'Population': [10000, 20000]
        }),
        pd.DataFrame({
            'Territory code': [1, 2],
            'Unit name': ['Xxxx', 'Yyyy'],
            'Area [km2]': [10000, 5000]
        }),
        pd.DataFrame({
            'Territory code': [1, 2, 2],
            'Voivodship': ['X', 'Y', 'Y'],
            'Powiat': ['A', 'B', 'B'],
            'Gmina': ['Xxxx', 'Yyyy', 'Yyyy2'],
            'Total number of fires': [10, 20, 30]
        })
    ]

    dfs[0] = dfs[0].drop(columns=[missing_column])

    with pytest.raises(ValueError, match='Missing required column'):
        dp.by_gmina(*dfs)