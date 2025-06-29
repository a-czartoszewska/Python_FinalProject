""" A module for data preparation for analysis.
For data related to alcohol permits, fire events, population, and area.

Functions:
    - load_data: Loads raw datasets from a specified folder.
    - data_inspection: Prints basic information and missing values from a DataFrame.
    - relevant_data_prep_and_rename: Selects relevant columns and standardizes naming.
    - by_voivodship: Aggregates datasets by voivodship and merges into a unified DataFrame.
"""

import os
import pandas as pd


def load_data(folder_path, file_list, population_rows_skip=[0, 1, 2], area_usecols=[0, 1, 2, 3]):
    """ Function loading all the data files from folder 'folder_path' and returning data frames.

    Args:
        folder_path (str): Path to the folder containing the data files.
        file_list (list[str]): List of file names to load.
            files containing data about: population, area, alcohol permits, fire events.
        territory_level (str, optional): Administrative level of the population data.
            Possible values: 'woj' (default), 'pow', 'gm'. Only 'woj' is currently supported.
        population_rows_skip (list[int], optional): List of rows to skip when loading the population data.
        area_usecols (list[int], optional): List of column numbers to use when loading the area data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        DataFrames for population, area, alcohol permits, and fire events.

    Raises:
        FileNotFoundError: If any file from file_list does not exist.
        ValueError: If any of the files from the file_list is in a wrong format.
    """

    # Raising Errors

    if len(file_list) != 4:
        raise ValueError('file_list must contain exactly 4 filenames')

    for file in file_list:
        if not os.path.exists(os.path.join(folder_path, file)):
            raise FileNotFoundError(f'File {file} not found. Path: {os.path.join(folder_path, file)}')

    if not os.path.join(folder_path, file_list[0]).endswith('.xls'):
        raise ValueError('File format not supported. Please provide a .xls file with population data.')

    if not os.path.join(folder_path, file_list[1]).endswith('.xlsx'):
        raise ValueError('File format not supported. Please provide a .xlsx file with area data.')

    if not os.path.join(folder_path, file_list[2]).endswith('.csv'):
        raise ValueError('File format not supported. Please provide a .csv file with alcohol permits data.')

    if not os.path.join(folder_path, file_list[3]).endswith('.csv'):
        raise ValueError('File format not supported. Please provide a .csv file with fire events data.')

    # paths to the data
    data_population_path = os.path.join(folder_path, file_list[0])
    data_area_path = os.path.join(folder_path, file_list[1])
    data_alcohol_stores_path = os.path.join(folder_path, file_list[2])
    data_fire_events_path = os.path.join(folder_path, file_list[3])

    # loading files and saving dataframes
    df_population_all = pd.read_excel(data_population_path, skiprows=population_rows_skip)
    df_area_all = pd.read_excel(data_area_path, usecols=area_usecols)
    df_alcohol_all = pd.read_csv(data_alcohol_stores_path)
    df_fire_events_all = pd.read_csv(data_fire_events_path)

    dfs = {
        'population data': df_population_all,
        'area data': df_area_all,
        'alcohol data': df_alcohol_all,
        'fire events data':df_fire_events_all
    }

    for name, df in dfs.items():
        if df.empty:
            raise ValueError(f"{name} is empty. Please provide a valid file.")

    return df_population_all, df_area_all, df_alcohol_all, df_fire_events_all


def data_inspection(df):
    """ Function printing basic information about the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to inspect.

    Prints:
        - First 10 rows,
        - Shape,
        - Column names,
        - Number of rows with missing values,
        - Rows with missing data (if any).
    """

    if df.empty:
        raise ValueError('DataFrame is empty, no data to inspect.')


    print('-------Data inspection------- \n\n1. First 10 rows: ')
    print(df.head(10))
    print('\n2. Data frame shape: ')
    print(df.shape)
    print('\n3. Columns: ')
    print(df.columns)

    nulls = df[df.isnull().any(axis=1)]
    print('\n4. Number of rows with missing data: ')
    print(nulls.shape[0])
    if nulls.shape[0] > 0:
        print('\n5. Rows with missing data: ')
        print(nulls)

def relevant_data_prep_and_rename_pop_voi(df_population_all):
    """ Function returning dataframe with renamed columns with data regarding population
    on a voivodship level, relevant for the analysis.

    Args:
        df_population_all (pd.DataFrame): Raw population data.

    Returns:
        pd.DataFrame:
        Cleaned DataFrame with population data.

    Raises:
        ValueError: If any of the relevant columns do not exist in provided DataFrame.
    """

    pop_relevant = ['Województwa\nVoivodships',
                    'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)']

    for col in pop_relevant:
        if col not in df_population_all.columns:
            raise ValueError(f'A relevant column {col} is not present in the dataframe. '
                             f'Please provide a valid dataframe.')

    df_population = df_population_all[pop_relevant].copy()

    df_population.rename(
        columns={'Województwa\nVoivodships': 'Voivodship',
                 'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': 'Population'},
        inplace=True)

    return df_population


def relevant_data_prep_and_rename_pop_pow(df_population_all):
    """ Function returning dataframe with renamed columns with data regarding population
    on a powiat level, relevant for the analysis.

    Args:
        df_population_all (pd.DataFrame): Raw population data.

    Returns:
        pd.DataFrame:
        Cleaned DataFrame with population data.

    Raises:
        ValueError: If any of the relevant columns do not exist in provided DataFrame.
    """
    pop_relevant = ['Województwa \nVoivodships\nPowiaty\nPowiats',
                    'Identyfikator terytorialny\nCode',
                    'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)']

    for col in pop_relevant:
        if col not in df_population_all.columns:
            raise ValueError(f'A relevant column {col} is not present in the dataframe. '
                             f'Please provide a valid dataframe.')

    df_population = df_population_all[pop_relevant].copy()

    df_population.rename(
        columns={'Województwa \nVoivodships\nPowiaty\nPowiats': 'Powiat',
                 'Identyfikator terytorialny\nCode': 'Territory code',
                 'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': 'Population'},
        inplace=True)

    df_population = df_population[pd.notna(df_population['Territory code'])]
    df_population = df_population[~df_population['Powiat'].str.startswith('WOJ.')]

    return df_population

def relevant_data_prep_and_rename_pop_gm(df_population_all):
    """ Function returning dataframe with renamed columns with data regarding population
    on a gmina level, relevant for the analysis.

    Args:
        df_population_all (pd.DataFrame): Raw population data.

    Returns:
        pd.DataFrame:
        Cleaned DataFrame with population data.

    Raises:
        ValueError: If any of the relevant columns do not exist in provided DataFrame.
    """
    pop_relevant = ['Województwa\nVoivodships\nGminy\nGminas',
                    'Identyfikator terytorialny\nCode',
                    'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)']

    for col in pop_relevant:
        if col not in df_population_all.columns:
            raise ValueError(f'A relevant column {col} is not present in the dataframe. '
                             f'Please provide a valid dataframe.')

    df_population = df_population_all[pop_relevant].copy()

    df_population.rename(
        columns={'Województwa\nVoivodships\nGminy\nGminas': 'Gmina',
                 'Identyfikator terytorialny\nCode': 'Territory code',
                 'Ludność\n(stan w dniu 31.12)\nPopulation\n(as of \nDecember 31)': 'Population'},
        inplace=True)

    df_population = df_population[pd.notna(df_population['Territory code'])]
    df_population = df_population[~df_population['Gmina'].str.startswith('WOJ.')]
    df_population.loc[:, 'Territory code'] = df_population['Territory code'].astype(int) // 10

    ids = df_population.groupby('Territory code')['Population'].idxmax()
    df_population = df_population.loc[ids].reset_index(drop=True)

    return df_population

def relevant_data_prep_and_rename(df_population_all, df_area_all,
                                  df_alcohol_all, df_fire_events_all,
                                  territory_level='v'):
    """ Function returning dataframes with data relevant for the analysis,
     dropping NaNs when necessary, renaming columns

    Args:
        df_population_all (pd.DataFrame): Raw population data.
        df_area_all (pd.DataFrame): Raw area data.
        df_alcohol_all (pd.DataFrame): Raw alcohol permits data.
        df_fire_events_all (pd.DataFrame): Raw fire events data.
        territory_level (str, optional): Which territory level to use.
            Possible values: 'v' (voivodship), 'g' (gmina), 'p' (powiat). Default value is 'v'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Cleaned DataFrames: population, area, alcohol permits, and fire events.
        If territory level is 'g' or 'p', df with alcohol data is empty

    Raises:
        ValueError: If any of the relevant columns do not exist in provided DataFrames.
    """

    # check if the relevant columns are in the dataframes
    are_relevant = ['TERYT', 'Nazwa jednostki', 'Powierzchnia [km2]']
    alc_relevant = ['Numer zezwolenia', 'Województwo']
    fir_relevant = ['TERYT', 'Województwo', 'Powiat', 'Gmina', 'OGÓŁEM Liczba zdarzeń']

    relevant = [are_relevant, alc_relevant, fir_relevant]
    dfs = [df_area_all, df_alcohol_all, df_fire_events_all]

    for i, rel_list in enumerate(relevant):
        for col in rel_list:
            if col not in dfs[i].columns:
                raise ValueError(f'A relevant column {col} is not present in the dataframe. '
                                 f'Please provide a valid dataframe.')


    # relevant information from the dataset about population in the areas
    if territory_level == 'g':
        df_population = relevant_data_prep_and_rename_pop_gm(df_population_all)
    if territory_level == 'p':
        df_population = relevant_data_prep_and_rename_pop_pow(df_population_all)
    if territory_level == 'v':
        df_population = relevant_data_prep_and_rename_pop_voi(df_population_all)

    # relevant information from the dataset about area of territory units
    df_area = df_area_all[are_relevant].copy()
    df_area.rename(columns={'TERYT': 'Territory code',
                            'Nazwa jednostki': 'Unit name',
                            'Powierzchnia [km2]': 'Area [km2]'},
                   inplace=True)
    df_area.loc[:, 'Territory code'] = df_area['Territory code'].astype(str).str.replace(' ', '').astype(int)

    # relevant information from alcohol permits dataset and renaming
    if territory_level == 'v':
        df_alcohol = df_alcohol_all[alc_relevant].copy()
        df_alcohol.rename(columns={'Numer zezwolenia': 'Permit number',
                                   'Województwo': 'Voivodship'},
                          inplace=True)
    else:
        df_alcohol = pd.DataFrame()

    # relevant information from fire events dataset and renaming
    df_fire_events = df_fire_events_all[fir_relevant].copy()
    df_fire_events.rename(
        columns={'TERYT': 'Territory code',
                 'Województwo': 'Voivodship',
                 'OGÓŁEM Liczba zdarzeń': 'Total number of fires'},
        inplace=True)


    return df_population, df_area, df_alcohol, df_fire_events


def by_voivodship(df_population, df_area, df_alcohol, df_fire_events):
    """ Function keeping only the information related to voivodships
    and concatenating it in one dataframe.
    Making the content of 'Voivodship' column consistent across dataframes.

    Args:
        df_population (pd.DataFrame): Cleaned population data.
        df_area (pd.DataFrame): Cleaned area data.
        df_alcohol (pd.DataFrame): Cleaned alcohol permits data.
        df_fire_events (pd.DataFrame): Cleaned fire events data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,  pd.DataFrame]:
        - Merged voivodship-level DataFrame,
        - Population per voivodship,
        - Area per voivodship,
        - Alcohol permits per voivodship,
        - Fire events per voivodship.
        - Rows of merged DataFrame with nulls, which are dropped.
    """
    # check if the relevant columns are in the dataframes
    pop_relevant = ['Voivodship', 'Population']
    are_relevant = ['Territory code', 'Unit name', 'Area [km2]']
    alc_relevant = ['Permit number', 'Voivodship']
    fir_relevant = ['Territory code', 'Voivodship', 'Powiat', 'Gmina', 'Total number of fires']

    relevant = [pop_relevant, are_relevant, alc_relevant, fir_relevant]
    dfs = [df_population, df_area, df_alcohol, df_fire_events]

    for i, rel_list in enumerate(relevant):
        for col in rel_list:
            if col not in dfs[i].columns:
                if col != dfs[i].index.name:
                    raise ValueError(f'A relevant column {col} is not present in the dataframe.'
                                     f'Please provide a valid dataframe.')


    # df population
    df_population.dropna(inplace=True)
    df_population_voi = df_population[df_population['Voivodship'].str.endswith('kie')].copy()
    df_population_voi = df_population_voi.groupby('Voivodship', as_index=False).max() # data not divided by city/country area
    df_population_voi['Voivodship'] = 'WOJ. ' + df_population_voi['Voivodship'].str.upper()
    df_population_voi.set_index('Voivodship', inplace=True)

    # df area
    df_area_voi = df_area[df_area['Unit name'].str.contains(r'WOJ. ')].copy()
    df_area_voi.rename(columns={'Unit name': 'Voivodship'}, inplace=True)
    df_area_voi.set_index('Voivodship', inplace=True)

    # df alcohol
    df_alcohol_voi = df_alcohol.groupby('Voivodship').count()
    df_alcohol_voi.rename(columns={'Permit number': 'Total number of alcohol permits'},
                          inplace=True)

    # df fire events
    df_fire_events_voi = (
        df_fire_events
        .groupby('Voivodship')['Total number of fires']
        .sum()
        .reset_index()
    )
    df_fire_events_voi['Voivodship'] = 'WOJ. ' + df_fire_events_voi['Voivodship'].str.upper()
    df_fire_events_voi.set_index('Voivodship', inplace=True)

    # merging into one df
    df_voi = pd.merge(df_area_voi, df_population_voi, on='Voivodship', how='outer')
    df_voi = pd.merge(df_voi, df_alcohol_voi, on='Voivodship', how='outer')
    df_voi = pd.merge(df_voi, df_fire_events_voi, on='Voivodship', how='outer')

    nulls = df_voi[df_voi.isnull().any(axis=1)]
    df_voi.dropna(inplace=True)

    return df_voi, df_population_voi, df_area_voi, df_alcohol_voi, df_fire_events_voi, nulls

def by_gmina(df_population, df_area, df_fire_events):
    """ Function keeping only the information related to gmins
    and concatenating it in one dataframe.
    Making the territory code format consistent across dataframes for merging.

    Args:
        df_population (pd.DataFrame): Cleaned population data.
        df_area (pd.DataFrame): Cleaned area data.
        df_fire_events (pd.DataFrame): Cleaned fire events data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        - Merged gmina-level DataFrame,
        - Population per gmina,
        - Area per gmina,
        - Fire events per gmina.
        - Rows of merged DataFrame with nulls, which are dropped.

    Raises:
        ValueError: If there are any relevant columns missing in the DataFrames.
    """
    # check if the relevant columns are in the dataframes
    pop_relevant = ['Gmina', 'Territory code', 'Population']
    are_relevant = ['Territory code', 'Unit name', 'Area [km2]']
    fir_relevant = ['Territory code', 'Voivodship', 'Powiat', 'Gmina', 'Total number of fires']

    relevant = [pop_relevant, are_relevant, fir_relevant]
    dfs = [df_population, df_area, df_fire_events]

    for i, rel_list in enumerate(relevant):
        for col in rel_list:
            if col not in dfs[i].columns:
                if col != dfs[i].index.name:
                    raise ValueError(f'A relevant column {col} is not present in the dataframe.'
                                     f'Please provide a valid dataframe.')

    # df population
    df_population_gm = df_population.copy()
    df_population.set_index('Territory code', inplace=True)

    # df area
    df_area_gm = df_area[df_area['Territory code'] > pow(10, 4)].copy()
    df_area_gm.rename(columns={'Unit name': 'Gmina'}, inplace=True)
    df_area_gm.loc[:, 'Territory code'] = df_area_gm['Territory code'] // 10
    ids = df_area_gm.groupby('Territory code')['Area [km2]'].idxmax()
    df_area_gm = df_area_gm.loc[ids].reset_index(drop=True)
    df_area_gm.set_index('Territory code', inplace=True)

    # df fire events
    df_fire_events_gm = df_fire_events.drop(columns=['Voivodship', 'Powiat'])
    df_fire_events_gm.set_index('Territory code', inplace=True)

    # merging into one df
    df_gm = pd.merge(df_area_gm, df_population_gm, on='Territory code', how='outer')
    df_gm = pd.merge(df_gm, df_fire_events_gm, on='Territory code', how='outer')

    nulls = df_gm[df_gm.isnull().any(axis=1)]
    df_gm.dropna(inplace=True)
    df_gm.drop(columns=['Gmina_x', 'Gmina_y'], inplace=True)

    return df_gm, df_population_gm, df_area_gm, df_fire_events_gm, nulls

def by_powiat(df_population, df_area, df_fire_events):
    """ Function keeping only the information related to powiats
    and concatenating it in one dataframe.
    Making the territory code format consistent across dataframes.

    Args:
        df_population (pd.DataFrame): Cleaned population data.
        df_area (pd.DataFrame): Cleaned area data.
        df_fire_events (pd.DataFrame): Cleaned fire events data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        - Merged powiat-level DataFrame,
        - Population per powiat,
        - Area per powiat,
        - Fire events per powiat.
        - Rows of merged DataFrame with nulls, which are dropped.

    Raises:
        ValueError: If there are any relevant columns missing in the DataFrames.
    """
    # check if the relevant columns are in the dataframes
    pop_relevant = ['Powiat', 'Territory code', 'Population']
    are_relevant = ['Territory code', 'Unit name', 'Area [km2]']
    fir_relevant = ['Territory code', 'Voivodship', 'Powiat', 'Gmina', 'Total number of fires']

    relevant = [pop_relevant, are_relevant, fir_relevant]
    dfs = [df_population, df_area, df_fire_events]

    for i, rel_list in enumerate(relevant):
        for col in rel_list:
            if col not in dfs[i].columns:
                if col != dfs[i].index.name:
                    raise ValueError(f'A relevant column {col} is not present in the dataframe.'
                                     f'Please provide a valid dataframe.')

    # df population
    df_population_pow = df_population.copy()
    df_population_pow.loc[:, 'Territory code'] = df_population_pow['Territory code'].astype(int)
    df_population.set_index('Territory code', inplace=True)

    # df area
    df_area_pow = df_area[df_area['Unit name'].str.startswith('Powiat')].copy()
    df_area_pow.rename(columns={'Unit name': 'Powiat'}, inplace=True)
    df_area_pow['Powiat'].str.split(' ').str[1]
    df_area_pow.set_index('Territory code', inplace=True)

    # df fire events
    df_fire_events_pow = df_fire_events.drop(columns=['Voivodship', 'Gmina'])
    df_fire_events_pow.loc[:, 'Territory code'] = df_fire_events_pow['Territory code'].astype(int)//100
    df_fire_events_pow = (df_fire_events_pow.groupby('Territory code')
                          .agg({'Powiat': 'min', 'Total number of fires': 'sum'}))

    # merging into one df
    df_pow = pd.merge(df_area_pow, df_population_pow, on='Territory code', how='outer')
    df_pow = pd.merge(df_pow, df_fire_events_pow, on='Territory code', how='outer')

    nulls = df_pow[df_pow.isnull().any(axis=1)]
    df_pow.dropna(inplace=True)
    df_pow.drop(columns=['Powiat_x', 'Powiat_y'], inplace=True)

    return df_pow, df_population_pow, df_area_pow, df_fire_events_pow, nulls

def by_area(df_population, df_area, df_alcohol, df_fire_events, territory_level):
    """ Function keeping only the information related to the area level specified
    and concatenating it in one dataframe.

    Args:
        df_population (pd.DataFrame): Cleaned population data.
        df_area (pd.DataFrame): Cleaned area data.
        df_alcohol (pd.DataFrame): Cleaned alcohol permits data.
        df_fire_events (pd.DataFrame): Cleaned fire events data.

    Returns:
        Aggregated DataFrames specific to the selected level.

    Raises:
        ValueError: If the territory level code is invalid.
    """

    if territory_level == 'v':
        return by_voivodship(df_population, df_area, df_alcohol, df_fire_events)
    if territory_level == 'p':
        return by_powiat(df_population, df_area, df_fire_events)
    if territory_level == 'g':
        return by_gmina(df_population, df_area, df_fire_events)

    raise ValueError(f'The territory level must be one of the following: v, p, g.')