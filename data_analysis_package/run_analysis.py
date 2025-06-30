import argparse
import cProfile
import os
import pstats

from . import data_prep as dp
from . import statistics_functions as stf

def load_and_prep(args, data_files_list):

    # load data
    df_population_all, df_area_all, df_alcohol_all, df_fire_events_all = (
        dp.load_data(args.data_folder, data_files_list, pop_rows_skip=[0, 1, 2], area_usecols=[0, 1, 2, 3]))

    # choose relevant data in the datasets, drop NaNs when necessary and rename for consistency
    df_population, df_area, df_alcohol, df_fire_events = (
        dp.relevant_data_prep_and_rename(df_population_all, df_area_all, df_alcohol_all, df_fire_events_all,
                                         territory_level=args.territory_level))

    # aggregate all the data about the area
    territory_level = args.territory_level
    df_terr, *_ = dp.by_area(df_population, df_area, df_alcohol, df_fire_events, territory_level)

    return df_terr

def basic_stats_and_plots(args, df_terr):

    # basic statistics
    stat_table = stf.statistics_table(df_terr)
    stat_table.to_csv(os.path.join(args.output_folder, 'statistics_table_'+args.territory_level+'.csv'))
    if args.territory_level == 'v':
        # basic barplots
        stf.barplots(df_terr, cols=['Area [km2]', 'Population', 'Total number of alcohol permits', 'Total number of fires'],
                     output_folder=args.output_folder, show=False)

        # comparative normalised barplot
        stf.normalised_barplot(df_terr, output_folder=args.output_folder, show=False)

    # correlation plot
    if args.territory_level == 'v':
        cols = ['Total number of alcohol permits', 'Total number of fires', 'Area [km2]', 'Population']
    else:
        cols = ['Total number of fires', 'Area [km2]', 'Population']
    stf.corr_plot(df_terr, cols,
                  output_folder=args.output_folder,
                  filename='corr_'+args.territory_level+'.png',
                  show=False)

    # correlation plot for densities
    if args.territory_level == 'v':
        cols = ['Total number of alcohol permits', 'Total number of fires', 'Population']
    else:
        cols = ['Total number of fires', 'Population']

    stf.density_corr_plot(df_terr.copy(), cols,
                          area='Area [km2]',
                          output_folder=args.output_folder,
                          filename='density_corr_'+args.territory_level+'.png',
                          show=False)

def corr_analysis_voi(args, df_voi):
    # correlation analysis
    # The number of people living in an area and the number of fire events,

    stf.corr_test(df_voi['Population'], df_voi['Total number of fires'],
                  'Test results for correlation between population and numer of fire events',
                  args.output_folder, args.output_file, overwrite=True)

    # The number of people living in an area and the number of alcohol selling companies.
    stf.corr_test(df_voi['Population'], df_voi['Total number of alcohol permits'],
                  'Test results for correlation between population and numer of alcohol stores',
                  args.output_folder, args.output_file)

    # The number of alcohol selling companies and the number of fire events.
    stf.corr_test(df_voi['Total number of alcohol permits'], df_voi['Total number of fires'],
                  'Test results for correlation between number of alcohol stores and numer of fire events',
                  args.output_folder, args.output_file)

    # The density of fire events and alcohol stores.
    x = df_voi['Total number of alcohol permits'] / df_voi['Area [km2]']
    y = df_voi['Total number of fires'] / df_voi['Area [km2]']
    stf.corr_test(x, y, 'Test results for correlation between the density of fires and alcohol stores in a voivodship',
                  args.output_folder, args.output_file)
def corr_analysis_pow_gm(args, df_terr):

    # The number of people living in an area and the number of fire events,
    stf.corr_test(df_terr['Population'], df_terr['Total number of fires'],
                  'Test results for correlation between population and numer of fire events in a gmina',
                  args.output_folder, args.output_file, overwrite=True)

    # The number of fires and area of a gmina
    stf.corr_test(df_terr['Total number of fires'], df_terr['Area [km2]'],
                  'Test results for correlation between number of fire events and area of a gmina',
                  args.output_folder, args.output_file)

    # Density of population and fire events
    x = df_terr['Population'] / df_terr['Area [km2]']
    y = df_terr['Total number of fires'] / df_terr['Area [km2]']
    stf.corr_test(x, y,
                  'Test results for correlation between the density of population and fire events in a gmina',
                  args.output_folder, args.output_file)

def corr_analysis(args, df_terr):

    if args.territory_level == 'v':
        corr_analysis_voi(args, df_terr)
    else:
        corr_analysis_pow_gm(args, df_terr)

def run_analysis(args):
    # file list
    data_files_list = ['data_population_' + args.territory_level + '.xls', 'data_area.xlsx', 'data_alcohol_stores.csv',
                       'data_fire_events.csv']
    # get the merged dataset
    df_terr = load_and_prep(args, data_files_list)
    # create basic stats df and plots
    basic_stats_and_plots(args, df_terr)
    # perform corr analysis
    corr_analysis(args, df_terr)

def main(args):
    parser = argparse.ArgumentParser(description='Run final project analysis')
    parser.add_argument('data_folder', help='Path to the folder containing the data')
    parser.add_argument('output_folder', help='Path to the folder to save output plots and results')
    parser.add_argument('output_file', help='Filename to save correlation analysis results')
    parser.add_argument('-t', '--territory_level',
                        help='Territory level of the data. '
                             'Possible values: g (for gmina), p (for powiat), v (for voivodship).')
    args = parser.parse_args()

    run_analysis(args)

if __name__ == '__main__':
    main()