import cProfile
import pstats
import argparse

args = argparse.Namespace(
    data_folder='data',
    output_folder='output',
    output_file='output.txt',
    territory_level='g'
)

cProfile.runctx('main(args)', globals(), locals(), 'profile.prof')

p = pstats.Stats('profile.prof')
p.strip_dirs().sort_stats('cumtime').print_stats(30)

# python profile_run_analysis.py
# snakeviz profile.prof
