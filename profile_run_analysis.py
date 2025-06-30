import cProfile
import pstats
import argparse
import sys
from data_analysis_package import run_analysis

sys.argv = [
    'run_analysis',
    'data',
    'output',
    'output.txt',
    '-t', 'g'
]

cProfile.runctx('run_analysis.main()', globals(), locals(), 'profile.prof')

p = pstats.Stats('profile.prof')
p.strip_dirs().sort_stats('cumtime').print_stats(30)

# python profile_run_analysis.py
# snakeviz profile.prof
