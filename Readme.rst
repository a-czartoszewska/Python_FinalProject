Data Analysis Package
================================

This is a package made for analysing data regarding population, fire events, alcohol stores (permits) and area of territory units in Poland. Analysis can be performed at three territory levels: voivodship, powiat, and gmina. Alcohol permits data avaiable only at the voivodship level.

Features:
---------

- Two modules for data preparation (`data_prep.py`) and statistics (`statistics_functions.py`)
- Loads and merges data from a provided folder (data from official Polish public datasets)
- Computes basic descriptive statistics
- Generates barplots and normalized barplot for comparison (only at voivodship level)
- Performs correlation analysis between variables
- Designed as a pip-installable package
- Can be used in command-line (CLI) or in a Jupyter notebook
- Includes unit tests (`test_data_prep.py`, `test_statistics_functions.py`)
- Including profiling script (`profile_run_analysis.py`) and results of profilig (`profile.prof`)

Data sources:
------------

- Population: https://stat.gov.pl/obszary-tematyczne/ludnosc/ludnosc/ludnosc-stan-i-struktura-ludnosci-oraz-ruch-naturalny-w-przekroju-terytorialnym-w-2024-r-stan-w-dniu-31-12,6,38.html
- Fire events: https://dane.gov.pl/pl/dataset/4695/resource/64722/table?page=1&per_page=20&q=&sort=
- Alcohol permits: https://dane.gov.pl/pl/dataset/1191,informacja-o-przedsiebiorcach-posiadajacych-zezwolenia-na-handel-hurtowy-napojami-alkoholowymi-1/resource/64402/table?page=1&per_page=20&q=&sort=
- Area (territory units' sizes): https://dane.gov.pl/pl/dataset/1447,oficjalny-wykaz-pol-powierzchni-geodezyjnych-wojewodztw-powiatow-i-gmin/resource/66239/table

Usage
-----

You can use the package in two ways:

1. As a command-line script (via ``run_analysis.py``, after importing the package in python):

.. code-block:: bash

   run-analysis data output output_file.txt -t v

This will load input files from the ``data`` directory and save results (plots, stats) to ``output``. Results of the correlation tests will be saved in ``output_file.txt``. Analysis will be performed on the voivodship level (as indicated by ``v``, other possible values: ``p`` for powiat and ``g`` for gmina).

2. Interactively through a Jupyter notebook (``final_project.ipynb``).


Requirements
------------

- pandas
- matplotlib.pyplot
- numpy
- scipy.stats
- openpyxl
- xlrd
- lxml
- seaborn
