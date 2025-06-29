Data Analysis Package
================================

This is a package made for analysing data regarding population, fire events, alcohol stores (permits) and area of territory units in Poland. Analysis can be performed at three territory levels: voivodship, powiat, and gmina. Alcohol permits data avaiable only at the voivodship level.

Features:
---------

- Loads and merges data from a provided folder (data from official Polish public datasets)
- Computes basic descriptive statistics
- Generates bar plots and normalized comparisons
- Performs correlation analysis between variables
- Designed as a pip-installable package with CLI support and notebook demos

Data sources:
------------

- population: https://stat.gov.pl/obszary-tematyczne/ludnosc/ludnosc/ludnosc-stan-i-struktura-ludnosci-oraz-ruch-naturalny-w-przekroju-terytorialnym-w-2024-r-stan-w-dniu-31-12,6,38.html
- fire events: https://dane.gov.pl/pl/dataset/4695/resource/64722/table?page=1&per_page=20&q=&sort=
- alcohol permits: https://dane.gov.pl/pl/dataset/1191,informacja-o-przedsiebiorcach-posiadajacych-zezwolenia-na-handel-hurtowy-napojami-alkoholowymi-1/resource/64402/table?page=1&per_page=20&q=&sort=
- area (voivodship sizes): https://dane.gov.pl/pl/dataset/1447,oficjalny-wykaz-pol-powierzchni-geodezyjnych-wojewodztw-powiatow-i-gmin/resource/66239/table

Usage
-----

You can use the package in two ways:

1. As a command-line script (via ``run_analysis.py``):

.. code-block:: bash

   python scripts/run_analysis.py data output output_file.txt -t v

This will load input files from the ``data`` directory and save results (plots, stats) to ``output``. Results of the correlation tests 
will be saved in ``output_file.txt``. Analysis will be performed on the voivodship level (as indicated by ``v``).

2. Interactively through a Jupyter notebook (``notebooks/final_project.ipynb``).


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