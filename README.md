# Mountains-30x30-code

The scripts in this folder enable the generation of prioritization maps for new protected areas in order to expand the national protected area coverage of the included countries to 30%. The conservation factors taken into account are biodiversity, carbon, water, tourism, human impact, and ecoregion representativeness.  

There are 4 python files:

- main_3030.py: the file to execute to run the entire program

- preprocessing: formatting of data used as input

- reference_values: this script generates the best and worst possible values ​​for each variable to optimize, for the reference point method used. 
We individually optimize each model variable under the same parameters and constraints.

- ilp: the Integer Linear Programming model used for prioritization.
