# konrad plots

A module that helps to plot `konrad` output and make comparisons between experiments.

## Description

The module consists of:

1. String constants: That define units, quantity simbols and axis labels
2. Operative dictionaries: Dictionaries that define the necessary fields (and how to obtain them) to calculate a given derived quantity, the recipe to perform the calculation and the symbols associated with the given derived quantity.
3. Extraction functions: Functions to get the actual derived quantities from the output files.
4. Plotting functions: Functions tailored to graph the derived quantities extracted from the `konrad` output files.

## Dependencies

To this moment, this module depends on numpy, netCDF4, matplotlib and typhon modules.

## What is implemented and in what you can contribute?

Until now, you can plot temperature, lapse-rate and radiative fluxes (except net radiative flux). If you want to contribute in this, you can just focus on the first and second sections of the module. If you want a certain derived quantity, just define its units (if not already defined), its symbol, the difference symbol and the axis labelling symbols following the logic of the already-defined string constants. Afterwars you will need to define the ingredients to calculate the quantity and assign a name for your quantity in the appropriate dictionary of section 2, as well as defining the recipe, following the logic of the already defined recipes. You should also register the axis labels of your quantities. Do not forget to test

For contributing to sections 3 and 4 you will need to be more thoughtful. It is possible that the already-defined extraction functions are enough for most of the purpouses of comparing final equilibrium states. Thus, the extraction functions will only change if you want a radically different plot. Thus, you should first think what you want to plot, then ask why the already-defined plotting functions do not make the work. Afterwards, you can design your new plotting function following the ideas of the present plotting functions. Once there, you can think about new extraction function(s).
