# Mathematical modelling project

## Dependencies
Python 3.9 together with the following packages:
- `numpy`
- `matplotlib`
- `scipy`

Functionality is written in the file `solvers_and_plots.py`, and different parts can be run using the script `run.py`.
All that has to be done, is to modify the variables deciding which functionality to run.

## `run.py`
The switches at the top of the file allows control over which modelling case to simulate. The important options are listed, and `produce_results` makes the plots used in the report.

`dirichlet`<br>
- `1:` Dirichlet conditions (adding Glia cells to the system)
- `0:` Neumann conditions (artificial non-penetrable boundary)

`flow`<br>
- `1:` Introduces a velocity field to the intercellular fluid (sets `dirichlet=1`)
- `0:` No flow in the intercellular fluid (bdr conditions decided by `dirichlet` switch)

`plot_animation`<br>
- `1:` Animates the time evolutions of all involved fields
- `0:` No animation

`plot_time`<br>
- `1:` Plots the fraction av bound transmitters as a function of time
- `0:` No fraction plot

`produce_results`<br>
- `1:` Produce report-style plots based on above config. Saves the figures as `.pdf` files in the relative path `plots/<cond>/`, where `<cond>` is either `dirchlet`, `neumann`, or `flow`, based on the config.
- `0:` No report plotting
