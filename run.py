from solvers_and_plots import *

dirichlet       = 0
flow            = 0

plot_animation  = 0
plot_time       = 0
matrix_debug    = 0
produce_results = 1

if dirichlet:
    conds = 'dirichlet'
else:
    conds = 'neumann'
if flow:
    conds = 'flow'

if plot_animation:
    plot_all_fields(41, 41, 200, conds)
if plot_time:
    time_exitation(41, 41, 1000, conds)
if matrix_debug:
    debug_matrix()
if produce_results:
    produce_evolutions_and_time(41, 41, 1000, conds)



