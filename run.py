from solvers_and_plots import *

plot_animation = 1
plot_time      = 0
matrix_debug   = 0
produce_results= 0

if plot_animation:
    plot_all_fields(41, 41, 200)
if plot_time:
    time_exitation(41, 41, 1000)
if matrix_debug:
    debug_matrix()
if produce_results:
    plot_evolution_and_time(41, 41, 1000)



