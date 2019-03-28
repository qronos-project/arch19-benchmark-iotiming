set terminal png font arial 14 size 1000, 1000
set output "reachability.png"

set title "Test1"
set xlabel "Tau"
set ylabel "YCont"

load "ranges.gnuplot.txt"

plot \
   1/0 lw 4 lc rgb 'green' with lines t 'Sample Sample Always Tick Always', \
   'sample_sample_always_tick_always.gnuplot.txt' with points lc rgb 'green' pt 0 notitle,\
   1/0 lw 4 lc rgb 'blue' with lines t 'Sample Wait Always Tick Always', \
   'sample_wait_always_tick_always.gnuplot.txt' with points lc rgb 'blue' pt 0 notitle,\
   1/0 lw 4 lc rgb 'orange' with lines t 'Wait Sample Always Tick Always', \
   'wait_sample_always_tick_always.gnuplot.txt' with points lc rgb 'orange' pt 0 notitle,\
   1/0 lw 4 lc rgb 'light-grey' with lines t 'Wait Wait Always Tick Always', \
   'wait_wait_always_tick_always.gnuplot.txt' with points lc rgb 'light-grey' pt 0 notitle, \
   'simulation.txt' using 1:2 with lines lw 2 lc rgb 'magenta' t 'Simulation'

