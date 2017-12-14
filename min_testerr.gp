set terminal pdf enhanced
set output "rel/graphs/min_testerr.pdf"
set title "Min Test Error By Digit"
set xlabel "Digit"
set ylabel "Test Error at T = 250"
set style line 1 lw 3
set key off
plot "results/min_testerr.dat" w linespoint
