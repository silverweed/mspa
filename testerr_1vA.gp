set term pdf enhanced
set output "rel/graphs/testerr_1vA.pdf"
set style line 1 lw 3
set title "One vs All Classifier - Test Error"
set xlabel "T (number of weak learners)"
set ylabel "Test Error"
plot "./results/1vA_testerr.dat" title "1 vs All" w l
