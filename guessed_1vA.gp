set term pdf enhanced
set output "rel/graphs/guessed_1vA.pdf"
set style line 1 lw 3
set title "One vs All Classifier - % of Guesses"
set xlabel "T (number of weak learners)"
set ylabel "% Guessed"
plot "./results/1vA_guessed.dat" title "1 vs All" w l
