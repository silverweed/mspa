set term pdf enhanced
set output "rel/graphs/guessed_single.pdf"
set style line 1 lw 3
set title "Binary Digit Classifiers - % of Guesses"
set xlabel "T (number of weak learners)"
set ylabel "% Guessed"
set key right bottom
plot	"results/0_guessed.dat" title "0" w l lc rgb '#800000',\
	"results/1_guessed.dat" title "1" w l lc rgb '#b22222',\
	"results/2_guessed.dat" title "2" w l lc rgb '#dc143c',\
	"results/3_guessed.dat" title "3" w l lc rgb '#ff4500',\
	"results/4_guessed.dat" title "4" w l lc rgb '#ffa500',\
	"results/5_guessed.dat" title "5" w l lc rgb '#7cfc00',\
	"results/6_guessed.dat" title "6" w l lc rgb '#20b2aa',\
	"results/7_guessed.dat" title "7" w l lc rgb '#1e90ff',\
	"results/8_guessed.dat" title "8" w l lc rgb '#00008b',\
	"results/9_guessed.dat" title "9" w l lc rgb '#9400d3'
