set term pdf enhanced size 6,4
set output "rel/graphs/weights_single.pdf"
set style line 1 lw 3
set title "Binary Digit Classifiers - Weights for T = 250"
set xlabel "i (index of the weak learner)"
set ylabel "w_i (weight of the i^{th} weak learner)"
plot	"<(head -100 results/0_weights.dat)" title "0" w l lc rgb '#800000',\
	"<(head -100 results/1_weights.dat)" title "1" w l lc rgb '#b22222',\
	"<(head -100 results/2_weights.dat)" title "2" w l lc rgb '#dc143c',\
	"<(head -100 results/3_weights.dat)" title "3" w l lc rgb '#ff4500',\
	"<(head -100 results/4_weights.dat)" title "4" w l lc rgb '#ffa500',\
	"<(head -100 results/5_weights.dat)" title "5" w l lc rgb '#7cfc00',\
	"<(head -100 results/6_weights.dat)" title "6" w l lc rgb '#20b2aa',\
	"<(head -100 results/7_weights.dat)" title "7" w l lc rgb '#1e90ff',\
	"<(head -100 results/8_weights.dat)" title "8" w l lc rgb '#00008b',\
	"<(head -100 results/9_weights.dat)" title "9" w l lc rgb '#9400d3'
