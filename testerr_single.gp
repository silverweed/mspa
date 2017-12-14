set term pdf enhanced
set output "rel/graphs/testerr_single.pdf"
set style line 1 lw 3
set title "Binary Digit Classifiers - Test Error"
set xlabel "T (number of weak learners)"
set ylabel "Test Error"
plot	"results/0_testerr.dat" title "0" w l lc rgb '#800000',\
	"results/1_testerr.dat" title "1" w l lc rgb '#b22222',\
	"results/2_testerr.dat" title "2" w l lc rgb '#dc143c',\
	"results/3_testerr.dat" title "3" w l lc rgb '#ff4500',\
	"results/4_testerr.dat" title "4" w l lc rgb '#ffa500',\
	"results/5_testerr.dat" title "5" w l lc rgb '#7cfc00',\
	"results/6_testerr.dat" title "6" w l lc rgb '#20b2aa',\
	"results/7_testerr.dat" title "7" w l lc rgb '#1e90ff',\
	"results/8_testerr.dat" title "8" w l lc rgb '#00008b',\
	"results/9_testerr.dat" title "9" w l lc rgb '#9400d3'
