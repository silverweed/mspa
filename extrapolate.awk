#!/usr/bin/awk -f

BEGIN {
	if (ARGC < 3) {
		print "Usage: " ARGV[0] " <T> <resultsfile>" > "/dev/stderr"
		do_exit = 1
		exit
	}
	T = ARGV[1]
	delete ARGV[1]
	cipher = -1
}

/^COEFF/ {
	if (t++ < T)
		print
}

/.* Coefficients/ {
	++cipher
	t = 0
	print
}

END {
	if (do_exit) exit
}
