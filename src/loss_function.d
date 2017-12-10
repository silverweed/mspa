module loss_function;

import std.algorithm;
import data;

auto loss01(H, X, Y)(in H h, in X x, in Y y) {
	return h(x) == y ? 0 : 1;
}

/// Given an algorithm, calculates its training/test error on the given set (xs, ys).
auto calcError(alias loss, Algo, Y)(in Algo algo, in Image[] xs, in Y[] ys) pure
in {
	assert(xs.length == ys.length);
}
out (result) {
	assert(result >= 0);
}
do {
	immutable n = xs.length;
	float_t err = 0;
	for (int i = 0; i < n; ++i) {
		err += loss(algo, xs[i], ys[i]);
	}
	return err / n;
}
