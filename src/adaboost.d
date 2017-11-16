module adaboost;

import std.math;
import std.typecons;
import std.conv;
import std.algorithm;
import std.algorithm.sorting;
version (parallel) import std.parallelism;
debug import std.stdio;
import loss_function;
import weak_classifier;
import data;

// Create all the base pixel classifiers.
auto create_hs(uint nRows, uint nCols) {
	auto hs = new image_weak_classifier_t[nRows * nCols];
	// hs[i] classifies pixel i.
	for (int i = 0; i < hs.length; ++i)
		hs[i] = ((i) => (in Image x) => bw_classifier(x[i]))(i);
	return hs;
}

// NUM = number to recognize
auto adaboost(ubyte NUM, uint T)(in Image[] xs, in ubyte[] ys_in)
in {
	static assert(NUM >= 0 && NUM <= 9);
	assert(xs.length == ys_in.length);
}
body {
	auto hs = create_hs(28, 28);
	assert(hs.length >= T);

	// Convert input labels [0-9] -> {-1, 1} (is NUM or not?)
	byte[] ys = cast(byte[])(ys_in).dup;
	ys.each!((ref y) => y = 2 * (y == NUM) - 1);
	assert(all!"a == 1 || a == -1"(ys));

	immutable m = xs.length;
	auto p = new float[T][](m);
	auto w = new float[T];
	auto h = new image_weak_classifier_t[T];
	alias l = loss_base;

	for (uint t = 0; t < m; ++t) {
		p[t][0] = 1.0 / m;
	}

	for (uint i = 0; i < T; ++i) {
		debug writeln("Iteration ", i, "/", T);

		// 1. choose h_i | e_i != 1/2
		auto res = choose_h!(NUM, T)(p, xs, ys, hs);
		h[i] = res[0];
		immutable e = res[1];
		debug writeln("e = ", e, "log((1-e)/e) = ", log((1 - e)/e));
		assert(0 < e && e < 1);

		// 2. compute weight
		w[i] = 0.5 * log((1 - e) / e);
		debug writeln("w[", i, "] = ", w[i]);

		// 3. Calculate P_(i+1)
		if (i < T - 1)
			calculate_next_p!(T, l)(p, h[i], i, xs, ys, w);
	}

	taskPool.stop();

	return (in Image x) {
		int sum = 0;
		for (uint i = 0; i < T; ++i) {
			debug writeln("w[", i, "] = ", w[i], ", h[i](x) = ", h[i](x));
			sum += w[i] * h[i](x);
		}
		debug writeln("Adaboost sum = ", sum);
		return sgn(sum);
	};
}

// Extract an h such that its epsilon != 1/2.
auto choose_h(ubyte NUM, uint T)(in float[T][] p, in Image[] xs, in byte[] ys, image_weak_classifier_t[] hs) {

	import std.algorithm.mutation;

	immutable m = xs.length;

	// FIXME?
	auto ok(float eps) {
		return abs(eps - 0.5) > 0.1;
	}

	// Find first good candidate
	int j = 0;
	float eps = 0;
	for ( ; j < hs.length; ++j) {
		eps = 0;
		for (uint t = 0; t < m; ++t) {
			assert(0 <= p[t][j] && p[t][j] <= 1, "p[" ~ to!string(t) ~ "][j] is " ~ to!string(p[t][j]));
			// Epsilon = sum{ P_j(t) * chi(L_j(t) != -1) }
			eps += p[t][j] * ((hs[j](xs[t]) * (2 * (ys[t] == NUM) - 1)) != -1);
		}
		assert(0 <= eps && eps <= 1);
		if (ok(eps))
			break;
	}

	if (j == hs.length)
		throw new Exception("Couldn't find a viable weak classifier!");

	const h = hs[j];
	hs = hs.remove(j);

	return tuple(h, eps);
}

void calculate_next_p(uint T, alias l)(float[T][] p, in image_weak_classifier_t h, uint i,
		in Image[] xs, in byte[] ys, in float[] w)
{
	immutable m = xs.length;
	auto E_i = 0;
	version (parallel) {
		foreach (t, pt; parallel(p)) {
			immutable l_i = l!h(xs[t], ys[t]);
			immutable ep = pt[i] * exp(-w[i] * l_i);
			E_i += ep;
			pt[i + 1] = ep;
		}
	} else {
		for (uint t = 0; t < m; ++t) {
			auto pt = p[t];
			immutable l_i = l!h(xs[t], ys[t]);
			immutable ep = pt[i] * exp(-w[i] * l_i);
			E_i += ep;
			pt[i + 1] = ep;
		}
	}
	version (parallel) {
		foreach (t, pt; parallel(p))
			pt[i + 1] /= E_i;
	} else {
		for (uint t = 0; t < m; ++t)
			p[t][i+1]/=E_i;
	}
}
