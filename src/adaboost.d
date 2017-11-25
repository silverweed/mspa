module adaboost;

import std.math;
import std.typecons;
import std.conv;
import std.algorithm;
import std.algorithm.sorting;
version (parallel) import std.parallelism;
debug import std.stdio;
debug import std.format;
import loss_function;
import weak_classifier;
import data;

/// NUM = number to recognize
/// T = number of weak classifiers
auto adaboost(ubyte NUM, uint T)(in Image[] xs, in ubyte[] ys_in)
in {
	static assert(NUM >= 0 && NUM <= 9);
	static assert(T > 0);
	assert(xs.length == ys_in.length);
}
body {
	const ys = transform!NUM(ys_in);

	immutable m = xs.length;
	auto p = new float[T][](m);
	auto w = new float[T];
	auto h = new image_weak_classifier_t[T];

	// Initially set P_1(t) = 1/f for t = 1, ..., m
	for (uint t = 0; t < m; ++t)
		p[t][0] = 1.0 / m;

	// Adaboost main loop
	for (uint i = 0; i < T; ++i) {
		debug writeln("Iteration ", i, "/", T);

		// 1. choose h_i such that e_i is far from 1/2
		auto res = choose_h!T(p, i, xs, ys);
		h[i] = res[0];
		immutable eps = res[1];
		debug writeln("log((1-e)/e) = ", log((1 - eps)/eps));

		// 2. compute weight
		w[i] = 0.5 * log((1 - eps) / eps);
		debug writeln("w[", i, "] = ", w[i]);

		// 3. Calculate P_(i+1)
		if (i < T - 1)
			calc_next_p!T(p, i, h[i], xs, ys, w[i]);

		debug print_p_sum!T(p, i);
	}

	version (parallel) taskPool.stop();

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

/// Transform labels [0-9] into {-1, 1}
byte[] transform(uint NUM)(in ubyte[] ys) pure {
	import std.array;
	return array(ys.map!(y => cast(byte)(2 * (y == NUM) - 1)));
}

float calc_epsilon(uint T)(in Image[] xs, in byte[] ys, in float[T][] p, uint i, uint pixel, ubyte tau) pure
in {
	assert(xs.length == ys.length);
	assert(pixel < xs[0].length);
	assert(i < p[0].length);
}
out (epsilon) {
	debug writeln("eps = ", epsilon);
	assert(0 < epsilon && epsilon < 1.1);
}
body {
	immutable m = xs.length;
	const h = make_image_weak_classifier(pixel, tau);

	float eps = 0;
	for (int t = 0; t < m; ++t) {
		debug if (p[t][i].isNaN) writeln("p[%d][%d] = %f".format(t, i, p[t][i]));
		assert(!p[t][i].isNaN);
		eps += (loss(h, xs[t], ys[t]) == -1) * p[t][i];
	}
	//debug writeln("eps = ", eps);
	return eps;
}

// Extract an h such that its epsilon is as far from 1/2 as possible
auto choose_h(uint T)(in float[T][] p, uint i, in Image[] xs, in byte[] ys)
in {
	//static assert(0 <= NUM && NUM <= 9);
	assert(p.length == xs.length);
	assert(xs.length == ys.length);
	assert(i < p[0].length);
}
out (result) {
	assert(0 < result[1] && result[1] < 1.1);
}
body {
	import std.algorithm.mutation;

	immutable m = xs.length;

	immutable struct HParams {
		uint pixel;
		ubyte tau;
	}
	// FIXME
	static HParams[] chosen;

	// Loop over the weak classifier parameters p and tau.
	// Choose the combination of parameters which minimizes epsilon.
	uint pixel = 0;
	uint tau = 0; // using uint instead of ubyte for proper loop condition check
	HParams* bestPair = null;
	auto bestEps = 0.5;
	bool found = false;
	for ( ; pixel < xs[0].length; ++pixel) {
		for ( ; tau < 256; ++tau) {
			immutable thisPair = HParams(pixel, cast(byte)tau);

			if (chosen.canFind(thisPair))
				continue;

			const eps = calc_epsilon!T(xs, ys, p, i, thisPair.pixel, thisPair.tau);
			if (abs(eps - 0.5) > abs(bestEps - 0.5)) {
				found = true;
				bestPair = &thisPair;
				bestEps = eps;
			}
		}
	}

	if (!found)
		throw new Exception("Couldn't find a viable weak classifier!");

	chosen ~= *bestPair;

	return tuple(make_image_weak_classifier(bestPair.pixel, bestPair.tau), bestEps);
}

void calc_next_p(uint T)(float[T][] p, uint i, in image_weak_classifier_t h,
		in Image[] xs, in byte[] ys, float w_i)
in {
	assert(xs.length == ys.length);
	assert(p.length == xs.length);
	assert(i < p[0].length);
}
out {
	assert(!(p[0][i + 1].isNaN && p[xs.length - 1][i + 1].isNaN));
}
body {
	immutable m = xs.length;
	float E_i = 0;
	version (parallel) {
		foreach (t, ref pt; parallel(p)) {
			const float l_i = loss(h, xs[t], ys[t]);
			assert(pt[i] != 0);
			const float ep = pt[i] * exp(-w_i * l_i);
			//debug writeln("ep = ", ep);
			assert(!ep.isNaN);
			E_i += ep;
			pt[i + 1] = ep;
		}
	} else {
		for (uint t = 0; t < m; ++t) {
			const float l_i = loss(h, xs[t], ys[t]);
			const float ep = p[t][i] * exp(-w_i * l_i);
			assert(!ep.isNaN);
			E_i += ep;
			p[t][i + 1] = ep;
			assert(!p[t][i + 1].isNaN);
		}
	}
	assert(E_i != 0);
	version (parallel) {
		foreach (t, ref pt; parallel(p))
			pt[i + 1] /= E_i;
	} else {
		for (uint t = 0; t < m; ++t) {
			assert(!p[t][i + 1].isNaN);
			p[t][i + 1] /= E_i;
		}
	}
}

void print_p_sum(uint T)(in float[T][] p, uint i) {
	float sum = 0;
	for (int t = 0; t < p[0].length; ++t)
		sum += p[t][i];
	writeln("sum p[", i, "] = ", sum);
}
