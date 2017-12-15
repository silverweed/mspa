module adaboost;

import std.math;
import std.typecons;
import std.conv;
import std.algorithm;
import std.algorithm.sorting;
import std.traits;
version (parallel) import std.parallelism;
debug import std.stdio;
debug import std.format;
import classifiers;
import data;

alias adaboost_t = ReturnType!(adaboost!(0, 1));

/// Expected order of magnitude of the floating point error
shared float_t fltErr = 0;

/// NUM = number to recognize
/// T = number of weak classifiers to use
/// Returns a tuple of slices {weights, params}
auto adaboost(ubyte NUM, uint T)(in Image[] xs, in ubyte[] ys_in)
in {
	static assert(NUM >= 0 && NUM <= 9);
	static assert(T > 0);
	assert(xs.length == ys_in.length);
}
do {
	const ys = toBinaryLabel!NUM(ys_in);

	immutable m = xs.length;
	auto p = new float_t[T][](m);
	/// Weights
	auto w = new float_t[T];
	auto h = new image_weak_classifier_t[T];
	HParams[] chosen;

	// Initially set P_1(t) = 1/f for t = 1, ..., m
	for (uint t = 0; t < m; ++t)
		p[t][0] = 1.0 / m;
	fltErr = abs(pSum!T(p, 0) - 1);
	debug writeln("fltErr = ", fltErr);

	// Adaboost main loop
	for (uint i = 0; i < T; ++i) {
		debug {
			writeln("Iteration ", i, "/", T);
			writeln("p sum = ", pSum!T(p, i));
			writeln(abs(pSum!T(p, i) - 1), " vs ", 10 * fltErr);
		}
		assert(abs(pSum!T(p, i) - 1) < 20 * fltErr);

		// 1. choose h_i such that e_i is far from 1/2
		const res = chooseH!T(p, i, xs, ys, chosen);
		debug writeln(res[0], ", ", chosen);
		const params = res[0];
		const eps = res[1];
		h[i] = makeImageWeakClassifier(params.pixel, cast(ubyte)params.tau);

		// 2. compute weight
		w[i] = 0.5 * log((1 - eps) / eps);
		debug (2) {
			writeln("chosen params: pixel = ", params.pixel.asCoords(28).pretty(), ", tau = ", params.tau);
			writeln("epsilon = ", eps);
			writeln("log((1-e)/e) = ", log((1 - eps)/eps));
			writeln("w[", i, "] = ", w[i]);
		}

		// 3. Calculate P_(i+1)
		if (i < T - 1)
			calcNextP!T(p, i, h[i], xs, ys, w[i]);
	}

	return tuple(w, chosen);
}


/// Transform labels [0-9] into {-1, 1}
byte[] toBinaryLabel(uint NUM)(in ubyte[] ys) pure {
	import std.array;
	return array(ys.map!(y => cast(byte)(2 * (y == NUM) - 1)));
}

/// Transform labels [0-9] into {-1, 1} (non-templatized version)
byte[] toBinaryLabel(in ubyte[] ys, uint num) pure {
	import std.array;
	return array(ys.map!(y => cast(byte)(2 * (y == num) - 1)));
}

private:

/// L function
///       / 1  if h(x) == y
/// L(x) =
///       \ -1 otherwise
auto L(H)(in H h, in Image x, byte y) pure
in {
	assert(h(x) == 1 || h(x) == -1);
	assert(y == 1 || y == -1);
}
out (result) {
	assert(result == 1 || result == -1);
}
do {
	return h(x) * y;
}

float_t calcEpsilon(uint T)(in Image[] xs, in byte[] ys, in float_t[T][] p, uint i, uint pixel, ubyte tau)
in {
	assert(xs.length == ys.length);
	assert(pixel < xs[0].length);
	assert(i < p[0].length);
}
out (epsilon) {
	debug if (epsilon <= 0 || epsilon > 1) writeln("eps = ", epsilon);
	assert(0 < epsilon && epsilon < 1 - fltErr);
}
do {
	immutable m = xs.length;
	const h = makeImageWeakClassifier(pixel, tau);

	float_t eps = 0;
	//int cnt = 0;
	for (int t = 0; t < m; ++t) {
		debug if (p[t][i].isNaN) writeln("p[%d][%d] = %f".format(t, i, p[t][i]));
		assert(!p[t][i].isNaN);
		eps += (L(h, xs[t], ys[t]) == -1) * p[t][i];
		//debug if (i > 0) writeln(t, " eps = ", eps);
	}
	//debug writeln("h_{", pixel, ", ", tau, "}:  eps = ", eps, " (cnt = ", cnt, ")");
	return eps;
}

/// Extract an h such that its epsilon is as far from 1/2 as possible
auto chooseH(uint T)(in float_t[T][] p, uint i, in Image[] xs, in byte[] ys, ref HParams[] chosen)
in {
	assert(p.length == xs.length);
	assert(xs.length == ys.length);
	assert(i < p[0].length);
}
out (result) {
	debug writeln("image len = ", xs[0].length, ", pixel = ", result[0].pixel);
	assert(0 <= result[0].pixel && result[0].pixel < xs[0].length);
	assert(0 <= result[0].tau && result[0].tau <= 255);
	assert(0 < result[1] && result[1] < 1.1);
}
do {
	immutable TAU_STEP = 64;

	HParams bestPair;
	auto bestEps = 0.5;
	bool found = false;

	// Loop over the weak classifier parameters p and tau.
	// Choose the combination of parameters which minimizes abs(epsilon - 1/2).
	version (parallel) {
		float[HParams] epsilons;
		
		for (uint tau = 0; tau < 256; tau += TAU_STEP)
			for (uint pixel = 0; pixel < xs[0].length; ++pixel) {
				auto k = HParams(pixel, tau);
				if (!chosen.canFind(k))
					epsilons[k] = 0.5;
			}

		assert(epsilons.length == 256 / TAU_STEP * xs[0].length - chosen.length);
		
		debug writeln("Calculating best of ", epsilons.length, " alternatives...");
		foreach (_, k; parallel(epsilons.byKey))
			epsilons[k] = calcEpsilon!T(xs, ys, p, cast(uint)i, k.pixel, cast(ubyte)k.tau);
		
		foreach (params, eps; epsilons) {
			debug (2) writeln("eps{", params.pixel, ", ", params.tau, "} = ", eps);
			if (abs(eps - 0.5) - abs(bestEps - 0.5) > fltErr) {
				bestPair = params;
				bestEps = eps;
				found = true;
			}
		}

	} else {
		for (uint tau = 0; tau < 256; tau += TAU_STEP) {
			for (uint pixel = 0; pixel < xs[0].length; ++pixel) {
				const thisPair = HParams(pixel, tau);

				if (chosen.canFind(thisPair))
					continue;

				const eps = calcEpsilon!T(xs, ys, p, i, thisPair.pixel, cast(ubyte)thisPair.tau);
				if (abs(eps - 0.5) - abs(bestEps - 0.5) > fltErr) {
					debug writeln("eps = ", eps, ", best = ", bestEps,
							"(", abs(eps - 0.5), " vs ", abs(bestEps - 0.5), ")");
					found = true;
					debug writeln("assigning bestPair = ", thisPair);
					bestPair = thisPair;
					bestEps = eps;
				}
			}
		}
	}

	if (!found)
		throw new Exception("Couldn't find a viable weak classifier!");

	assert(!chosen.canFind(bestPair));
	chosen ~= bestPair;

	return tuple(bestPair, bestEps);
}

void calcNextP(uint T)(float_t[T][] p, uint i, in image_weak_classifier_t h,
		in Image[] xs, in byte[] ys, float_t w_i)
in {
	assert(xs.length == ys.length);
	assert(p.length == xs.length);
	assert(i < p[0].length);
}
out {
	assert(!(p[0][i + 1].isNaN && p[xs.length - 1][i + 1].isNaN));
}
do {
	immutable m = xs.length;
	float_t E_i = 0;
	for (uint t = 0; t < m; ++t) {
		const float_t l_i = L(h, xs[t], ys[t]);
		const float_t ep = p[t][i] * exp(-w_i * l_i);
		assert(!ep.isNaN);
		E_i += ep;
		p[t][i + 1] = ep;
		assert(!p[t][i + 1].isNaN);
	}
	assert(E_i != 0);

	for (uint t = 0; t < m; ++t) {
		assert(!p[t][i + 1].isNaN);
		p[t][i + 1] /= E_i;
	}
}

/// Returns the summation of all P's
float_t pSum(uint T)(in float_t[T][] p, uint i) pure {
	float_t sum = 0;
	for (int t = 0; t < p.length; ++t) {
		sum += p[t][i];
	}
	return sum;
}

debug {
	Tuple!(uint, uint) asCoords(uint pixel, uint nCols) pure {
		return tuple(pixel / nCols, pixel % nCols);
	}

	string pretty(T)(in T tup) {
		return "(" ~ to!string(tup[0]) ~ ", " ~ to!string(tup[1]) ~ ")";
	}
}
