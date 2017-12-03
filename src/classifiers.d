module classifiers;

import std.math;
import data;
debug import std.stdio;

// Weak classifier h_{p,τ}(x) = sgn(x_p - τ), where p is the pixel coordinate in x

/// An image_weak_classifier_t is a function which takes a pixel array and returns
/// -1 or 1 based on the sign of (x_p - τ).
/// p and τ are fixed for a single classifier.
alias image_weak_classifier_t = byte delegate(in Image) pure;

/// Creates a weak classifier given the p and tau parameters
auto makeImageWeakClassifier(uint pixel, ubyte tau) {
	return (in Image img) => cast(byte)(2 * (img[pixel] > tau) - 1);
}

/// Given weights and parameters, returns the function implementing the corresponding algorithm
auto makeImageStrongClassifier(in float_t[] w, in HParams[] params)
in {
	assert(w.length == params.length);
}
do {
	auto h = new image_weak_classifier_t[w.length];
	for (int i = 0; i < h.length; ++i)
		h[i] = makeImageWeakClassifier(params[i].pixel, cast(ubyte)params[i].tau);

	return (in Image x) {
		float_t sum = 0;
		for (uint i = 0; i < h.length; ++i) {
			debug writeln("w[", i, "] = ", w[i], ", h[i](x) = ", h[i](x));
			sum += w[i] * h[i](x);
		}
		debug stderr.writeln("Adaboost sum = ", sum);
		return sum;
	};
}
