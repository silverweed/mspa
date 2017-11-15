module adaboost;

import loss_function;
import weak_classifier;
import data;
import std.math;

auto create_hs(int nRows, int nCols) {
	auto hs = new image_weak_classifier_t[nRows * nCols];
	// hs[i] classifies pixel i.
	for (int i = 0; i < hs.length; ++i)
		hs[i] = ((i) => (in Image x) => bw_classifier(x[i]))(i);
	return hs;
}

// Extract an h | its epsilon != 1/2.
auto choose_h(in float[][] p, in Image[] xs, in byte[] ys, image_weak_classifier_t[] hs) {

	import std.algorithm.mutation;

	immutable m = xs.length;

	auto ok(float eps) {
		return abs(eps - 0.5) > 0.001;
	}

	int j = 0;
	for ( ; j < hs.length; ++j) {
		float eps = 0;
		for (uint t = 0; t < m; ++t) {
			eps += p[j][t] * (hs[j](xs[t]) != ys[t]);
		}
		if (ok(eps))
			break;
	}

	if (j == hs.length)
		throw new Exception("Couldn't find a viable weak classifier!");

	const h = hs[j];
	hs.remove(j);

	return h;
}

auto adaboost(uint T)(in Image[] xs, in byte[] ys)
in {
	assert(xs.length == ys.length);
}
body {
	const hs = create_hs(28, 28);

	immutable m = xs.length;
	auto p = new float[T][m];
	auto w = new float[T];
	auto h = new image_weak_classifier_t[T];
	auto l = loss01;

	for (uint t = 0; t < m; ++t)
		p[0][t] = 1.0 / m;

	for (uint i = 0; i < T; ++i) {
		// 1. choose h_i | e_i != 1/2
		h[i] = choose_h();

		// 2. compute weight
		w[i] = 0.5 * ln((1 - e) / e);

		// 3. Calculate P_(i+1)
		if (i < T - 1) {
			auto E_i = 0;
			for (uint t = 0; t < m; ++t) {
				immutable l_i = l(h[i](xs[t]), ys[t]);
				immutable ep = p[i][t] * exp(-w[i] * l_i);
				E_i += ep;
				p[i + 1][t] = ep;
			}
			for (uint t = 0; t < m; ++t)
				p[i + 1][t] /= E_i;
		}
	}

	return (x) => {
		auto sum = 0;
		for (uint i = 0; i < T; ++i)
			sum += w[i] * h[i](x);
	};
}
