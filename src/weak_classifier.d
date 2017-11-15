module weak_classifier;

import data;

// Weak classifier h_{i,τ}(x) = sgn(x_i - τ)

alias weak_classifier_t = byte delegate(byte);
alias image_weak_classifier_t = byte delegate(in Image);

auto make_pixel_classifier(byte tau) {
	return (byte pixel) => cast(byte)(2 * (pixel > tau) - 1);
}

// Black/white classifier
weak_classifier_t bw_classifier;

static this() {
	bw_classifier = make_pixel_classifier(0);
}
