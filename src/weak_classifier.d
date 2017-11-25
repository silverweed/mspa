module weak_classifier;

import data;

// Weak classifier h_{p,τ}(x) = sgn(x_p - τ), where p is the pixel coordinate in x

/// An image_weak_classifier_t is a function which takes a pixel array and returns
/// -1 or 1 based on the sign of (x_p - τ).
/// p and τ are fixed for a single classifier.
alias image_weak_classifier_t = byte delegate(in Image) pure;

/// Creates a weak classifier given the p and tau parameters
auto make_image_weak_classifier(uint pixel, ubyte tau) {
	return (in Image img) => cast(byte)(2 * (img[pixel] > tau) - 1);
}
