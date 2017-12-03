module main;

import std.stdio;
import std.algorithm;
import std.traits;
import std.typecons;
import std.conv;
import std.math;
import read_data;
import data;
import classifiers;
import adaboost : adaboost, adaboost_t;

enum trainLabelsFname = "data/train-labels-idx1-ubyte";
enum trainImagesFname = "data/train-images-idx3-ubyte";
enum testLabelsFname = "data/t10k-labels-idx1-ubyte";
enum testImagesFname = "data/t10k-images-idx3-ubyte";

enum T = 5;

void main() {
	auto algo = train();
	//auto algo = readSavedCoeffs("results/resultsgood10.dat");
	auto data = getTestData();
	testSingle(algo, data[0], data[1]);
	testOnevsAll(algo, data[0], data[1]);
}

auto getTestData()
out (result) {
	assert(result[0].length == result[1].length);
}
do {
	// Labels
	auto testLabelsFile = File(testLabelsFname, "r");
	immutable nLabels = getNItemsLabel(testLabelsFile);
	const ys = getLabels(testLabelsFile, nLabels);
	testLabelsFile.close();

	// Images
	auto testImagesFile = File(testImagesFname, "r");
	immutable info = getNItemsImages(testImagesFile);
	const xs = getImages(testImagesFile, info);
	testImagesFile.close();
	
	return tuple(xs, ys);
}

/// Tests each digit detection algorithm separately
void testSingle(in adaboost_t[] algo, in Image[] xs, in ubyte[] ys)
in {
	assert(xs.length == ys.length);
}
do {
	for (int n = 0; n < algo.length; ++n) {
		writefln("[%d] Coefficients:", n);
		for (int j = 0; j < algo[n][0].length; ++j) {
			// weights,pixel,tau
			writefln("COEFF   %f,%d,%d", algo[n][0][j], algo[n][1][j].pixel, algo[n][1][j].tau);
		}
		int ok = 0;
		for (uint i = 0; i < xs.length; ++i) {
			const alg = makeImageStrongClassifier(algo[n][0], algo[n][1]);
			immutable pred = alg(xs[i]).sgn;
			bool isOk = pred == 2 * (ys[i] == n) - 1;
			debug stderr.writefln("predicted = %s, real = %d  | %s",
					pred > 0 ? n.to!string : "not " ~ n.to!string, ys[i], isOk);
			if (isOk) ++ok;
		}
		writefln("[%d] guessed %d out of %d (%f%%)", n, ok, xs.length, ok * 100.0 / xs.length);
	}
}

/// Performs a 1-vs-all test
void testOnevsAll(in adaboost_t[] algo, in Image[] xs, in ubyte[] ys)
in {
	assert(xs.length == ys.length);
}
do {
	int ok = 0;
	for (int i = 0; i < xs.length; ++i) {
		int predicted;
		float_t maxRes;
		for (int n = 0; n < algo.length; ++n) {
			const alg = makeImageStrongClassifier(algo[n][0], algo[n][1]);
			immutable sum = alg(xs[i]);
			if (maxRes.isNaN || sum > maxRes) {
				maxRes = sum;
				predicted = n;
			}
		}
		bool isOk = predicted == ys[i];
		debug stderr.writefln("predicted = %d, real = %d | %s", predicted, ys[i], isOk);
		if (isOk) ++ok;
	}
	writefln("Guessed %d numbers out of %d (%f%%)", ok, xs.length, ok * 100.0 / xs.length);
}

/// Trains 10 separate algorithms for single digit binary detection and returns a slice containing them
auto train() {
	adaboost_t[10] algo;

	// Labels
	auto trainLabelsFile = File(trainLabelsFname, "r");
	immutable nLabels = getNItemsLabel(trainLabelsFile);
	const ys = getLabels(trainLabelsFile, nLabels);
	trainLabelsFile.close();

	// Images
	auto trainImagesFile = File(trainImagesFname, "r");
	immutable info = getNItemsImages(trainImagesFile);
	const xs = getImages(trainImagesFile, info);
	trainImagesFile.close();

	// This is where I'd put a static foreach
	stderr.writeln("Calculating algo 0");
	algo[0] = adaboost!(0, T)(xs, ys);
	stderr.writeln("Calculating algo 1");
	algo[1] = adaboost!(1, T)(xs, ys);
	stderr.writeln("Calculating algo 2");
	algo[2] = adaboost!(2, T)(xs, ys);
	stderr.writeln("Calculating algo 3");
	algo[3] = adaboost!(3, T)(xs, ys);
	stderr.writeln("Calculating algo 4");
	algo[4] = adaboost!(4, T)(xs, ys);
	stderr.writeln("Calculating algo 5");
	algo[5] = adaboost!(5, T)(xs, ys);
	stderr.writeln("Calculating algo 6");
	algo[6] = adaboost!(6, T)(xs, ys);
	stderr.writeln("Calculating algo 7");
	algo[7] = adaboost!(7, T)(xs, ys);
	stderr.writeln("Calculating algo 8");
	algo[8] = adaboost!(8, T)(xs, ys);
	stderr.writeln("Calculating algo 9");
	algo[9] = adaboost!(9, T)(xs, ys);
	stderr.writeln("Algorithms calculated.");
	// ...if only I had one

	return algo;
}

void dump() {
	{
		/// Train Labels
		auto trainLabelsFile = File(trainLabelsFname, "r");
		immutable nItems = getNItemsLabel(trainLabelsFile);

		debug writeln("n items = ", nItems);

		//dumpLabels(trainLabelsFile, nItems);
	}

	{
		/// Train Images
		auto trainImagesFile = File(trainImagesFname, "r");
		immutable info = getNItemsImages(trainImagesFile);

		debug writeln("n images: ", info.nImg, "\nn rows: ", info.nRows, "\nn cols: ", info.nCols);

		const imgs = getImages(trainImagesFile, info);

		//dumpImages(imgs, info);
	}

	{
		/// Test Labels
		auto testLabelsFile = File(testLabelsFname, "r");
		immutable nItems = getNItemsLabel(testLabelsFile);

		debug writeln("n items = ", nItems);

		//dumpLabels(testLabelsFile, nItems);
	}

	{
		/// Test Images
		auto testImagesFile = File(testImagesFname, "r");
		immutable info = getNItemsImages(testImagesFile);

		debug writeln("n images: ", info.nImg, "\nn rows: ", info.nRows, "\nn cols: ", info.nCols);

		const imgs = getImages(testImagesFile, info);

		dumpImages(imgs, info);
	}
}
