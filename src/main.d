module main;

import std.stdio;
import std.algorithm;
import std.traits;
import read_data;
import data;
import adaboost : adaboost, makeAlgo;

enum trainLabelsFname = "data/train-labels-idx1-ubyte";
enum trainImagesFname = "data/train-images-idx3-ubyte";
enum testLabelsFname = "data/t10k-labels-idx1-ubyte";
enum testImagesFname = "data/t10k-images-idx3-ubyte";

enum T = 20;

void main() {
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

	ReturnType!(adaboost!(0, 1))[10] algo;

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

	for (int n = 0; n < 10; ++n) {
		writefln("[%d] Coefficients:", n);
		for (int j = 0; j < algo[n][0].length; ++j)
			writefln("COEFF   %f,%d,%d", algo[n][0][j], algo[n][1][j].pixel, algo[n][1][j].tau);
		int ok = 0;
		for (uint i = 0; i < 100; ++i) {
			const alg = makeAlgo(algo[n][0], algo[n][1]);
			immutable pred = alg(xs[i]);
			bool isOk = pred == 2 * (ys[i] == 2) - 1;
			debug stderr.writefln("predicted = %s, real = %d  | %s", pred > 0 ? "2" : "not 2", ys[i], isOk);
			if (isOk) ++ok;
		}
		writefln("[%d] guessed %d out of %d (%f%%)", n, ok, 100, ok * 100 / 100.0);
	}
}

void tests() {
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
