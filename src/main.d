module main;

import std.stdio;
import std.algorithm;
import read_data;
import data;
import adaboost;

immutable trainLabelsFname = "data/train-labels-idx1-ubyte";
immutable trainImagesFname = "data/train-images-idx3-ubyte";
immutable testLabelsFname = "data/t10k-labels-idx1-ubyte";
immutable testImagesFname = "data/t10k-images-idx3-ubyte";

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

	auto algo = adaboost.adaboost!(2, 10)(xs, ys);
	debug writeln("Algorithm calculated.");

	int ok = 0;
	for (uint i = 0; i < 100; ++i) {
		immutable pred = algo(xs[i]);
		bool isOk = pred == 2 * (ys[i] == 2) - 1;
		writefln("predicted = %s, real = %d  | %s", pred > 0 ? "2" : "not 2", ys[i], isOk);
		if (isOk) ++ok;
	}
	writefln("guessed %d out of %d (%f%%)", ok, 100, ok * 100 / 100.0);
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
