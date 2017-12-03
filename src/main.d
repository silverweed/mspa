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

enum T = 7;

void main() {
	auto algo = train();
	test(algo);
}

void test(in ReturnType!(adaboost!(0, 1))[] algo) {
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

	//test7(xs, ys);
	//return;

	for (int n = 0; n < 10; ++n) {
		writefln("[%d] Coefficients:", n);
		for (int j = 0; j < algo[n][0].length; ++j) {
			// weights,pixel,tau
			writefln("COEFF   %f,%d,%d", algo[n][0][j], algo[n][1][j].pixel, algo[n][1][j].tau);
		}
		int ok = 0;
		for (uint i = 0; i < xs.length; ++i) {
			const alg = makeAlgo(algo[n][0], algo[n][1]);
			immutable pred = alg(xs[i]);
			bool isOk = pred == 2 * (ys[i] == n) - 1;
			debug stderr.writefln("predicted = %s, real = %d  | %s",
					pred > 0 ? n.to!string : "not " ~ n.to!string, ys[i], isOk);
			if (isOk) ++ok;
		}
		writefln("[%d] guessed %d out of %d (%f%%)", n, ok, xs.length, ok * 100.0 / xs.length);
	}
}

auto train() {
	ReturnType!(adaboost!(0, 1))[10] algo;

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

void test7(in Image[] xs, in ubyte[] ys) {
	import weak_classifier;
	auto w = new double[20];
	auto p = new HParams[20];
	w[0] = 1.187066;
	p[0].pixel = 740;
	p[0].tau = 0;
	w[1] = -0.613129;
	p[1].pixel = 405;
	p[1].tau = 0;
	w[2] = 0.492180;
	p[2].pixel = 709;
	p[2].tau = 0;
	w[3] = -0.539642;
	p[3].pixel = 155;
	p[3].tau = 0;
	w[4] = 0.420202;
	p[4].pixel = 231;
	p[4].tau = 0;
	w[5] = -0.399228;
	p[5].pixel = 401;
	p[5].tau = 0;
	w[6] = 0.318886;
	p[6].pixel = 296;
	p[6].tau = 64;
	w[7] = 0.309534;
	p[7].pixel = 715;
	p[7].tau = 0;
	w[8] = -0.316851;
	p[8].pixel = 578;
	p[8].tau = 0;
	w[9] = -0.284271;
	p[9].pixel = 432;
	p[9].tau = 0;
	w[10] = 0.231159;
	p[10].pixel = 744;
	p[10].tau = 0;
	w[11] = -0.273023;
	p[11].pixel = 540;
	p[11].tau = 0;
	w[12] = 0.213836;
	p[12].pixel = 679;
	p[12].tau = 0;
	w[13] = -0.224914;
	p[13].pixel = 376;
	p[13].tau = 0;
	w[14] = 0.168546;
	p[14].pixel = 438;
	p[14].tau = 0;
	w[15] = -0.181043;
	p[15].pixel = 603;
	p[15].tau = 0;
	w[16] = 0.175717;
	p[16].pixel = 266;
	p[16].tau = 128;
	w[17] = -0.168264;
	p[17].pixel = 485;
	p[17].tau = 0;
	w[18] = 0.186916;
	p[18].pixel = 284;
	p[18].tau = 0;
	w[19] = -0.197780;
	p[19].pixel = 580;
	p[19].tau = 0;

	auto algo7 = makeAlgo(w, p);

	for (int i = 0; i < xs.length; ++i) {
		auto pred = algo7(xs[i]);
		bool isOk = pred == 2 * (ys[i] == 7) - 1;
		debug stderr.writefln("predicted = %s, real = %d  | %s", pred > 0 ? "7" : "not 7", ys[i], isOk);
	}
}
