module read_data;

// http://yann.lecun.com/exdb/mnist/
import std.stdio;
import std.algorithm;
import std.bitmanip;
import std.typecons;
import weak_classifier;
import data;

void main() {
	immutable trainLabelsFname = "data/train-labels-idx1-ubyte";
	immutable trainImagesFname = "data/train-images-idx3-ubyte";
	immutable testLabelsFname = "data/t10k-labels-idx1-ubyte";
	immutable testImagesFname = "data/t10k-images-idx3-ubyte";

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

		const imgs = getImages(trainImagesFile, new Image[info.nImg], info.nRows, info.nCols);

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

		const imgs = getImages(testImagesFile, new Image[info.nImg], info.nRows, info.nCols);

		dumpImages(imgs, info);
	}
}

auto getNItemsLabel(File f) {
	const buf = f.rawRead(new ubyte[8]);

	// Check magic number
	immutable magicNumber = bigEndianToNative!int(buf[0..4]);
	assert(magicNumber == 2049);

	return bigEndianToNative!int(buf[4..8]);
}

auto getNItemsImages(File f) {
	const buf = f.rawRead(new ubyte[16]);

	// Check magic number
	immutable magicNumber = bigEndianToNative!int(buf[0..4]);
	assert(magicNumber == 2051);

	immutable nImg = bigEndianToNative!int(buf[4..8]);
	immutable nRows = bigEndianToNative!int(buf[8..12]);
	immutable nCols = bigEndianToNative!int(buf[12..16]);

	return ImagesInfo(nImg, nRows, nCols);
}

void dumpLabels(File f, uint n) {
	const buf = f.rawRead(new ubyte[n]);

	buf.each!(b => write(b, ", "));
	writeln();
}

auto getImages(File f, Image[] buf, int rows, int cols) {

	for (int i = 0; i < buf.length; i++) {
		auto pixels = f.rawRead(new ubyte[rows * cols]);
		buf[i] = pixels;
	}

	return buf;
}

void dumpImages(in Image[] imgs, in ImagesInfo info) {
	foreach (img; imgs) {
		for (int i = 0; i < info.nRows; i++) {
			for (int j = 0; j < info.nCols; j++)
				writef("%c ", img[i * info.nCols + j] == 0 ? ' ' : '#');
				//writef("%d ", bw_classifier(img[i * info.nCols + j]));
			writeln();
		}
	}
}
