module read_data;

// http://yann.lecun.com/exdb/mnist/
import std.stdio;
import std.algorithm;
import std.bitmanip;
import std.typecons;
import weak_classifier;
import data;

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

auto getLabels(File f, uint n) {
	return f.rawRead(new ubyte[n]);
}

void dumpLabels(File f, uint n) {
	const buf = f.rawRead(new ubyte[n]);

	buf.each!(b => write(b, ", "));
	writeln();
}

auto getImages(File f, in ImagesInfo info) {
	
	auto buf = new Image[info.nImg];

	for (int i = 0; i < buf.length; i++) {
		auto pixels = f.rawRead(new ubyte[info.nRows * info.nCols]);
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
