module read_data;

// http://yann.lecun.com/exdb/mnist/
import std.stdio;
import std.algorithm;
import std.bitmanip;
import std.typecons;
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

/// Fills the algorithms' parameters from a file. The file must contain lines
/// starting with COEFF and followed by 3 spaces and a comma-separated list of weights,pixel,tau.
/// The number of coefficients must is guessed by the function by counting how many consecutive 
/// lines starting with COEFF are found.
/// FIXME written in haste and currently bugged
auto readSavedCoeffs(string fname) {
	import std.file;
	import std.string;
	import std.conv;
	import adaboost;

	float_t[] w;
	HParams[] params;

	auto algo = new adaboost_t[10];

	int guessedT = 0;
	int curCoeff = -1;
	bool guessingT = true;
	int n = 0;

	foreach (line; File(fname).byLine) {
		debug writeln("read line ", line);
		if (line[0 .. 5] == "COEFF") {
			++curCoeff;
			if (guessingT)
				++guessedT;
			else if (curCoeff > guessedT)
				throw new Exception("Inconsistent coefficient number in saved file!");

			assert(line.length > 8);
			const csv = line[8 .. $];
			const splitted = csv.split(",");
			w ~= splitted[0].to!float_t;
			params ~= HParams(splitted[1].to!uint, splitted[2].to!uint);

		} else {
			if (guessingT) {
				if (guessedT > 0) {
					guessingT = false;
					debug writeln("Guessed T = ", guessedT);
				}
				continue;
			} else if (curCoeff > 0 && curCoeff != guessedT - 1)
				throw new Exception("Inconsistent coefficient number in saved file! ("
						~ curCoeff.to!string ~ " vs " ~ guessedT.to!string ~ ")");

			if (n > 9)
				break;
			
			if (curCoeff < 0)
				continue;

			debug writeln("curCoeff = ", curCoeff, ", w = ", w, ", params = ", params);
			assert(w.length == guessedT && params.length == guessedT);
			debug writeln("algo[", n, "] ok");
			algo[n] = tuple(w, params);
			w = [];
			params = [];
			++n;
			curCoeff = -1;
			if (n > 10)
				throw new Exception("Too many digits in saved file!");
		}
	}

	return algo;
}
