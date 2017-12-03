module data;

immutable struct ImagesInfo {
	int nImg;
	int nRows;
	int nCols;
}

alias Image = ubyte[];

struct HParams {
	uint pixel;
	uint tau;
	invariant {
		assert(tau < 256);
	}
}

alias float_t = double;
