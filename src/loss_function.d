module loss_function;

import data;

float loss(H)(in H h, in Image x, byte y) pure
in {
	assert(h(x) == 1 || h(x) == -1);
	assert(y == 1 || y == -1);
}
out (result) {
	assert(result == 1 || result == -1);
}
do {
	return h(x) * y;
}
