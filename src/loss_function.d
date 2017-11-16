module loss_function;

import std.conv;
import data;

float loss_base(alias h)(in Image x, byte y)
in {
	assert(h(x) == 1 || h(x) == -1);
	assert(y == 1 || y == -1);
}
out (result) {
	assert(result == 1 || result == -1, "Result is " ~ to!string(result));
}
body {
	return h(x) * y;
} 
