module loss_function;

float loss01(X, Y)(X x, Y y) pure nothrow @safe @nogc {
	return x == y;
}
