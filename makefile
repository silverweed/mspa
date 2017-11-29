all: par

par:
	dmd -debug -version=parallel src/*.d -of=main.x

seq:
	dmd -debug src/*.d -of=main.x

release:
	dmd -release -version=parallel src/*.d -of=main.x

release-seq:
	dmd -release src/*.d -of=main.x
