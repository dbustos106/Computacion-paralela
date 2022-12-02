all: 
	make
	time ./DifuminadoOpenMP input.mp4 output.mp4 16
	time ./DifuminadoOpenMP input.mp4 output.mp4 8
	time ./DifuminadoOpenMP input.mp4 output.mp4 6
	time ./DifuminadoOpenMP input.mp4 output.mp4 4
	time ./DifuminadoOpenMP input.mp4 output.mp4 2
	time ./DifuminadoSecuencial input.mp4 output.mp4