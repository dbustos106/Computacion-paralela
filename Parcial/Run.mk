all: 
	@echo "Ejecución secuencial"
	make
	time ./MultiplicacionOpenMP 8 8 8 8 16
	time ./MultiplicacionOpenMP 8 8 8 8 8
	time ./MultiplicacionOpenMP 8 8 8 8 6
	time ./MultiplicacionOpenMP 8 8 8 8 4
	time ./MultiplicacionOpenMP 8 8 8 8 2
	time ./MultiplicacionSecuencial 8 8 8 8