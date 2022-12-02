#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

using namespace std;


// Function to multiply matrices
void multiplicarMatrices(int numHilos, int **matrix1, int **matrix2, int **result, int num_rows, int num_cols, int COMM);

// Function to print a matrix
void printMatrix(int **matrix, int num_rows, int num_cols);


int main(int argc, const char** argv){

    // Get the number of threads
    int numHilos = atoi(argv[5]);

    int num_rows1 = atoi(argv[1]);
    int num_cols1 = atoi(argv[2]);

    int num_rows2 = atoi(argv[3]);
    int num_cols2 = atoi(argv[4]);

    // Create matrices
    int **matrix1 = (int**) malloc(num_rows1*sizeof(int*));
    for(int i = 0; i < num_rows1; i++){
        matrix1[i] = (int*) malloc(num_cols1*sizeof(int));
    }

    int **matrix2 = (int**) malloc(num_rows2*sizeof(int*));
    for(int i = 0; i < num_rows2; i++){
        matrix2[i] = (int*) malloc(num_cols2*sizeof(int));
    }

    int **result = (int**) malloc(num_rows1*sizeof(int*));
    for(int i = 0; i < num_rows1; i++){
        result[i] = (int*) malloc(num_cols2*sizeof(int));
    }

    // Initialize matrices
    for(int i = 0; i < num_rows1; i++){
        for(int j = 0; j < num_cols1; j++){
            matrix1[i][j] = rand() % 100;
        }
    }

    for(int i = 0; i < num_rows2; i++){
        for(int j = 0; j < num_cols2; j++){
            matrix2[i][j] = rand() % 100;
        }
    }

    if(num_cols1 == num_rows2){

        // Get the starting time
        auto start = chrono::high_resolution_clock::now();

        multiplicarMatrices(numHilos, matrix1, matrix2, result, num_rows1, num_cols2, num_cols1);
        //printMatrix(matrix1, num_rows1, num_cols1);
        //printMatrix(matrix2, num_rows2, num_cols2);
        //printMatrix(result, num_rows1, num_cols2);
    
        // Get the end time
        auto end = chrono::high_resolution_clock::now();

        freopen("times.txt", "a", stdout);
        // Get elapsed time in sequential execution
        auto elapsed = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "The elapsed time in the openMP execution is " << elapsed.count() / (float)1e6 << endl;

    }else{
        printf(" Error en a dimensión de las matrices\n");
    }

    free(matrix1);
    free(matrix2);
    free(result);

    return 0;
}


void multiplicarMatrices(int numHilos, int **matrix1, int **matrix2, int **result, int num_rows, int num_cols, int COMM){
    
    #pragma omp parallel num_threads(numHilos)
    {

    int id = omp_get_thread_num();
    int init = (num_rows/numHilos) * id;
    int finish = init + ((num_rows/numHilos) - 1);

    if(id == numHilos-1){
        finish = num_rows-1;
    }

    //printf("id: %d, init: %d, finish: %d\n", id, init, finish);

    for(int i = init; i <= finish; i++){
        for(int j = 0; j < num_cols; j++){
            result[i][j] = 0;
            for(int k = 0; k < COMM; k++){
                result[i][j] += matrix1[i][k]*matrix2[k][j];
            }
        }
    }
    }
}


void printMatrix(int **matrix, int num_rows, int num_cols){
    printf("Matrix: \n");
    for(int i = 0; i < num_rows; i++){
        for(int j = 0; j < num_cols; j++){
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}