#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;


// Function to print a matrix
void printMatrix(int *matrix, int num_rows, int num_cols);

// Function to multiply matrices
__global__ void multiplicarMatrices(int *d_matrix1, int *d_matrix2, int *d_result, int num_rows, int num_cols, int COMM)
{

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < num_rows && col < num_cols){
        int sum = 0;
        for(int k = 0; k < COMM; k++){
            sum += d_matrix1[row*COMM + k]*d_matrix2[k*num_cols + col];
        }
        d_result[row*num_cols + col] = sum;
    }

    __syncthreads();

}


int main(int argc, const char **argv)
{

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int num_rows1 = atoi(argv[1]);
    int num_cols1 = atoi(argv[2]);

    int num_rows2 = atoi(argv[3]);
    int num_cols2 = atoi(argv[4]);

    // Create matrices
    int *h_matrix1 = (int*) malloc(num_rows1*num_cols1*sizeof(int));
    int *h_matrix2 = (int*) malloc(num_rows2*num_cols2*sizeof(int));
    int *h_result = (int*) malloc(num_rows1*num_cols2*sizeof(int));

    // Initialize matrices
    for(int i = 0; i < num_rows1*num_cols1; i++){
      h_matrix1[i] = rand() % 100;
    }

    for(int i = 0; i < num_rows2*num_cols2; i++){
      h_matrix2[i] = rand() % 100;
    }

    // Reserve memory and copy
    int *d_matrix1 = NULL;
    err = cudaMalloc((int **)&d_matrix1, num_rows1*num_cols1*sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to malloc (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Reserve memory and copy
    int *d_matrix2 = NULL;
    err = cudaMalloc((int **)&d_matrix2, num_rows2*num_cols2*sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to malloc (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Reserve memory and copy
    int *d_result = NULL;
    err = cudaMalloc((int **)&d_result, num_rows1*num_cols2*sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to malloc (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy matrix to global memory
    err = cudaMemcpy(d_matrix1, h_matrix1, num_rows1*num_cols1*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy matrix to global memory
    err = cudaMemcpy(d_matrix2, h_matrix2, num_rows2*num_cols2*sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    if(num_cols1 == num_rows2){

        // Get the starting time
        auto start = chrono::high_resolution_clock::now();

        dim3 threadsPerBlock(32, 32, 1);
        dim3 numBlocks((int)ceil(num_cols2 / 32.0), (int)ceil(num_rows1 / 32.0), 1);
        multiplicarMatrices<<<numBlocks, threadsPerBlock>>>(d_matrix1, d_matrix2, d_result, num_rows1, num_cols2, num_cols1);

        err = cudaMemcpy(h_result, d_result, num_rows1*num_cols2*sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        //printMatrix(h_matrix1, num_rows1, num_cols1);
        //printMatrix(h_matrix2, num_rows2, num_cols2);
        //printMatrix(h_result, num_rows1, num_cols2);
    
        // Get the end time
        auto end = chrono::high_resolution_clock::now();

        freopen("times.txt", "a", stdout);
        // Get elapsed time in sequential execution
        auto elapsed = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "The elapsed time in the cuda execution is " << elapsed.count() / (float)1e6 << endl;

    }else{
        printf(" Error en a dimensión de las matrices\n");
    }

    // free memory
    err = cudaFree(d_matrix1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // free memory
    err = cudaFree(d_matrix2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // free memory
    err = cudaFree(d_result);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    free(h_matrix1);
    free(h_matrix2);
    free(h_result);

    return 0;
}


void printMatrix(int *matrix, int num_rows, int num_cols){
    printf("Matrix: \n");
    for(int i = 0; i < num_rows*num_cols; i++){
      printf("%d ", matrix[i]);
      if((i+1) % num_cols == 0){
        printf("\n");
      }
    }
    printf("\n");
}
