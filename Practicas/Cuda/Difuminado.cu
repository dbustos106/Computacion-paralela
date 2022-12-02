#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

// Object for cascading classification
CascadeClassifier frontFace_cascade;

// Object for video capture
VideoCapture capture;

// Object for saving the video
VideoWriter writer;

__global__ void difuminarMatrix(unsigned char *d_frame, int frameWidth, int frameHeight, int frontFaceX, int frontFaceY, int frontFaceWidth, int frontFaceHeight)
{

    __shared__ unsigned char segment_image[110][148][3];

    // Define kernel size and radius
    int KernelSize = 21;
    int radius = (KernelSize - 1) / 2;

    // Define global indexes
    int gIndexX = frontFaceX + blockIdx.x * 90;
    int gIndexY = frontFaceY + blockIdx.y * 128 + threadIdx.x;

    // Define local indexes
    int lIndexY = threadIdx.x + radius;

    int gMaxIndexX = frontFaceX + frontFaceWidth;
    int gMaxIndexY = frontFaceY + frontFaceHeight;

    // Define iterators
    int elem, i, j;

    if (gIndexY < gMaxIndexY)
    {
        for (elem = 0; elem < 110 && gIndexX + elem < gMaxIndexX; elem++)
        {
            int gIndexTotal = 3 * (frameWidth * gIndexY + gIndexX + elem - radius);
            segment_image[elem][lIndexY][0] = *(d_frame + gIndexTotal);
            segment_image[elem][lIndexY][1] = *(d_frame + gIndexTotal + 1);
            segment_image[elem][lIndexY][2] = *(d_frame + gIndexTotal + 2);
        }

        if (threadIdx.x < radius)
        {
            for (elem = 0; elem < 110 && gIndexX + elem < gMaxIndexX; elem++)
            {
                int gIndexSup = 3 * (frameWidth * (gIndexY - radius) + gIndexX + elem - radius);
                segment_image[elem][lIndexY - radius][0] = *(d_frame + gIndexSup);
                segment_image[elem][lIndexY - radius][1] = *(d_frame + gIndexSup + 1);
                segment_image[elem][lIndexY - radius][2] = *(d_frame + gIndexSup + 2);

                int gIndexInf = 3 * (frameWidth * ((gIndexY + 128) % gMaxIndexY) + gIndexX + elem - radius);
                segment_image[elem][lIndexY + 128][0] = *(d_frame + gIndexInf);
                segment_image[elem][lIndexY + 128][1] = *(d_frame + gIndexInf + 1);
                segment_image[elem][lIndexY + 128][2] = *(d_frame + gIndexInf + 2);
            }
        }

        __syncthreads();

        for (elem = radius; elem < 90 + radius && gIndexX + elem < gMaxIndexX; elem++)
        {

            int sumX = 0;
            int sumY = 0;
            int sumZ = 0;

            for (i = -radius; i < radius; i++)
            {
                for (j = -radius; j < radius; j++)
                {
                    int lIndexXMod = abs(elem + i) % 110;
                    int lIndexYMod = abs(lIndexY + j) % 148;
                    sumX = sumX + segment_image[lIndexXMod][lIndexYMod][0];
                    sumY = sumY + segment_image[lIndexXMod][lIndexYMod][1];
                    sumZ = sumZ + segment_image[lIndexXMod][lIndexYMod][2];
                }
            }

            int div = KernelSize * KernelSize;
            int promX = sumX / div;
            int promY = sumY / div;
            int promZ = sumZ / div;

            int gIndexTotal = 3 * (frameWidth * gIndexY + gIndexX + elem);
            *(d_frame + gIndexTotal) = promX;
            *(d_frame + gIndexTotal + 1) = promY;
            *(d_frame + gIndexTotal + 2) = promZ;
        }
    }
}

int main(int argc, const char **argv)
{

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Get input video path
    char *rutaIn = (char *)malloc(49 * sizeof(char));
    strcat(strcpy(rutaIn, "./Archivos/"), argv[1]);

    // Get output video path
    char *rutaOut = (char *)malloc(49 * sizeof(char));
    strcat(strcpy(rutaOut, "./Archivos/"), argv[2]);

    // Load cascading sort algorithms
    if (!frontFace_cascade.load("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"))
    {
        cout << " Error en la carga del algoritmo de cascada " << endl;
        return -1;
    };

    // Load video from input file
    capture.open(rutaIn);

    // Verify success in uploading the video
    if (!capture.isOpened())
    {
        printf(" Error en la carga de video");
    }
    else
    {

        // Get the starting time
        auto start = chrono::high_resolution_clock::now();
        cout << " Detección de caras iniciado... " << endl;

        // Initialize the writer
        int fcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        Size frame_size(int(capture.get(3)), int(capture.get(4)));
        writer = VideoWriter(rutaOut, fcc, 30, frame_size, true);

        Mat frame;
        while (capture.read(frame))
        {
            Mat frameClone = frame.clone();

            if (frameClone.empty())
            {
                cout << " No capturó ningún frame " << endl;
                break;
            }

            // Define host and device
            int size = sizeof(unsigned char) * frameClone.total() * 3;
            unsigned char *h_frame = frameClone.data;
            unsigned char *d_frame;

            // Reserve memory and copy
            err = cudaMalloc((void **)&d_frame, size);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to malloc (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            err = cudaMemcpy(d_frame, h_frame, size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            // Detect the faces
            Mat frame_gray;
            std::vector<Rect> frontFaces;
            cvtColor(frameClone, frame_gray, COLOR_BGR2GRAY);
            equalizeHist(frame_gray, frame_gray);
            frontFace_cascade.detectMultiScale(frame_gray, frontFaces);

            // Blur the faces of the frame
            for (Rect frontFace : frontFaces)
            {
                int threadsPerBlock = 128;
                dim3 numBlocks((int)ceil(frontFace.width / 90.0), (int)ceil(frontFace.height / 128.0), 1);
                difuminarMatrix<<<numBlocks, threadsPerBlock>>>(d_frame, frameClone.cols, frameClone.rows, frontFace.x, frontFace.y, frontFace.width, frontFace.height);
            }
            cudaDeviceSynchronize();

            err = cudaMemcpy(h_frame, d_frame, size, cudaMemcpyDeviceToHost);
            Mat frameSol = Mat(frameClone.rows, frameClone.cols, CV_8UC3, h_frame);

            // free memory
            err = cudaFree(d_frame);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to free (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            writer.write(frameSol);
        }

        // Get the end time
        auto end = chrono::high_resolution_clock::now();

        freopen("times.txt", "a", stdout);
        // Get elapsed time in sequential execution
        auto elapsed = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "The elapsed time in the cuda execution is " << elapsed.count() / (float)1e6 << endl;

    }

    capture.release();
    writer.release();

    free(rutaIn);
    free(rutaOut);

    return 0;
}
