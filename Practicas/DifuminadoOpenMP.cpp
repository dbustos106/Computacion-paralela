#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace cv;


// Object for face classification
CascadeClassifier frontFace_cascade;

// Object for video capture
VideoCapture capture;

// Object for saving the video
VideoWriter writer;


// Function that detects faces in a frame
void detectarRostros(Mat *frame);

// Function to blur a pixel
void difuminarPixel(Mat *frame, Mat *frameClone, int Px, int Py);


int main(int argc, const char** argv){

    // Get input video path
    char *rutaIn = (char*) malloc(49*sizeof(char));
    strcat(strcpy(rutaIn, "./Archivos/"), argv[1]);

    // Get output video path
    char *rutaOut = (char*) malloc(49*sizeof(char));
    strcat(strcpy(rutaOut, "./Archivos/"), argv[2]);

    // Get the number of threads
    int numHilos = atoi(argv[3]);

    // Load cascading sort algorithms
    if(!frontFace_cascade.load("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")){
        cout<< " Error en la carga del algoritmo de cascada" <<endl;
        return -1;
    };

    // Load video from input file
    capture.open(rutaIn);

    // Verify success in uploading the video
    if(!capture.isOpened()){
        printf(" Error en la carga de video");
    }else{

        // Get the starting time
        auto start = chrono::high_resolution_clock::now();
        cout<< " Detección de caras iniciado..." <<endl;

        // Initialize the writer
        int fcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        Size frame_size(int(capture.get(3)), int(capture.get(4)));
        writer = VideoWriter(rutaOut, fcc, 30, frame_size, true);

        // Calculate number of frames in the video
        int countFrame = (int) capture.get(CAP_PROP_FRAME_COUNT);

        // Create vector of frames
        vector<Mat> frames((countFrame+1)*8*sizeof(Mat));
        Mat frame;

        // Inject the frames to the vector
        for(int i = 0; i < countFrame; i++){      
  		    capture >> frame;

            if(frame.empty()){
                cout<< " No capturó ningún frame" <<endl;
                break;
            }

  		    frames[i*8] = frame.clone();
  	    }

        #pragma omp parallel num_threads(numHilos)
        {

        int id = omp_get_thread_num();
        int init = (countFrame/numHilos) * id;
        int finish = init + ((countFrame/numHilos) - 1);

        if(id == numHilos-1){
            finish = countFrame-1;
        }

        //printf("id: %d, init: %d, finish: %d\n", id, init, finish);

        for(int i = init; i <= finish; i++){
            
            // Detect faces
            detectarRostros(&frames[i*8]);

            // Abort if escape          
            if(waitKey(10) == 27){
                break; 
            }

            // Show the frame
            imshow("Detección de rostros", frames[i*8]);

  	    }
        }

        // Write the frame
        for(int i = 0; i < countFrame; i++){
            writer.write(frames[i*8]);  
        }

        // Get the end time
        auto end = chrono::high_resolution_clock::now();

        freopen("times.txt", "a", stdout);
        // Get elapsed time in sequential execution
        auto elapsed = chrono::duration_cast<chrono::microseconds>(end - start);
        cout << "The elapsed time in the openMP execution is " << elapsed.count() / (float)1e6 << endl;

    }
  
    capture.release();
    writer.release();

    free(rutaIn);
    free(rutaOut);

    return 0;  
}


void detectarRostros(Mat *frame){

    Mat frame_gray;
    Mat frameClone = frame->clone();

    // Convert the frame to grayscale
    cvtColor(*frame, frame_gray, COLOR_BGR2GRAY);

    // Normalize the brightness and increase contrast of the image
    equalizeHist(frame_gray, frame_gray);

    // Detect faces
    std::vector<Rect> frontFaces;
    #pragma omp critical
    {
    frontFace_cascade.detectMultiScale(frame_gray, frontFaces);
    }

    // Analyze the faces
    for (size_t i = 0; i < frontFaces.size(); i++){
        for(int fila = 0; fila < frontFaces[i].height; fila++){
            for(int columna = 0; columna < frontFaces[i].width; columna++){
                
                int Px = frontFaces[i].x + columna;
                int Py = frontFaces[i].y + fila;
                difuminarPixel(frame, &frameClone, Px, Py);
            
            }
        }
    }
}


void difuminarPixel(Mat *frame, Mat *frameClone, int Px, int Py){
    
    int sumX = 0;
    int sumY = 0;
    int sumZ = 0;

    for(int i = -10; i < 11; i++){
        cv::Vec3b* ptr = frameClone->ptr<cv::Vec3b>((Py + i) % frameClone->rows);
        for(int j = -10; j < 11; j++){
            sumX = sumX + ptr[(Px + j) % frameClone->cols][0];
            sumY = sumY + ptr[(Px + j) % frameClone->cols][1];
            sumZ = sumZ + ptr[(Px + j) % frameClone->cols][2];
        }
    }

    sumX = sumX/441;
    sumY = sumY/441;
    sumZ = sumZ/441;

    cv::Vec3b* ptr = frame->ptr<cv::Vec3b>(Py);
    ptr[Px] = cv::Vec3b(sumX, sumY, sumZ);
    
}