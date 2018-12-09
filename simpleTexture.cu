/*
todo:
1. Dispersion : code source sur la page du prof
2. Median filter
2. Sobel dans une copie
3. Multiplication du Sobel et du median
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

// Includes CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>         // helper functions for CUDA error check

//#include opencv
#include <opencv2/opencv.hpp>

__device__ void triArray(unsigned char *a, const int size){
    float tmp;
    for(int i = 0; i < size; i ++){
        for(int j = 0; j < size-1; j++){
            if(a[j] > a[j+1]){
                tmp = a[j];
                a[j] = a[j+1];
                a[j+1] = tmp;
            }
        }
    }
}

__device__ void medianFilter(unsigned char*input, unsigned char *outputData, unsigned char*devMed, int window, int width,int height){
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int id = y * width + x, idx3 = 3*id;
    unsigned int w2 = window*window, arrayId = 3*w2*id;
    int w_div2 = window/2;

    //to find the median -> tri à bulle + valeur du milieu
    if(x > w_div2 && x < width-w_div2 && y > w_div2 && y < height - w_div2){
        int index = 0;
        for(int i = -w_div2; i <= w_div2; i++){
            for(int j = -w_div2; j <= w_div2; j++){
                devMed[arrayId + index] = input[3 * ((y+j)*width+(x+i)) + 0];
                devMed[arrayId + w2 + index] = input[3 * ((y+j)*width+(x+i)) + 1];
                devMed[arrayId + 2*w2 + index] = input[3 * ((y+j)*width+(x+i)) + 2];
                index++;
            }
        }
        triArray(&devMed[arrayId], w2);triArray(&devMed[arrayId + w2], w2);triArray(&devMed[arrayId + 2*w2], w2);
        outputData[idx3 + 0] = devMed[arrayId + (w2/2)];//apply median
        outputData[idx3 + 1] = devMed[arrayId + w2 + (w2/2)];
        outputData[idx3 + 2] = devMed[arrayId + 2*w2 + (w2/2)];
    }else{//copy
        outputData[idx3 + 0] =  input[idx3 + 0];
        outputData[idx3 + 1] =  input[idx3 + 0];
        outputData[idx3 + 2] =  input[idx3 + 0];
    }
}

__device__ void sobelFilter(unsigned char *input, unsigned char *outputData, int width, int height){
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int id = y * width + x;

    if(x > 0 && x < width-1 && y > 0 && y < height -1){
        char sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        char sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
        float b_pixel[2] = {0, 0}, g_pixel[2] = {0, 0}, r_pixel[2] = {0, 0};
        int index = 0;
        for(int i = -1; i < 2; i++){
            for(int j = -1; j < 2; j++){
                b_pixel[0] += input[3*((y+j)*width + (x+i))] * sobel_x[index];
                b_pixel[1] += input[3*((y+j)*width + (x+i))] * sobel_y[index];
                g_pixel[0] += input[3*((y+j)*width + (x+i))] * sobel_x[index];
                g_pixel[1] += input[3*((y+j)*width + (x+i))] * sobel_y[index];
                r_pixel[0] += input[3*((y+j)*width + (x+i))] * sobel_x[index];
                r_pixel[1] += input[3*((y+j)*width + (x+i))] * sobel_y[index];
                index++;
            }
        }

        float b = sqrt((b_pixel[0]*b_pixel[0])+(b_pixel[1]*b_pixel[1]));
        float g = sqrt((g_pixel[0]*g_pixel[0])+(g_pixel[1]*g_pixel[1]));
        float r = sqrt((r_pixel[0]*r_pixel[0])+(r_pixel[1]*r_pixel[1]));

        outputData[3 * id + 0] = (b>128)?255:b;
        outputData[3 * id + 1] = (g>128)?255:g;
        outputData[3 * id + 2] = (r>128)?255:r;
    }
}

__device__ void multiply(unsigned char*input_1, unsigned char*output, int width, int height){
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int id = y * width + x;

    output[3 * id + 0] *= input_1[3 * id + 0] / 255.0f;
    output[3 * id + 1] *= input_1[3 * id + 1] / 255.0f;
    output[3 * id + 2] *= input_1[3 * id + 2] / 255.0f;
}

__device__ void dispersionFilter(unsigned char*input, unsigned char *outputData,char*commutation_array, int window, int width, int height){
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int id = y * width + x, w_div2 = window/2, arrayId = 3*id;

    if(x > w_div2 && y > w_div2 && x < width-w_div2 && y < height-w_div2){
        unsigned int index = 3 * ((y+commutation_array[id+1])*width + x + commutation_array[id]);
        outputData[arrayId+0] = input[index+0];
        outputData[arrayId+1] = input[index+1];
        outputData[arrayId+2] = input[index+2];
    }else outputData[arrayId] = input[arrayId];
}

__global__ void applyFilters(unsigned char *input, unsigned char *outputData, char*commutation_array, unsigned char*devMed, int window,int width, int height){
    dispersionFilter(input, outputData,commutation_array, window, width, height);
    __syncthreads();
    medianFilter(outputData, input, devMed, window, width, height);
    __syncthreads();
    sobelFilter(input, outputData, width, height);//result of sobel in input
    __syncthreads();
    multiply(input, outputData, width, height);//multiply result of sobel with result of median*/
}

void getCommutationArray(char* arr, int size, unsigned int window){for(int i = 0; i < size; i++)arr[i] = (char)rand()%window - (window/2);}

bool processEvent(bool&continuer, int&window){
    bool result = true;
    char carac = cv::waitKey(33);
    switch(carac){
        case 32://spacebar
            continuer = false;
        break;
        case 82://up
            window = (window < 11) ? window+2 : 11;
            printf("window size = %dx%d \n",window, window);
        break;
        case 84://down
            window = (window > 3) ? window - 2 : window;
            printf("window size = %dx%d \n",window, window);
        break;
        default:result = false;break;
    }
    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    findCudaDevice(argc, (const char **) argv);
    srand(time(NULL));
    bool continuer = true, config = true;
    int width = 0, height = 0, window = 5;
    cv::VideoCapture cam(0); cv::Mat img;
    unsigned char*data = NULL, *devData = NULL, *devBuffer = NULL, *devMed = NULL;
    cam >> img; width = img.cols; height = img.rows; data = (unsigned char*)img.data;//getting size of the frame

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    
    checkCudaErrors(cudaMalloc((void**) &devData, 3*(width*height) * sizeof(char)));
    checkCudaErrors(cudaMalloc((void**) &devBuffer, 3*(width*height) * sizeof(char)));
    checkCudaErrors(cudaMalloc((void**) &devMed, 3*window*window*(width*height) * sizeof(char)));
    char*com = (char*)malloc((width*height+1)*sizeof(char)), *devCom;
    checkCudaErrors(cudaMalloc((void**) &devCom, (width*height+1) * sizeof(char)));

    while(continuer){
        if(config){
            config = false;
            getCommutationArray(com, width*height+1, window);
            checkCudaErrors(cudaMemcpy(devCom, com, (width*height+1)*sizeof(char), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaFree(devMed));
            checkCudaErrors(cudaMalloc((void**) &devMed, 3*window*window*(width*height) * sizeof(char)));
        }
        cam >> img; width = img.cols; height = img.rows; data = (unsigned char*)img.data;
        checkCudaErrors(cudaMemcpy(devData, data, 3*width*height*sizeof(char), cudaMemcpyHostToDevice));
        //begin main

        StopWatchInterface *timer = NULL;
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);
        applyFilters<<<dimGrid, dimBlock, 0>>>(devData, devBuffer, devCom, devMed, window, width, height);
        checkCudaErrors(cudaMemcpy(data, devData, 3*width*height*sizeof(char), cudaMemcpyDeviceToHost));
        sdkStopTimer(&timer);
        printf("Processing time: %f (ms)\r\n", sdkGetTimerValue(&timer));

        //end main

        cv::imshow("test", img);
        config = processEvent(continuer, window);
    }

    cam.release();
    free(com);
    checkCudaErrors(cudaFree(devData));
    checkCudaErrors(cudaFree(devBuffer));
    checkCudaErrors(cudaFree(devMed));
    checkCudaErrors(cudaFree(devCom));
    return 0;
}
