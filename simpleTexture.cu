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
#include <curand.h>

//#include opencv
#include <opencv2/opencv.hpp>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check


////////////////////////////////////////////////////////////////////////////////
// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

// Auto-Verification Code
bool testResult = true;

__device__ void triArray(float *a, const int size){
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

__device__ void medianFilter(float*input, float *outputData,int width,int height){
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    //to find the median -> tri à bulle + valeur du milieu
    float values[9];
    if(x > 0 && x < width-1 && y > 0 && y < height -1){
        int index = 0;
        for(int i = -1; i < 2; i++){
            for(int j = -1; j < 2; j++){
                values[index] = input[(y+j)*width+(x+i)];
                index++;
            }
        }
        triArray(values, 9);
        outputData[y*width + x] = values[4];
    }else outputData[y*width + x] =  input[y*width + x];
}

__device__ void sobelFilter(float *input, float*outputData, int width, int height){
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x > 0 && x < width-1 && y > 0 && y < height -1){
        char sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        char sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
        float pixel_x = 0.0, pixel_y = 0.0;
        int index = 0;
        for(int i = -1; i < 2; i++){
            for(int j = -1; j < 2; j++){
                pixel_x += input[(y+j)*width + (x+i)] * sobel_x[index];
                pixel_y += input[(y+j)*width + (x+i)] * sobel_y[index];
                index++;
            }
        }
        outputData[y*width + x] = sqrt((pixel_x*pixel_x)+(pixel_y*pixel_y)); 
    }
}

__device__ void multiply(float*input_1, float*output, int width, int height){
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    output[y*width + x] *= input_1[y*width + x];
}

__device__ void dispersionFilter(float *outputData,char*commutation_array, int width, int height){
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    unsigned int id = y*width+x;
    
    float xstep = 1.0f / width, ystep = 1.0f / height;
    float u = x * xstep, v = y * ystep;

    if(x > 0 && y > 0 && x < width-1 && y < height-1){
        outputData[id] = tex2D(tex, u+xstep*commutation_array[id], v+ystep*commutation_array[id+1]);
    }else outputData[id] = tex2D(tex, u,v);
}

__global__ void applyFilters(unsigned char *input, unsigned char *outputData, char*commutation_array, int width, int height){
    dispersionFilter(input,commutation_array, width, height);
    /*__syncthreads();
    medianFilter(input, outputData, width, height);
    __syncthreads();
    sobelFilter(outputData, input, width, height);//result of sobel in input
    __syncthreads();
    multiply(input, outputData, width, height);//multiply result of sobel with result of median*/
}

void getCommutationArray(char* arr, int size, unsigned int window){for(int i = 0; i < size; i++)arr[i] = (char)rand()%window - (window/2);}

void sendDataToGpu(unsigned char* img, unsigned char*devImg){
    
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    findCudaDevice(argc, (const char **) argv);
    srand(time(NULL));
    bool continuer = true;
    int width = 0, height = 0, window = 3;
    cv::VideoCapture cam(0); cv::Mat img;
    unsigned char*data = NULL, *devDat = NULL;
    cam >> img; width = img.cols; height = img.rows; data = (unsigned char*)img.data;//getting size of the frame
    char*com = (char*)malloc((width*height+1)*sizeof(char));

    //while(continuer){
        getCommutationArray(com, width*height, window);
        cam >> img; width = img.cols; height = img.rows; data = (unsigned char*)img.data;
        printf("%d %d %d\n" , img.at<cv::Vec3b>(0,0).val[0], img.at<cv::Vec3b>(0,0).val[1], img.at<cv::Vec3b>(0,0).val[2]);
        printf("%d %d %d\n\n" , data[0], data[1], data[2]);
        if(cv::waitKey(33) == ' ')continuer = false;
    //}

    cam.release();
    free(com);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
