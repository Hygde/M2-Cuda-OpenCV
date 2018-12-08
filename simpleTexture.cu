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

__device__ void dispersionFilter(unsigned char*input, unsigned char *outputData,char*commutation_array, int width, int height){
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int id = y * width + x;

    if(x > 1 && y > 1 && x < width-2 && y < height-2){
        //tex2D(tex, u+xstep*commutation_array[id], v+ystep*commutation_array[id+1]);
        unsigned int index = 3 * ((y+commutation_array[id+1])*width + x + commutation_array[id]);
        if(id == 2*width+15)printf("cuda print: %d %d %d\n", outputData[id+0], outputData[id+1], outputData[id+2]);
        outputData[3*id+0] = input[index+0];
        outputData[3*id+1] = input[index+1];
        outputData[3*id+2] = input[index+2];
        if(id == 2*width+15)printf("cuda print: %d %d %d\n", outputData[3*id+0], outputData[3*id+1], outputData[3*id+2]);
    }else outputData[3*id] = input[3*id];
}

__global__ void applyFilters(unsigned char *input, unsigned char *outputData, char*commutation_array, int width, int height){
    dispersionFilter(input, outputData,commutation_array, width, height);
    //__syncthreads();
    /*medianFilter(input, outputData, width, height);
    __syncthreads();
    sobelFilter(outputData, input, width, height);//result of sobel in input
    __syncthreads();
    multiply(input, outputData, width, height);//multiply result of sobel with result of median*/
}

void getCommutationArray(char* arr, int size, unsigned int window){for(int i = 0; i < size; i++)arr[i] = (char)rand()%window - (window/2);}

void sendDataToGpu(unsigned char* data, unsigned char*devData, unsigned char*devBuffer, char*com, char*devCom, int size){
    checkCudaErrors(cudaMemcpy(devData, data, 3*size*sizeof(char), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devBuffer, data, 3*size*sizeof(char), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devCom, com, (size+1)*sizeof(char), cudaMemcpyHostToDevice));
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    findCudaDevice(argc, (const char **) argv);
    srand(time(NULL));
    bool continuer = true;
    int width = 0, height = 0, window = 5;
    cv::VideoCapture cam(0); cv::Mat img;
    unsigned char*data = NULL, *devData = NULL, *devBuffer = NULL;
    cam >> img; width = img.cols; height = img.rows; data = (unsigned char*)img.data;//getting size of the frame

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    
    checkCudaErrors(cudaMalloc((void**) &devData, 3*(width*height) * sizeof(char)));
    checkCudaErrors(cudaMalloc((void**) &devBuffer, 3*(width*height) * sizeof(char)));
    char*com = (char*)malloc((width*height+1)*sizeof(char)), *devCom;
    checkCudaErrors(cudaMalloc((void**) &devCom, (width*height+1) * sizeof(char)));

    while(continuer){
        getCommutationArray(com, width*height+1, window);
        cam >> img; width = img.cols; height = img.rows; data = (unsigned char*)img.data;
        sendDataToGpu(data, devData, devBuffer, com, devCom, width*height);
        //begin main
        
        printf("before %d %d %d\n" , data[3*(2*width+15)], data[3*(2*width+15)+1], data[3*(2*width+15)+1]);

        applyFilters<<<dimGrid, dimBlock, 0>>>(devBuffer, devData, devCom, width, height);

        //end main
        checkCudaErrors(cudaMemcpy(data, devData, 3*width*height*sizeof(char), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());

        printf("after %d %d %d\n\n" , data[3*(2*width+15)], data[3*(2*width+15)+1], data[3*(2*width+15)+1]);

        cv::imshow("test", img);
        if(cv::waitKey(33) == ' ')continuer = false;
    }

    cam.release();
    free(com);
    checkCudaErrors(cudaFree(devData));
    checkCudaErrors(cudaFree(devBuffer));
    checkCudaErrors(cudaFree(devCom));
    return 0;
}
