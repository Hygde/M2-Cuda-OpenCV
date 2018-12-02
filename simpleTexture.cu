/*
todo:
1. Dispersion : code source sur la page du prof
2. Median filter
2. Sobel dans une copie
3. Multiplication du Sobel et du median
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Includes CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <curand.h>

//#include opencv
#include <opencv2/opencv.hpp>
//#include <opencv2/gpu/gpu.hpp>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f

// Define the files that are to be save and the reference images for validation
const char *imageFilename = "lena_bw.pgm";
const char *refFilename   = "ref_rotated.pgm";

const char *sampleName = "simpleTexture";

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

__device__ void medianFilter(float *outputData,int width,int height){
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    //float u = x / (width - 0.0f);
    //float v = y / (height - 0.0f);

    //to find the median -> tri à bulle + valeur du milieu
    float values[9];
    if(x > 0 && x < width-1 && y > 0 && y < height -1){
        int index = 0;
        for(int i = -1; i < 2; i++){
            for(int j = -1; j < 2; j++){
                values[index] = outputData[(y+j)*width + (x+i)];
                index++;
            }
        }
    }

    triArray(values, 9);

    // read from texture and write to global memory
    outputData[y*width + x] = values[4];
}

__device__ void dispersionFilter(float *outputData, int width, int height){
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    int acom[3] = {-1,0,1};

    float xstep = 1.0f / width, ystep = 1.0f / height;
    float u = x * xstep, v = y * ystep;

    outputData[y*width + x] = tex2D(tex, u+xstep*acom[threadIdx.x%3], v+ystep*acom[threadIdx.y%3]);
}

__global__ void applyFilters(float *outputData, int width, int height){
    dispersionFilter(outputData, width, height);
    __syncthreads();
    medianFilter(outputData, width, height);
    __syncthreads();
}

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);
    // Process command-line arguments
    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **) argv, "input"))
        {
            getCmdLineArgumentString(argc,(const char **) argv,"input",(char **) &imageFilename);

            if (checkCmdLineFlag(argc, (const char **) argv, "reference"))getCmdLineArgumentString(argc,(const char **) argv,"reference",(char **) &refFilename);
            else{
                printf("-input flag should be used with -reference flag");
                exit(EXIT_FAILURE);
            }
        }
        else if (checkCmdLineFlag(argc, (const char **) argv, "reference")){
            printf("-reference flag should be used with -input flag");
            exit(EXIT_FAILURE);
        }
    }

    runTest(argc, argv);

    printf("%s completed, returned %s\n",sampleName,testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv){
    int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL){
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    //Load reference image from image (output)
    float *hDataRef = (float *) malloc(size);
    char *refPath = sdkFindFilePath(refFilename, argv[0]);

    if (refPath == NULL){
        printf("Unable to find reference image file: %s\n", refFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(refPath, &hDataRef, &width, &height);

    // Allocate device memory for result
    float *dData = NULL, *iData = NULL;
    checkCudaErrors(cudaMalloc((void **) &dData, size));
    checkCudaErrors(cudaMalloc((void **) &iData, size));

    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray,&channelDesc,width,height));
    checkCudaErrors(cudaMemcpyToArray(cuArray,0,0,hData,size,cudaMemcpyHostToDevice));

    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModePoint;//cudaFilterModeLinear;
    tex.normalized = true;// access with normalized texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);

    // Execute the kernel
    applyFilters<<<dimGrid, dimBlock, 0>>>(dData, width, height);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",(width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);

    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData,dData,size,cudaMemcpyDeviceToHost));

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Wrote '%s'\n", outputFilename);

    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFree(iData));
    checkCudaErrors(cudaFreeArray(cuArray));
    free(imagePath);
    free(refPath);
    free(hOutputData);
}
