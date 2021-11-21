/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 1024
#define BLOCKS 16384
#define NUM_VALS THREADS*BLOCKS

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = length-i;
  }
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
 

void bitonic_sort(float *values)
{
 cudaEvent_t start_1, start_2, start_3, stop_1, stop_2, stop_3;
 float milliseconds_1 = 0;
 float milliseconds_2 = 0;
 float milliseconds_3 = 0;
cudaEventCreate(&start_1);
cudaEventCreate(&start_2);
cudaEventCreate(&start_3);
cudaEventCreate(&stop_1);
cudaEventCreate(&stop_2);
cudaEventCreate(&stop_3);
  
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  
  //MEM COPY FROM HOST TO DEVICE
  cudaEventRecord(start_1); // START 1
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  cudaEventRecord(stop_1);  // STOP 1
  cudaEventSynchronize(stop_1); // STOP 1 SYNC
  cudaEventElapsedTime(&milliseconds_1, start_1, stop_1);   // TIME 1 CALC

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  
  int j, k;
  /* Major step */
cudaEventRecord(start_2); // START 2
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
  cudaEventRecord(stop_2);  // STOP 2
  cudaEventSynchronize(stop_2); // STOP 2 SYNC
  cudaEventElapsedTime(&milliseconds_2, start_2, stop_2);   // TIME 2 CALC
  
  //MEM COPY FROM DEVICE TO HOST
  cudaEventRecord(start_3); // START 3
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
  cudaEventRecord(stop_3);  // STOP 3
  cudaEventSynchronize(stop_3); // STOP 3 SYNC
  cudaEventElapsedTime(&milliseconds_3, start_3, stop_3);   // TIME 3 CALC
  
  printf("Host to Device: %lf  Kernel: %lf  Device to Host:  %lf\n", milliseconds_1, milliseconds_2, milliseconds_3);
}

int main(void)
{
  clock_t start, stop;

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);

  start = clock();
  bitonic_sort(values); /* Inplace */
  stop = clock();

  print_elapsed(start, stop);
}