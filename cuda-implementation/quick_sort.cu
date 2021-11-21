 #include <time.h>
 #include <stdio.h>
 #include <stdlib.h>
 
 #define MAX_THREADS	1024
 #define THREADS	64
 #define BLOCKS 1024
 #define N		16777216

 int*	r_values;
 int*	d_values;

 // initialize data set
 void Init(int* values, int i) {
	srand( time(NULL) );
        printf("\n------------------------------\n");

        if (i == 0) {
        // Sorted distribution
                printf("Sorted Distribution\n");
                for (int x = 0; x < N; ++x) {
                        values[x] = x;
                }
        }
        else if (i == 1) {
        // Reverse sorted distribution
                printf("Reverse Sorted Distribution\n");
                for (int x = 0; x < N; ++x) {
                        values[x] = N - x;
                }
        }
    	else if (i == 2) {
        // Zero distribution
                printf("Random Distribution\n");
                int r = rand() % 10000000;
                for (int x = 0; x < N; ++x) {
                        values[x] = r;
                }
            }
        printf("\n");
}

 // Kernel function
 __global__ static void quicksort(int* values) {
 #define MAX_LEVELS	300

	int pivot, L, R;
	int idx =  threadIdx.x + blockIdx.x * blockDim.x;
	int start[MAX_LEVELS];
	int end[MAX_LEVELS];

	start[idx] = idx;
	end[idx] = N - 1;
	while (idx >= 0) {
		L = start[idx];
		R = end[idx];
		if (L < R) {
			pivot = values[L];
			while (L < R) {
				while (values[R] >= pivot && L < R)
					R--;
				if(L < R)
					values[L++] = values[R];
				while (values[L] < pivot && L < R)
					L++;
				if (L < R)
					values[R--] = values[L];
			}
			values[L] = pivot;
			start[idx + 1] = L + 1;
			end[idx + 1] = end[idx];
			end[idx++] = L;
			if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1]) {
	                        // swap start[idx] and start[idx-1]
        	                int tmp = start[idx];
                	        start[idx] = start[idx - 1];
                        	start[idx - 1] = tmp;

	                        // swap end[idx] and end[idx-1]
        	                tmp = end[idx];
                	        end[idx] = end[idx - 1];
                        	end[idx - 1] = tmp;
	                }

		}
		else
			idx--;
	}
}
 
 // program main
 int main(int argc, char **argv) {
     
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

	printf("./quicksort starting with %d numbers...\n", N);
 //	unsigned int hTimer;
 	size_t size = N * sizeof(int);
 	
 	// allocate host memory
 	r_values = (int*)malloc(size);
 	
	// allocate device memory
    cudaMalloc((void**)&d_values, size);

	// allocate threads per block
    const unsigned int cThreadsPerBlock = 64;
                
	/* Types of data sets to be sorted:
         *      1. Normal distribution
         *      2. Gaussian distribution
         *      3. Bucket distribution
         *      4. Sorted Distribution
         *      5. Zero Distribution
         */

	for (int i = 0; i < 3; ++i) {
                // initialize data set
                Init(r_values, i);

	 	// copy data to device	
	 	cudaEventRecord(start_1); // START 1
		cudaMemcpy(d_values, r_values, size, cudaMemcpyHostToDevice);
		cudaEventRecord(stop_1);  // STOP 1
        cudaEventSynchronize(stop_1); // STOP 1 SYNC
        cudaEventElapsedTime(&milliseconds_1, start_1, stop_1);   // TIME 1 CALC

		//printf("Beginning kernel execution...\n");

	//	cutCreateTimer(&hTimer);
 	//	cudaThreadSynchronize();
	//	cutResetTimer(hTimer);
	 //	cutStartTimer(hTimer);
	
		// execute kernel
		cudaEventRecord(start_2); // START 2
 		quicksort <<< MAX_THREADS / cThreadsPerBlock, MAX_THREADS / cThreadsPerBlock, cThreadsPerBlock >>> (d_values);
 		cudaEventRecord(stop_2);  // STOP 2
        cudaEventSynchronize(stop_2); // STOP 2 SYNC
        cudaEventElapsedTime(&milliseconds_2, start_2, stop_2);   // TIME 2 CALC
	 	// cutilCheckMsg( "Kernel execution failed..." );

 		//cudaThreadSynchronize();
	 	//cutStopTimer(hTimer);
	 	//double gpuTime = cutGetTimerValue(hTimer);

 		// printf( "\nKernel execution completed in %f ms\n", gpuTime );
 	
	 	// copy data back to host
	 	cudaEventRecord(start_3); // START 3
		cudaMemcpy(r_values, d_values, size, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop_3);  // STOP 3
        cudaEventSynchronize(stop_3); // STOP 3 SYNC
        cudaEventElapsedTime(&milliseconds_3, start_3, stop_3);   // TIME 3 CALC
        
          printf("Host to Device: %lf  Kernel: %lf  Device to Host:  %lf\n", milliseconds_1, milliseconds_2, milliseconds_3);
 	
	 	// test print
 		/*for (int i = 0; i < N; i++) {
 			printf("%d ", r_values[i]);
 		}
 		printf("\n");
		*/

		// test
               // printf("\nTesting results...\n");
                for (int x = 0; x < N - 1; x++) {
                        if (r_values[x] > r_values[x + 1]) {
                                printf("Sorting failed.\n");
                                break;
                        }
                        else
                                if (x == N - 2)
                                        printf("SORTING SUCCESSFUL\n");
                }

	}
 	
 	// free memory
	cudaFree(d_values);
}