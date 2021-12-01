#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

double proc_times[2];
double proc_sumtimes[2];

int randomInRange(int min, int max) {
  return rand() % 10000000;
}

int max(int * input, int n) {
  int retval = 0;
  for (int i = 0; i < n; i++) {
    if (input[i] > retval) {
      retval = input[i];
    }
  }
  return retval;
}

void logger(char * str, int thread_id) {
  int mult = 3;
  char out[strlen(str) + thread_id * mult + 1];
  strcpy(out + thread_id * mult, str);
  for (int i = 0; i < thread_id; i++) {
    for (int j = 0; j < mult; j++) {
      out[mult * i + j] = '\t';
    }
  }
  out[strlen(out)] = '\0';
  printf("%s\n", out);
}

void countingSort(int * input, int * count, int n, int exp) {
  int * output = malloc(n * sizeof(int));
  int workingCount[10];
  int i;
  int index;

  for (i = 0; i < n; i++) {
    count[(input[i] / exp) % 10]++;
  }
  for (i = 0; i < 10; i++) {
    workingCount[i] = count[i];
  }
  for (i = 1; i < 10; i++) {
    workingCount[i] += workingCount[i - 1];
  }
  for (i = n - 1; i >= 0; i--) {
    index = (input[i] / exp) % 10;
    output[workingCount[index] - 1] = input[i];
    workingCount[index]--;
  }
  for (i = 0; i < n; i++) {
    input[i] = output[i];
  }
  free(output);
}

int compareArrays(int * arrA, int * arrB, int n) {
  for (int i = 0; i < n; i++) {
    if (arrA[i] != arrB[i]) {
      return 0;
    }
  }
  return 1;
}

void uniprocessorRadixSort(int * input, int n) {
  int m = max(input, n);
  for (int exp = 1; m / exp > 0; exp *= 10) {
    int count[10] = {
      0
    };
    countingSort(input, count, n, exp);
  }
}

int checkCorrect(int * originalInput, int * sortedInput, int n) {
  int * copy = (int * ) malloc(n * sizeof(int));
  for (int i = 0; i < n; i++) {
    copy[i] = originalInput[i];
  }
  uniprocessorRadixSort(copy, n);
  int correct = compareArrays(copy, sortedInput, n);
  free(copy);
  return correct;
}

int main(int argc,
  const char * argv[]) {
  MPI_Init(NULL, NULL);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, & world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, & world_size);

  int seeds[] = { 1, 2, 22443, 16882, 7931, 10723, 24902, 124, 25282, 2132 };
  int runs = 10;

  clock_t start, end;
  double cpu_time_used;
  double total;

  for (int size = 1; size < 20000000; size *= 2) {
    MPI_Barrier(MPI_COMM_WORLD);
    for (int run = 0; run < runs; run++) {
      MPI_Barrier(MPI_COMM_WORLD);

      int * input; 
      int * adjustedInput;
      int n;
      int adjustedN;
      int m;
      int subinputSize;
      int tailOffset;

      if (world_rank == 0) {
        srand(seeds[run]);
        n = size;
        input = (int * ) malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
          input[j] = randomInRange(0, n);
        }

        start = clock();
        subinputSize = ceil(n / ((double) world_size));
        tailOffset = world_size * subinputSize - n;

        adjustedN = subinputSize * world_size;
        adjustedInput = (int * ) malloc(adjustedN * sizeof(int));
        for (int i = 0; i < adjustedN; i++) {
          if (i < tailOffset) {
            adjustedInput[i] = 0;
          } else {
            adjustedInput[i] = input[i - tailOffset];
          }
        }

        if (run == 0) {
          total = 0;
        }
        m = max(input, n);
      }

      MPI_Bcast( & m, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast( & n, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast( & subinputSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
      MPI_Bcast( & adjustedN, 1, MPI_INT, 0, MPI_COMM_WORLD);

      MPI_Barrier(MPI_COMM_WORLD);

      int * subinput = (int * ) malloc(subinputSize * sizeof(int));
      MPI_Scatter(adjustedInput, subinputSize, MPI_INTEGER, subinput, subinputSize, MPI_INTEGER, 0, MPI_COMM_WORLD);
      int allCounts[10 * world_size];
      int * newSubinput = malloc(subinputSize * sizeof(int));
      for (int exp = 1; m / exp > 0; exp *= 10) {

        int count[10] = { 0 };
        int allCountsSum[10] = { 0 };
        int allCountsPrefixSum[10] = { 0 };
        int allCountsSumLeft[10] = { 0 };

        countingSort(subinput, count, subinputSize, exp);
        MPI_Allgather(count, 10, MPI_INTEGER, allCounts, 10, MPI_INTEGER, MPI_COMM_WORLD);

        for (int i = 0; i < 10 * world_size; i++) {
          int lsd = i % 10;
          int p = i / 10;
          int val = allCounts[i];

          if (p < world_rank) {
            allCountsSumLeft[lsd] += val;
          }
          allCountsSum[lsd] += val;
          allCountsPrefixSum[lsd] += val;
        }

        for (int i = 1; i < 10; i++) {
          allCountsPrefixSum[i] += allCountsPrefixSum[i - 1];
        }

        MPI_Request request;
        MPI_Status status;

        int lsdSent[10] = {
          0
        };

        int valIndexPair[2];
        int val, lsd, destIndex, destProcess, localDestIndex;

        for (int i = 0; i < subinputSize; i++) {
          val = subinput[i];
          lsd = (subinput[i] / exp) % 10;

          destIndex = allCountsPrefixSum[lsd] - allCountsSum[lsd] + allCountsSumLeft[lsd] + lsdSent[lsd];

          lsdSent[lsd]++;
          destProcess = destIndex / subinputSize;

          valIndexPair[0] = val;
          valIndexPair[1] = destIndex;

    	  proc_times[0] = MPI_Wtime();
          MPI_Isend( & valIndexPair, 2, MPI_INT, destProcess, 0, MPI_COMM_WORLD, & request);
		  proc_times[0] = MPI_Wtime() - proc_times[0];

    	  proc_times[1] = MPI_Wtime();
          MPI_Recv(valIndexPair, 2, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, & status);
    	  proc_times[1] = MPI_Wtime() - proc_times[1];

          localDestIndex = valIndexPair[1] % subinputSize;
          val = valIndexPair[0];
          newSubinput[localDestIndex] = val;
        }

        for (int i = 0; i < subinputSize; i++) {
          subinput[i] = newSubinput[i];
        }

      }

      int * output;
      if (world_rank == 0) {
        output = (int * ) malloc(adjustedN * sizeof(int));
      }

      MPI_Gather(subinput, subinputSize, MPI_INTEGER, & output[world_rank * subinputSize], subinputSize, MPI_INTEGER, 0, MPI_COMM_WORLD);
	  
	  MPI_Reduce(&proc_times, &proc_sumtimes, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); //sum -> average    
	  double receive_avg, send_avg;
	  double avg_num_proc = numtasks-1;

      if (world_rank == 0) {
        output += tailOffset;
        end = clock();
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
        total += cpu_time_used;

        if (!checkCorrect(input, output, n)) {
          return 1;
        }

        output -= tailOffset;
        free(output);
        free(adjustedInput);

        if (run == runs - 1) {
          printf("%d,%f\n", size, total / runs);
        }

		receive_avg = proc_sumtimes[0]/avg_num_proc;
    	send_avg = proc_sumtimes[1]/avg_num_proc;

		printf("Receiving: \n");
		printf("Min: %lf  Max: %lf  Avg:  %lf\n", proc_mintimes[0], proc_maxtimes[0], receive_avg);
		printf("Sending: \n");
		printf("Min: %lf  Max: %lf  Avg:  %lf\n", proc_mintimes[1], proc_maxtimes[1], send_avg);
        free(input);
      }
      free(subinput);
      free(newSubinput);
    }
  }
  MPI_Finalize();
}