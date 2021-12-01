#include <stdio.h>
#include <time.h>
#include <math.h>    
#include <stdlib.h>
#include "mpi.h"

#define MASTER 0
#define NUM_OUTPUT 10

double startTime, endTime;
int processRank, numProcesses, arraySize;
int * array;
double proc_times[2];
double proc_sumtimes[2];

/***********************************************************************************************/
int comparator(const void * a, const void * b) {
    return ( * (int *)a - * (int *)b );
};

/***********************************************************************************************/
void bottomCompare(int j) {
    int sendCounter = 0;
    int * sendBuffer = malloc((arraySize + 1) * sizeof(int));
    
    proc_times[0] = MPI_Wtime();
    MPI_Send(&array[arraySize - 1], 1, MPI_INT, processRank ^ (1 << j), 0, MPI_COMM_WORLD);
    proc_times[0] = MPI_Wtime() - proc_times[0];

    int min, recieveCounter;
    int * recieveBuffer = malloc((arraySize + 1) * sizeof(int));
    proc_times[1] = MPI_Wtime();
    MPI_Recv(&min, 1, MPI_INT, processRank ^ (1 << j), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    proc_times[1] = MPI_Wtime() - proc_times[1];

    for (int i = 0; i < arraySize; i++) {
        if (array[i] > min) {
            sendBuffer[sendCounter + 1] = array[i];
            sendCounter++;
        } else {
            break;
        }
    }

    sendBuffer[0] = sendCounter;

    proc_times[0] = MPI_Wtime();
    MPI_Send(sendBuffer, sendCounter, MPI_INT, processRank ^ (1 << j), 0, MPI_COMM_WORLD);
    proc_times[0] += MPI_Wtime() - proc_times[0];

    proc_times[1] = MPI_Wtime();
    MPI_Recv(recieveBuffer, arraySize, MPI_INT, processRank ^ (1 << j), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    proc_times[1] += MPI_Wtime() - proc_times[1];

    for (int i = 1; i < recieveBuffer[0] + 1; i++) {
        if (array[arraySize - 1] < recieveBuffer[i]) {
            array[arraySize - 1] = recieveBuffer[i];
        } else {
            break;
        }
    }

    qsort(array, arraySize, sizeof(int), comparator);
    free(sendBuffer);
    free(recieveBuffer);
    return;
};

/***********************************************************************************************/
void topCompare(int j) {
    int max, recieveCounter;
    int * recieveBuffer = malloc((arraySize + 1) * sizeof(int));
    MPI_Recv(&max, 1, MPI_INT, processRank ^ (1 << j), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int sendCounter = 0;
    int * sendBuffer = malloc((arraySize + 1) * sizeof(int));
    MPI_Send(&array[0], 1, MPI_INT, processRank ^ (1 << j), 0, MPI_COMM_WORLD);

    for (int i = 0; i < arraySize; i++) {
        if (array[i] < max) {
            sendBuffer[sendCounter + 1] = array[i];
            sendCounter++;
        } else {
            break;
        }
    }

    MPI_Recv(recieveBuffer, arraySize, MPI_INT, processRank ^ (1 << j), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    recieveCounter = recieveBuffer[0];

    sendBuffer[0] = sendCounter;
    MPI_Send(sendBuffer, sendCounter, MPI_INT, processRank ^ (1 << j), 0, MPI_COMM_WORLD);

    for (int i = 1; i < recieveCounter + 1; i++) {
        if (recieveBuffer[i] > array[0]) {
            array[0] = recieveBuffer[i];
        } else {
            break;
        }
    }

    qsort(array, arraySize, sizeof(int), comparator);
    free(sendBuffer);
    free(recieveBuffer);

    return;
};

/***********************************************************************************************/
int main(int argc, char * argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);

    arraySize = atoi(argv[1]) / numProcesses;
    array = (int *) malloc(arraySize * sizeof(int));
    srand(time(NULL));
    for (int i = 0; i < arraySize; i++) {
        array[i] = rand() % (atoi(argv[1]));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    int dimensions = (int)(log2(numProcesses));    // Cube Dimension

    if (processRank == MASTER) {
        printf("Number of Processes spawned: %d\n", numProcesses);
        startTime = MPI_Wtime();
    }

    qsort(array, arraySize, sizeof(int), comparator);

    for (int i = 0; i < dimensions; i++) {
        for (int j = i; j >= 0; j--) {
            if (((processRank >> (i + 1)) % 2 == 0 && (processRank >> j) % 2 == 0) || 
                ((processRank >> (i + 1)) % 2 != 0 && (processRank >> j) % 2 != 0)) {
                bottomCompare(j);
            } else {
                topCompare(j);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    MPI_Reduce(&proc_times, &proc_sumtimes, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); //sum -> average    
    double receive_avg, send_avg;
    double avg_num_proc = numtasks-1;

    if (processRank == MASTER) {
        endTime = MPI_Wtime();
        printf("Displaying sorted array (only 10 elements for quick verification)\n");

        for (i = 0; i < arraySize; i++) {
            if ((i % (arraySize / NUM_OUTPUT)) == 0) {
                printf("%d ",array[i]);
            }
        }
        printf("\n\n");
        printf("Time Elapsed (Sec): %f\n", endTime - startTime);

        receive_avg = proc_sumtimes[0]/avg_num_proc;
        send_avg = proc_sumtimes[1]/avg_num_proc;

        printf("Receiving: \n");
        printf("Min: %lf  Max: %lf  Avg:  %lf\n", proc_mintimes[0], proc_maxtimes[0], receive_avg);
        printf("Sending: \n");
        printf("Min: %lf  Max: %lf  Avg:  %lf\n", proc_mintimes[1], proc_maxtimes[1], send_avg);
    }

    free(array);
    MPI_Finalize();
    return 0;
};