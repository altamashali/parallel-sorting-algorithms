#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// copies d1 array to d2
void copyArray(double *d1, double *d2, int start, int length) {
  int i;
  int j = start;
  for (i = 0; i < length; i++) {
    d2[i] = d1[j];
    j++;
  }
}

// verify array is sorted
int is_sorted(double* data, int length) {
  int i;
  for (i = 0; i < length-1; ++i) {
    if (data[i] > data[i+1]) {
      return 0;
    }
  }
  return 1;
}


 // Finds the pivot point of the array
int find_pivot(double *data, int length, double pivot) {
  int i = 0;

  for (i = 0; i < length; i++) {

    if (data[i] > pivot) {
      return i - 1;
    }
  } 
  return i  - 1; 
}

/*
 * merge_arr two sorted arrays into one Sort d1 and d2 into data_sort
 * d1 and d2 have to be already sorted incrementally.
 */ 
void merge_arr(double *d1, int len1, double *d2, int len2, double *data_sort) {
  int i = 0, j = 0, index = 0;
  
  while(i < len1 && j < len2) {
    if (d1[i] < d2[j]) {
      data_sort[index] = d1[i]; 
      i++;
    } else {

      data_sort[index] = d2[j];
      j++;
    }
    index++;
  }  

  if (i >= len1) {
    while (j < len2) {

      data_sort[index] = d2[j];
      j++;
      index++;
    }
  }

  if (j >= len2) {
    while (i < len1) {
      
      data_sort[index] = d1[i];
      i++;
      index++;
    }
  }
}

// Parallel Quicksort Implementation using MPI
void quick_sort(double *data, int length, MPI_Comm comm, int *last_length){

  int num_processors, rank, pivot_idx, element_amount, keep_len, i;
  double pivot, mean_local[2] = {0.0, length}, mean_global[2]={0.0, 0.0}; 
  double *data_recieve, *data_keep;
  MPI_Status status;
  MPI_Comm new_comm;
  MPI_Comm_size(comm, &num_processors);
  MPI_Comm_rank(comm, &rank);
    
  if (num_processors == 1) {
    *last_length = length;
    return;
  }

  // calculate pivot
  for (i = 0; i < length; i++) {
    mean_local[0] = mean_local[0] + data[i];
  }

  MPI_Reduce(&mean_local, &mean_global, 2, MPI_DOUBLE, MPI_SUM, 0, comm);

  if (rank == 0) {
    pivot = mean_global[0] / mean_global[1];
  }

  MPI_Bcast(&pivot, 1, MPI_DOUBLE, 0, comm);

  // finding and setting pivot 
  pivot_idx = find_pivot(data, length, pivot);

  if (rank < num_processors / 2) {
    // Send elementa greater than pivot to left processor
    MPI_Send(data + (pivot_idx + 1), (length - 1) - pivot_idx, MPI_DOUBLE, rank + num_processors / 2, 00, comm);

    // Recieve elements lower than pivot from right processor
    MPI_Probe(rank + num_processors / 2, 11, comm, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &element_amount);
    data_recieve = (double *)malloc(element_amount*sizeof(double));
    MPI_Recv(data_recieve, element_amount, MPI_DOUBLE, rank + num_processors / 2, 11, comm, MPI_STATUS_IGNORE);
  } else {

    // Recieve elem higher than pivot from left hand side processor 
    MPI_Probe(rank - num_processors / 2, 00, comm, &status);
    MPI_Get_count(&status, MPI_DOUBLE, &element_amount);

    data_recieve = (double *)malloc(element_amount*sizeof(double));

    MPI_Recv(data_recieve, element_amount, MPI_DOUBLE, rank - num_processors / 2, 00, comm, MPI_STATUS_IGNORE);

    // Send elem lower than pivot from right hand side processor
    MPI_Send(data, pivot_idx + 1, MPI_DOUBLE, rank - num_processors / 2, 11, comm); 
  }

  // Create new array data to be kept, and to be merged with data recieved
  if (rank < num_processors / 2) {

    keep_len = pivot_idx + 1;
    data_keep = (double *)malloc(keep_len*sizeof(double));
    copyArray(data, data_keep, 0, keep_len);
  } else {
    keep_len = (length - 1) - pivot_idx;
    data_keep = (double *)malloc(keep_len*sizeof(double));
    copyArray(data, data_keep, pivot_idx+1, keep_len);
  }

  // merge arrays
  merge_arr(data_recieve, element_amount, data_keep, keep_len, data);

  // split mpi comm into left and right
  int com_split = rank / (num_processors / 2);
  MPI_Comm_split(comm, com_split, rank, &new_comm);

  // recursive call
  quick_sort(data, keep_len + element_amount, new_comm, last_length);
}

int main(int argc, char *argv[]) {  
  int rank;                                   /* Id of processors*/
  int num_processors;                                 /* Number of processor*/
  int n = atoi(argv[1]);                      /* n vector long*/
  int root = 0, i;
  double pivot, time_init, time_end;
  double *data, *data_sub, *data_sorted;
  int *last_length, *recieve_counts, *receive_displacements;
  last_length = (int *)malloc(1*sizeof(int));
  
  MPI_Init(&argc, &argv);                    /* Initialize MPI*/
  MPI_Comm_size(MPI_COMM_WORLD, &num_processors);    /* Get the number of processors */4
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);      /* Get my number*/
  
  int elements_per_proc = n/num_processors;          /* Elements per processor*/

  /* Generate random numbers, pivot and start time */
  if (rank == 0) {
    if (n <= 1) {
      printf("Already sorted, only one element");
      return 1;    
    }
    data = (double *)malloc(n*sizeof(double));
    data_sorted = (double *)malloc(n*sizeof(double));
    recieve_counts = (int *)malloc(num_processors*sizeof(int));
    receive_displacements = (int *)malloc(num_processors*sizeof(int));

    srand(time(NULL));
    for (i = 0; i < n; i++) {
        data[i] = drand48();
    }
    time_init = MPI_Wtime();  
  }

  /* Scatter chunk of data into smaller data_sub vectors */  
  data_sub = (double *)malloc(n*sizeof(double)); 
  MPI_Scatter(data, elements_per_proc, MPI_DOUBLE, data_sub, elements_per_proc, MPI_DOUBLE, root, MPI_COMM_WORLD);
  
  /* parallel quick sort algorithm */
  quick_sort(data_sub, elements_per_proc, MPI_COMM_WORLD, last_length);
  elements_per_proc = *last_length;

  /* Number of elements in each processor */  
  MPI_Gather(&elements_per_proc, 1, MPI_INT, recieve_counts, 1, MPI_INT, root, MPI_COMM_WORLD);
  
  /* Fill displacement vector for gatherv function */
  if (rank == 0) {
    int index = 0; receive_displacements[0] = index;
    for (i = 1; i < num_processors; i++) {
      index = index + recieve_counts[i-1];
      receive_displacements[i] = index;
    }
  }
 
 /* Gather all sub vectors into one large sorted vector  */
  MPI_Gatherv(data_sub, elements_per_proc, MPI_DOUBLE, data_sorted, recieve_counts, receive_displacements, MPI_DOUBLE, root, MPI_COMM_WORLD);
  
  /* End time and free datatypes */
  if (rank == 0) {
    time_end = MPI_Wtime() - time_init;
    printf("\nElapsed time: %.16f s\n", time_end);

    if (!is_sorted(data_sorted, n)) {
       printf("Error not sorted \n");
    }
  }

  free(last_length);
  free(data);
  free(data_sub);
  free(data_sorted);
  free(receive_displacements);
  free(recieve_counts);
  MPI_Finalize();
  return 0;
}