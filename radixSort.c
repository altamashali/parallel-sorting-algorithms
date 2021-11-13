#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <math.h>
#include <mpi.h>

#define MASTER 0
using namespace std;

double startTime, stopTime;
void sortFunction(int *array, int s);
int position(int *array, int eff_size);

/*********************** functions that are used ******************************/

void sortFunction(int *array, int s)
{
	int eff_size, minpos;
	int tmp;
	
	for(eff_size = s; eff_size > 1; eff_size--) {
		minpos = position(array, eff_size);
		tmp = array[minpos];
		array[minpos] = array[eff_size-1];
		array[eff_size-1] = tmp;
	}
}

int position(int *array, int eff_size)
{
	int i, minpos = 0;
	
	for(i=0; i<eff_size; i++)
		minpos = array[i] > array[minpos] ? i: minpos;
	return minpos;
}

/************************** main source code************************/

main(int argc, char **argv)
{
 	int * ary;
 	int * local_array;
 	int n = atoi(argv[1]);
 	int s;
	int p;
	int my_rank;
	int i,q;
	
	MPI_Status status;
	
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
	MPI_Comm_size(MPI_COMM_WORLD,&p);

/************************ master *********************************/
    
    if(my_rank==0)
    {
    	ary = (int *)malloc(n*sizeof(int));
    	srand((unsigned) time(NULL));
    	s = n/p;
    	for (q=0; q<p; q++) {
    		for (i=0; i<s; i++){
    			ary[q*s+i] = rand()%(2*n/p)+(q*2*n/p);
    		}
    	}
    	
    	FILE * fout;
 
    	fout = fopen("input","w");
    	for(i=0;i<n;i++)
        	fprintf(fout,"%d\n",ary[i]);
    	fclose(fout);
    }
    
    startTime = MPI_Wtime();
    
    s = n/p;
    local_array	 = (int*)malloc(s*sizeof(int));
    
    MPI_Scatter(ary, s, MPI_INT, local_array, s, MPI_INT, 0, MPI_COMM_WORLD);
    
    sortFunction(local_array, s);
    
    MPI_Gather(local_array, s, MPI_INT, ary, s, MPI_INT, 0, MPI_COMM_WORLD);
    
	stopTime = MPI_Wtime();
	if(my_rank==0)
 	{
    	FILE * fout;
 
    	printf("%d; %d processors; %f secs\n", s, p, (stopTime-startTime));
 
    	fout = fopen("result","w");
    	for(i=0;i<n;i++)
        	fprintf(fout,"%d\n",ary[i]);
    	fclose(fout);
	}
	
	free(local_array);
	if (my_rank==0) free(ary);
    
	MPI_Finalize();
}