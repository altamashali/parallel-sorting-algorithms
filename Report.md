# ParallelComputing CSCE 435 Group project

## 1. _due 10/27_ Group members:
1. Altamash Ali
2. Asad Ali
3. Jay Lenner
4. Mary Faith Mitchell

---

## 2. _due 11/3_ Project topic 

For our project, we have decided to sort strings based on their ASCII character values in ascending order. Our plan is to:

1. Take in a string as input and convert the letters to a list of ASCII characters
2. Run parallelized versions of Batcher’s Bitonic Sort, Counting Sort, and Radix Sort to sort the list 
3. Convert the sorted lists from ASCII characters back to letters in a string.
4. Return the sorted string and runtimes of each algorithm to determine which was the most efficient.

## 2. _due 11/3_ Brief project description (what algorithms will you be comparing and on what architectures)

#### For the project, we will be comparing the following three algorithms:
- Batcher's Bitonic Sort Algorithm
  - Architectures: MPI + CUDA
  - Description:
	- This sort algorithm focuses on converting a random sequence of numbers into a bitonic sequence, one that monotonically increases, then decreases. 
	- Bitonic sort is better suited parallel implementation since elements are compared in predefined sequence and the sequence of comparison doesn’t depend on the data. 
	- We are to assume that the input is a power of 2. 
	- If the input size is 1, then you are finished. 
	- If not, partition the input into two subarrays of size n/2, recursively sort these two subarrays in parallel, then merge the two sorted subarrays.
	- It has a runtime of O(n log^2 n) and depth O(log^2 n).
  - Psuedocode:
```
bitonic_sort(n):
	let be #n = size of n
	if (#n == 1) then n
	else
	    let
		bot = subseq(n,0, #n/2);
		top = subseq(n,#n/2,#n);
		mins = {min(bot,top):bot;top};
		maxs = {max(bot,top):bot;top};
	    in flatten({bitonic_sort(x) : x in [mins,maxs]});
batcher_sort(n):
	if (#n == 1) then n
	else
	    let b = {batcher_sort(x) : x in bottop(n)};
	    in bitonic_sort(b[0]++reverse(b[1]));
```
- Quick Sort Algorithm
  - Architectures: MPI + CUDA
  - Description: 
	- Quicksort is a divide and conquer algorithm which will pick a pivot element and will partition the input array around the chosen pivot element.
	- The partitioning given an input array and pivot element you sort the pivot element to the correct position in the sorted array and then
	- The runtime is typically O(nlogn) when the pivot for each recursive call is equal to the median element of the subarray. This is because the problem size is being halved for each subarray so they can be sorted with log n nested calls.
	- An advantage of this approach is that it has a short inner loop and it requires only nlogn time to sort for n items.
	- A disadvantage of this approach is that in its worst case implementation it can have a runtime of O(n^2).
  - Psuedocode:
```
parallel_quicksort(arr, len, comm):
  if num_processor = 1 then
      return
  Find mean value of all processors in group communication and set to Pivot
  Cast calculated pivot from root processor
  Split local subarray by pivot with one half less than the pivot and one half greater
  if (rank < num_processor / 2)
      Send greater half of subarry to processor with rank of rank + proc/2
      Receive lesser half of subarray from processor with rank of rank + proc/2
  else
      Seceive greater half of subarray from processor with rank of rank - proc/2
      Send lesser half of subarry to processor with rank of rank - proc/2
  Merge received array with retained array
  Split communication into two halves
  Call parallel_quicksort recursively
```
- Radix Sort Algorithm
  - Architectures: MPI + CUDA
  - Description:
	- The radix sort algorithm is a non-comparative sorting algorithm.
	- This is a multiple pass distribution sort algorithm that distributes each item to a bucket according to part of the items key beginning with the least significant part of the key.
	- The runtime of radix sort is O(d*(n+b)) where b is the base which represents the numbers. 
	- Advantages of the radix sort algorithm are that it is fast when the range of the array elements are less.
	- Some disadvantages of radix sort are that it depends on digits or letters, also radix sort is much less flexible then the other sorting algorithms. For example, it takes more space compared to quicksort which is inplace sorting.
  - Psuedocode:
```
radixSort(int A[], int n):
    int i, digitPlace = 1;
    int result[n];
    int largestNum = getMax(A, n);
    while(largestNum/digitPlace >0){
        int count[10] = {0};
        for (i = 0; i < n; i++)
            count[ (A[i]/digitPlace)%10 ]++;
        for (i = 1; i < 10; i++)
            count[i] += count[i - 1];
        for (i = n - 1; i >= 0; i--)
            result[count[ (A[i]/digitPlace)%10 ] - 1] = A[i];
            count[ (A[i]/digitPlace)%10 ]--;
        for (i = 0; i < n; i++)
            A[i] = result[i];
            digitPlace *= 10;
```

#### Communication:
* Discord: used for audio/text communication, hosting meetings
* Google Drive: used to collaborate on technical writing pieces and presentations
* GitHub: used to collaborate on coding and development  

## 3. _due 11/12_ Pseudocode for each algorithm and implementation

The Pseudocode for all three of the algorithms with their MPI implementations is added to GitHub. We used C/C++ for the implementations. We used GitHub as our primary resource for figuring out how to make the implementations. 
- See section 2 for pseudocode of Bitonic sort, Radix sort, and Quicksort. 
- See *bitonicSort.cpp*, *radixSort.c*, *quickSort.c* for the MPI implementations of chosen algorithms.

## 3. _due 11/12_ Evaluation plan - what and how will you measure and compare

For all three of the sorting algorithms (Bitonic, Radix, Quick) we plan to measure the following:
- Effective use of GPU:
	- Input Size: 2^16, 2^20, 2^24
	- Threads: 64, 128, 512, 1024
- Strong scaling to more nodes:
	- Input Size: 10m
	- Processes: 2, 4, 8, 16, 32, 64
- Weak scaling:
	- Input Size: 1m, 10m, 100m
	- Processes: 2, 4, 8, 16, 32, 64

We will be implementing Strong & Weak Scaling using MPI on the Grace Cluster. We will also compare weak scaling (# of processors) and the effective use of the GPU (# of threads) using the same input sizes to see if performance improves with in-node support.

## 4. _due 11/19_ Performance evaluation

CPU Resources:
- Requested 16 nodes
- Assigned 4 tasks/cores per node
- Requested 8GB memory per node

GPU Resources:
- Requested 1 GPU per node
- Assigned 4 tasks/cores per node
- Requested 8GB memory per node 

Important Note: For strong and weak scaling, we’re only showing the comparison with computation time for right now. Once we finish collecting our communication time, we will have the data for total and communication. Our GPU data includes total, computation, and communication time. 

### Weak Scaling:

For weak scaling, each graph is based on the runtime being compared and the input type. The graphs show each type of algorithm and input size reflected by the color and type of line. We noted that overall, communication takes the longest with our weak scaling implementation. As expected, the 100M input size takes the longest to compute, with Radix sort being the highest, followed by Bitonic and Quicksort respectively. There is a spike in communication runtime around 15-20 processors. This is due to the structure of the network, as the fat tree communication links are being built. The point of diminishing return is reached around 30 processors, as the graphs begin to flatline.

![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/weak/weak-comm-random.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/weak/weak-comm-reverse.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/weak/weak-comm-sorted.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/weak/weak-comp-random.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/weak/weak-comp-reverse.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/weak/weak-comp-sorted.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/weak/weak-total-random.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/weak/weak-total-reverse.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/weak/weak-total-sorted.png?raw=true)

### Strong Scaling:

Blue lines represent bitonic sort, red lines represent radix sort, and yellow lines represent quick sort. The input size used was 10 million and the number of processors was increased for each of the runs for each sort. Increasing the number of processors for bitonic generally decreases the runtime. Radix sort also decreases until it reaches the point of diminishing returns at 16 processors at which the time will no longer decrease. Quicksort for sorted and random input experiences a sharp increase in runtime from 16 to 32 cores for sorted and random inputs. It then declines when the count is increased to 64. For reversed input, quicksort declines normally as the number of processors increases. All of the algorithms reached points of diminishing returns eventually and we can see that they are about equivalent according to runtime.

![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/strong/strong-comm-random.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/strong/strong-comm-reversed.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/strong/strong-comm-sorted.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/strong/strong-comp-random.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/strong/strong-comp-reversed.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/strong/strong-comp-sorted.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/strong/strong-total-random.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/strong/strong-total-reversed.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/strong/strong-total-sorted.png?raw=true)

### GPU Experimentation:

Below we have compared each of the different algorithms with the number of threads versus the time taken to run. Each graph is labeled with which input size it represents, as well as the input type and the runtimes being compared.  Overall, each sorting algorithm looks as if they are behaving as expected.

#### Bitonic Sort
For Bitonic Sort, we noted that when the input size is 2^16, the sorted input takes the longest, due to the computation time. This in turn increases the total time. Otherwise, the timing remains fairly consistent.  As the input sizes increase, the total time decreases as more threads are used.

![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/gpu/gpu-bitonic-16.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/gpu/gpu-bitonic-20.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/gpu/gpu-bitonic-24.png?raw=true)

#### Radix Sort

Throughout all the input sizes, the trends remain fairly similar. As the number of threads increase, the time it takes to complete the computations and communication decrease. Communication times are low, while computation times for sorted and random inputs are the highest. 

![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/gpu/gpu-radix-16.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/gpu/gpu-radix-20.png?raw=true)
![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/gpu/gpu-radix-24.png?raw=true)

#### Quicksort

Quicksort was not efficient, as it was unable to handle large input sizes. This is due to the fact that Quicksort has not been optimized for parallelization on the GPU at this time. Since the purpose of parallelization is to work with larger problem sizes, this is not an effective algorithm. For the input size we were able to run, one can see that the time increases as the number of threads increases, and computations have the highest runtimes. Since the communication times are small, the total time and computation time are basically the same value, and therefore the lines on the graph overlap to where only computation time is visible.

![alt text](https://github.tamu.edu/altamashali/csce435project/blob/master/graphs/gpu/gpu-quicksort-16.png?raw=true)

## 5. _due 12/1_ Presentation, 5 min + questions

[Link to Presentation](https://docs.google.com/presentation/d/1fNbkdVu2Ie0tsaBjLZ3Rz6JXMZGCsNaxGDqnb1Kosuk/edit?usp=sharing)

## 6. _due 12/8_ Final Report

[Link to Final Report](https://docs.google.com/document/d/10iN08tt-xM4xwN7D1FSMKx9e05i9fdwhPVykCVvH6iA/edit?usp=sharing)
