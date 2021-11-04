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
- Counting Sort Algorithm
  - Architectures: MPI + CUDA
  - Description: 
	- The counting sort algorithm is a linear sorting technique which is based upon keys within a specified range.
	- The runtime is always O(n+k) since no matter how many times elements are put into the array the algorithm will always run n + k times.
	- An advantage if this is that it isn’t a comparison based algorithm so the values being sorted will not affect the functionality of the algorithm other than potentially increasing runtime.
	- A disadvantage of this approach is that for large integers an array the size of the integer will be created thus increasing runtime.
  - Psuedocode:
```
counting_sort(a):
   Max <- largest element in array
   Count array <- init with all zero
   For j = 0 to size
      Find total count of each unique element and store count at j index on count array
   For i = 1 to max
      Find the total sum and store within count array
   For j = size to 1
      Reconstruct array with elements
      Decrease the count of each element that was reconstructed by 1
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

## 3. _due 11/12_ Evaluation plan - what and how will you measure and compare

For example:
- Effective use of a GPU (play with problem size and number of threads)
- Strong scaling to more nodes (same problem size, increase number of processors)
- Weak scaling (increase problem size, increase number of processors)

## 4. _due 11/19_ Performance evaluation

Include detailed analysis of computation performance, communication performance.

Include figures and explanation of your analysis.

![alt text](image.jpg)

## 5. _due 12/1_ Presentation, 5 min + questions

- Power point is ok

## 6. _due 12/8_ Final Report

Example [link title](https://) to preview _doc_.
