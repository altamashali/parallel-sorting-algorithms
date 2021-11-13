#include <iostream> 
#include <string>
#include <vector> 

using namespace std;

int main() {

string input = "hello";
vector<char> ascii_list;

//string ascii_input;


// take input regarding the string 

// convert characters to ASCII

for (int i = 0; i < input.length(); i++) {
    char x = input.at(i);
    cout << x << endl;
    ascii_list.push_back(x);
}
cout << "converted to ascii: " << endl;
for (int i = 0; i < ascii_list.size(); i++){
    cout << ascii_list.at(i) << endl;
}
// run Batcher's Bitonic Sort

// run Counting Sort 

// run Radix Sort 

// convert characters back to string 

// print sorted string

// print runtimes 
string sorted_atoi; 
for (int i = 0; i < ascii_list.size(); i++){
    sorted_atoi.push_back(i);
 
}
cout << "back to string " << endl;
cout << sorted_atoi;
return 0;

}