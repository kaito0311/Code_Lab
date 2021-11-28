#include "utils.h"
int main(){
    vector<int> array; 
    array.push_back(5);
    array.push_back(6);
    array.push_back(7);
    array.push_back(8);
    array.push_back(9);

    cout << max_element(array.begin(), array.end()) -array.begin();



}