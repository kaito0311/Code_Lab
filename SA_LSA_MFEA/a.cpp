#include<bits/stdc++.h>
using namespace std; 

int main(){
    int a[10][10] = {0}; 
    int *p = &a[0][0];
    for(int i = 0; i < 100; i++){
        *(p+i) += 1; 
    }
    
    for(int i = 0; i < 10; i++){
        for(int j = 0; j < 10; j++){

        }
    }

    for(int i = 0; i < 10; i++){
        for(int j = 0; j < 10; j++){
            cout << a[i][j] << " "; 
        }
        cout << endl;
    }
}