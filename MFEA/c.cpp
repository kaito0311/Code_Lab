#include<bits/stdc++.h>
using namespace std;

int main()
{
    unsigned seed = chrono:: steady_clock::now().time_since_epoch().count();
    std:: default_random_engine e(seed);

    normal_distribution<double> dis(0, 50);
    cout << dis(e) << endl;


  return 0;
  
}