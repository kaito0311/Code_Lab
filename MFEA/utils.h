#include<bits/stdc++.h>
using namespace std;
#define D 30
#define range 50
#define POPULATION 100
#define pi 3.14
#define rmp 0.3
#define pm 1
#define LOOP 500
#define NUMBER_CHILD 100
#define NUMBER_RUN 1
#define NUMBER_TASK 2

template <typename T>
double sphere_function(vector<T> X)
{
    double sum = 0;
    for (int i = 0; i < X.size(); i++)
    {
        // X[i] -= 1;
        sum += X[i] * X[i];
    }
    return sum;
}
template <typename T>
double cos_sum(vector<T> &X)
{
    double sum = 0.0;
    for (int i = 0; i < X.size(); i++)
    {
        sum += cos(2 * pi * X[i]);
    }
    return sum;
}

template <typename T>
double rastrigin_function(vector<T> X)
{
    double result = 0.0;
    int A = 10;
    for (int i = 0; i < X.size(); i++)
    {
        // X[i] -= 1;
        result += X[i] * X[i] - A * cos(2 * pi * X[i]);
    }
    return result + A * X.size();
}
template<typename T>
double ackley_function(vector<T> &X)
{

    double result = 0;
    double a = 20;
    double b = 0.2;

    result = -1.0 * a * exp(-1.0 * b * sqrt(sphere_function(X) / X.size())) - exp(cos_sum(X) / X.size()) + a + exp(1);

    return result;
}


template <typename type>
void print(vector<vector<type>> &X)
{
    cout << "==================START======================" << endl;
    for (int row = 0; row < X.size(); row++)
    {
        for (int col = 0; col < X[row].size(); col++)
        {
            cout << X[row][col] << " ";
        }
        // cout << sphere_function(X[row]) << " " << ackley_function(X[row]) << endl;
        cout << endl;
        cout << endl;
    }
    cout << "==================END=======================" << endl;
}

template <typename type>
void allocate(vector<vector<type>> &X, int row, int col)
{
    X.resize(row);
    for (int i = 0; i < row;)
    {
        X[i].resize(col);
        i++;
    }
}

double random_double(double min, double max){
    double random = (double)rand() / RAND_MAX;
    return min + random*(max-  min);
}

template <typename T>
void print_min(vector<T> array)
{
    cout << *min_element(array.begin(), array.end());
    cout << endl;
}