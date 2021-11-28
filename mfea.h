#include <bits/stdc++.h>
#include "utils.h"
using namespace std;

void initial_population(vector<vector<double>> &X)
{
    for (int row = 0; row < POPULATION; row++)
    {
        X[row].resize(D);
        for (int col = 0; col < D; col++)
        {
            X[row][col] = random_double(-range, range);
        }
    }
}

void transform_to_rank(vector<vector<int>> &factorial_rank, vector<vector<double>> factorial_cost)
{
    vector<double> temp;
    temp.resize(factorial_cost[0].size());
    temp = factorial_cost[0];
    sort(temp.begin(), temp.end());
    for (int i = 0; i < factorial_cost[0].size(); i++)
    {
        factorial_rank[0][i] = find(temp.begin(), temp.end(), factorial_cost[0][i]) - temp.begin() + 1;
    }
    temp = factorial_cost[1];
    sort(temp.begin(), temp.end());
    for (int i = 0; i < factorial_cost[0].size(); i++)
    {
        factorial_rank[1][i] = find(temp.begin(), temp.end(), factorial_cost[1][i]) - temp.begin() + 1;
    }
}

void cal_factorial_cost(vector<vector<double>> X, vector<vector<int>> &factorial_rank, vector<vector<double>> &factorial_cost)
{
    // Compute fitness of each individual with two fitness function
    for (int individual = 0; individual < X.size(); individual++)
    {
        factorial_cost[0][individual] = sphere_function(X[individual]);
        factorial_cost[1][individual] = ackley_function(X[individual]);
    }
    transform_to_rank(factorial_rank, factorial_cost);
}
double compute_scalar_fitness(int individual, vector<vector<int>> &factorial_rank)
{

    int min_element = min(factorial_rank[0][individual], factorial_rank[1][individual]);
    return 1.0 / min_element;
}
double compute_skill_factor(int individual, vector<vector<int>> &factorial_rank)
{
    // return 0 while task 0 has rank higher task 1
    // return 1 while task 1 has rank higher task 0
    return factorial_rank[0][individual] > factorial_rank[1][individual] ? 0 : 1;
}
void onepoint_crossover(int parent1, int parent2, vector<vector<double>> X, vector<vector<double>> &offspring, int order)
{
    int point1 = rand() % (X[0].size() - 1);
    int i = 0;
    for (i = 0; i < point1; i++)
    {
        offspring[order + 0][i] = X[parent1][i];
        offspring[order + 1][i] = X[parent2][i];
    }
    for (i = point1; i < X[0].size(); i++)
    {
        offspring[order + 0][i] = X[parent2][i];
        offspring[order + 1][i] = X[parent1][i];
    }
}
// void onepoint_crossove
double cal_Beta(double nc, double u)
{
    if (u < 0.5)
    {
        return pow(2 * u, 1.0 / (nc + 1));
    }
    else
        return pow(1.0 / (2 * (1 - u)), 1.0 / (nc + 1));
}
void sbx_crossover(int parent1, int parent2, vector<vector<double>> X, vector<vector<double>> &offspring, int order)
{
    // cout << beta << endl;
    double random = random_double(0, 1);
    double beta = cal_Beta(10, random);
    for (int i = 0; i < X[0].size(); i++)
    {
        offspring[order][i] = 0.5 * ((1.0 + beta) * X[parent1][i] + (1.0 - beta) * X[parent2][i]);
        offspring[order + 1][i] = 0.5 * ((1.0 - beta) * X[parent1][i] + (1.0 + beta) * X[parent2][i]);
    }
}
void poly_mutation(vector<vector<double>> X, vector<vector<double>> &offspring, int parent, int child)
{
    double u = random_double(0, 1);
    double random = random_double(0, 1);
    double nm = 20; 
    double r =1 - pow(2 * (1 - u), 1.0 / (nm + 1));
    double l = pow(2 * u, 1.0/(nm + 1)) - 1;
    for (int i = 0; i < X[parent].size(); i++)
    {

        if (u > 0.5)
        {
            offspring[child][i] = X[parent][i]* ( 1- r) + r;
        }
        else 
            offspring[child][i] = X[parent][i]* ( 1 + l);
        
    }
}
void mutation(vector<vector<double>> X, vector<vector<double>> &offspring, int parent, int child)
{
    float random;
    for (int i = 0; i < D; i++)
    {

        if (random <= pm)
        {
            offspring[child][i] = X[parent][i] + gauss_mutation();
        }
        else
            offspring[child][i] = X[parent][i];
    }
}
void imitate(vector<double> &offspring, int t, int index, vector<vector<double>> &factorial_cost, int order)
{
    if (t == 0)
    {

        factorial_cost[0][index] = sphere_function(offspring);
        factorial_cost[1][index] = INT_MAX;
    }
    else
    {

        factorial_cost[1][index] = ackley_function(offspring);
        factorial_cost[0][index] = INT_MAX;
    }
}
void create_two_child(vector<vector<double>> &X, vector<vector<double>> &offspring, vector<vector<int>> &factorial_rank,
                      vector<vector<double>> &factorial_cost, int &parent1, int &parent2, int order_of_child, vector<bool> &skill_factor_parent, vector<int> &has_two_parent)
{

    int t1 = compute_skill_factor(parent1, factorial_rank);
    int t2 = compute_skill_factor(parent2, factorial_rank);
    // bool skill_factor_parent = true;

    float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    if (t1 == t2 || random < rmp)
    {
        sbx_crossover(parent1, parent2, X, offspring, order_of_child);
        // onepoint_crossover(parent1, parent2, X, offspring, order_of_child);
        skill_factor_parent[order_of_child] = t1;
        skill_factor_parent[order_of_child + 1] = t2;
        has_two_parent[order_of_child] = 1;
        has_two_parent[order_of_child+1] = 1;
    }
    else
    {
        poly_mutation(X, offspring, parent1, order_of_child);
        poly_mutation(X, offspring, parent2, order_of_child + 1);
        skill_factor_parent[order_of_child] = t1;
        skill_factor_parent[order_of_child + 1] = t2;
        has_two_parent[order_of_child] = 0; 
        has_two_parent[order_of_child+1] = 0; 
    }
}

void evaluate_offspring(vector<vector<double>> &factorial_cost, vector<vector<int>> &factorial_rank,
                         vector<vector<double>> &offspring, vector<bool> skill_factor_parent, int order_of_child, vector<int>has_two_parent)
{
    int index = factorial_cost[0].size();
    factorial_cost[0].resize(index + 2);
    factorial_cost[1].resize(index + 2);
    factorial_rank[0].resize(index + 2);
    factorial_rank[1].resize(index + 2);
    float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    

    // Imitate for offspring
    if (has_two_parent[order_of_child])
    {
        if (random < 0.5)
        {
            imitate(offspring[order_of_child], skill_factor_parent[order_of_child], index, factorial_cost, order_of_child);
        }
        else
        {
            imitate(offspring[order_of_child], skill_factor_parent[order_of_child + 1], index, factorial_cost, order_of_child);
        }
    }
    else
    {
        imitate(offspring[order_of_child], skill_factor_parent[order_of_child], index, factorial_cost, order_of_child);
    }
    index += 1;

    if (has_two_parent[order_of_child])
    {
        if (random < 0.5)
        {
            imitate(offspring[order_of_child + 1], skill_factor_parent[order_of_child], index, factorial_cost, order_of_child);
        }
        else
        {
            imitate(offspring[order_of_child + 1], skill_factor_parent[order_of_child + 1], index, factorial_cost, order_of_child);
        }
    }
    else
    {
        imitate(offspring[order_of_child + 1], skill_factor_parent[order_of_child + 1], index, factorial_cost, order_of_child);
    }
}

void assortative_mating(vector<vector<double>> &X, vector<vector<double>> &offspring,
                        vector<vector<int>> &factorial_rank, vector<vector<double>> &factorial_cost, vector<bool> &skill_factor_parent, vector<int> &has_two_parent)
{

    // Create new offspring
    for (int order = 0; order < NUMBER_CHILD; order += 2)
    {
        int parent1 = rand() % POPULATION;
        
        
        int parent2 = parent1;
        while (parent2 == parent1)
        {
            parent2 = rand() % POPULATION;
        }
        create_two_child(X, offspring, factorial_rank, factorial_cost, parent1, parent2, order, skill_factor_parent, has_two_parent);
    }
}

void update(vector<vector<double>> &X, vector<vector<int>> &factorial_rank, vector<vector<double>> &factorial_cost)
{

    int weak1 = 1;
    double value_weak1 = 1;
    int weak2 = 1;
    double value_weak2 = 1;
    for (int i = 0; i < X.size(); i++)
    {
        if (value_weak1 >= compute_scalar_fitness(i, factorial_rank))
        {

            weak2 = weak1;
            weak1 = i;

            value_weak2 = value_weak1;
            value_weak1 = compute_scalar_fitness(i, factorial_rank);
        }
    }

    X.erase(X.begin() + weak1);
    X.erase(X.begin() + weak2);
    factorial_rank[0].erase(factorial_rank[0].begin() + weak1);
    factorial_rank[0].erase(factorial_rank[0].begin() + weak2);

    factorial_rank[1].erase(factorial_rank[1].begin() + weak1);
    factorial_rank[1].erase(factorial_rank[1].begin() + weak2);

    factorial_cost[0].erase(factorial_cost[0].begin() + weak1);
    factorial_cost[0].erase(factorial_cost[0].begin() + weak2);

    factorial_cost[1].erase(factorial_cost[1].begin() + weak1);
    factorial_cost[1].erase(factorial_cost[1].begin() + weak2);

}


void mfea()
{
    // Initialize variable
    int count = 0;
    vector<vector<double>> X; // Population
    vector<vector<double>> factorial_cost;
    vector<vector<int>> factorial_rank;
    vector<vector<double>> offspring;
    vector<bool> skill_factor_parent;
    vector<int> has_two_parent; 

    allocate(offspring, NUMBER_CHILD, D);
    allocate(factorial_rank, 2, POPULATION); // Two tasks :))
    allocate(factorial_cost, 2, POPULATION); //
  

    X.resize(POPULATION);
    skill_factor_parent.resize(NUMBER_CHILD);
    has_two_parent.resize(NUMBER_CHILD);

    // MFEA
    /*
    1.	Generate an initial population of individuals and store it in current-pop (P). 
    2.	Evaluate every individual with respect to every optimization task in the multitasking environment. 
    3.	Compute the skill factor (τ) of each individual.  
    4.	while (stopping conditions are not satisfied) do 
        i.	Apply genetic operators on current-pop to generate an offspring-pop (C). Refer to Algorithm 2. 
        ii.	Evaluate the individuals in offspring-pop for selected optimization tasks only (see Algorithm 3). 
        iii.	Concatenate offspring-pop and current-pop to form an intermediate-pop (P   C).
        iv.	 Update the scalar fitness (φ) and skill factor (τ) of every individual in intermediate-pop. 
        v. Select the fittest individuals from intermediate-pop to 
        form the next current-pop (P). 
    5.	end while 

    */
    // Step 1
    initial_population(X);

    // Step 2
    cal_factorial_cost(X, factorial_rank, factorial_cost);

    // Step 3: Khong thuc hien o day :)
    // print(X);
    // print(factorial_cost);
    // Step 4:
    while (count < LOOP)
    {
        // In ra vong lap thoi
        if (count % 1000 == 0)
            cout << count << endl;

        // Step 4.1
        assortative_mating(X, offspring, factorial_rank, factorial_cost, skill_factor_parent, has_two_parent);

        // Step 4.2
        for (int order = 0; order < NUMBER_CHILD; order += 2)
        {
            evaluate_offspring(factorial_cost, factorial_rank, offspring, skill_factor_parent, order, has_two_parent);
        }

        // Step 4.3
        for (int i = 0; i < offspring.size(); i++)
        {
            X.push_back(offspring[i]);
        }

        // Step 4.4:hmmm, Hoi khac moit ti la factorial cost tinh san o tren roi, chi chuyen tu cost-> rank thoi
        transform_to_rank(factorial_rank, factorial_cost);

        // Step 4.5: Update and select
        for (int i = 0; i < offspring.size(); i += 2)
        {
            update(X, factorial_rank, factorial_cost);
        }
        // print(factorial_cost);
        // cout <<"X = : " << endl;
        // print(X);

        count++;
    }

    // print_min(factorial_cost[0]);
    // print_min(factorial_cost[1]);
    print(X);
    print(factorial_cost);
}