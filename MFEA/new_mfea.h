#include "utils.h"

void initial_population(vector<vector<double>> &X)
{
    for (int row = 0; row < POPULATION; row++)
    {
        for (int col = 0; col < D; col++)
        {
            X[row][col] = random_double(-range, range);
        }
    }
}
// void()

void compute_skill_factor(vector<vector<double>> factorial_cost, vector<int> &skill_factor, vector<double> &scalar_fitness)
{
    vector<vector<double>> temp;
    allocate(temp, factorial_cost.size(), factorial_cost[0].size());
    temp = factorial_cost;
    for (int i = 0; i < temp.size(); i++)
    {
        sort(temp[i].begin(), temp[i].end());
    }

    for (int individual = 0; individual < factorial_cost[0].size(); individual++)
    {
        int min_rank = INT_MAX;
        int best_task = -1;
        for (int task = 0; task < NUMBER_TASK; task++)
        {
            int rank = find(temp[task].begin(), temp[task].end(), factorial_cost[task][individual]) - temp[task].begin() + 1;
            if (rank < min_rank)
            {
                min_rank = rank;
                best_task = task;
            }
        }
        skill_factor[individual] = best_task;
        scalar_fitness[individual] = 1.0 / min_rank;
    }
}
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
    double r = 1 - pow(2 * (1 - u), 1.0 / (nm + 1));
    double l = pow(2 * u, 1.0 / (nm + 1)) - 1;
    for (int i = 0; i < X[parent].size(); i++)
    {

        if (u > 0.5)
        {
            offspring[child][i] = X[parent][i] * (1 - r) + r;
        }
        else
            offspring[child][i] = X[parent][i] * (1 + l);
    }
}
void create_two_child(vector<vector<double>> X, vector<vector<double>> &offspring, int parent1, int parent2, int order,
                      vector<int> skill_factor_parent, vector<int> &skill_factor_child)
{
    double random = random_double(0, 1);
    if (skill_factor_parent[parent1] == skill_factor_parent[parent2] || random < rmp)
    {
        sbx_crossover(parent1, parent2, X, offspring, order);

        if (random_double(0, 1) < 0.5)
        {
            skill_factor_child[order] = skill_factor_parent[parent1];
            skill_factor_child[order + 1] = skill_factor_parent[parent2];
        }
        else
        {
            skill_factor_child[order] = skill_factor_parent[parent2];
            skill_factor_child[order + 1] = skill_factor_parent[parent1];
        }
        // if (random_double(0, 1) < 0.5)
        // {
        //     skill_factor_child[order + 1] = skill_factor_parent[parent1];
        // }
        // else
        // {
        //     skill_factor_child[order + 1] = skill_factor_parent[parent2];
        // }
    }
    else
    {
        poly_mutation(X, offspring, parent1, order);
        poly_mutation(X, offspring, parent2, order + 1);

        skill_factor_child[order] = skill_factor_parent[parent1];
        skill_factor_child[order + 1] = skill_factor_parent[parent2];
    }
}

void assortative_mating(vector<vector<double>> &X, vector<vector<double>> &offspring,
                        vector<vector<double>> &factorial_cost, vector<int> skill_factor_parent, vector<int> &skill_factor_children, vector<double> &scalar_fitness)
{
    for (int i = 0; i < NUMBER_CHILD; i += 2)
    {
        int parent1 = rand() % POPULATION;
        while (scalar_fitness[parent1] <= 0.5)
        {
            parent1 = rand() % POPULATION;
        }
        int parent2 = parent1;

        while (parent1 == parent2 )
        {
            parent2 = rand() % POPULATION;
        }

        // create_two_child(X, offspring, factorial_cost, skill_factor_children, skill_factor_parent, has_two_parent);
        create_two_child(X, offspring, parent1, parent2, i, skill_factor_parent, skill_factor_children);
    }
}

double compute_cost_with_task(int i, vector<double> individual)
{
    if (i == 0)
    {
        return rastrigin_function(individual);
    }
    if (i == 1)
    {
        return ackley_function(individual);
    }
    if (i == 2)
    {
        return rastrigin_function(individual);
    }

    return 0;
}

void imitate(vector<double> offspring, int skill_factor, vector<vector<double>> &factorial_cost, int order)
{
    for (int task = 0; task < factorial_cost.size(); task++)
    {
        if (task == skill_factor)
        {
            factorial_cost[task][order] = compute_cost_with_task(task, offspring);
        }
        else
        {
            factorial_cost[task][order] = INT_MAX;
            // factorial_cost[task][order] = compute_cost_with_task(task, offspring);
        }
    }
}
void evaluate_offspring(vector<vector<double>> offspring, 
                        vector<vector<double>> &factorial_cost, vector<int> skill_factor_child, vector<int> has_two_parent)
{
    for (int individual = 0; individual < offspring.size(); individual++)
    {
        imitate(offspring[individual], skill_factor_child[individual], factorial_cost, individual);
    }
}

void update(vector<vector<double>> &X, vector<vector<double>> &factorial_cost,
         vector<int> &skill_factor, vector<double> &scalar_fitness)
{
    // chon danso ca the co scalar fitness tot nhat
    vector<vector<double>> temp;
    allocate(temp, factorial_cost.size(), factorial_cost[0].size());
    // vector<double> scalar_fitness;
    scalar_fitness.resize(X.size());
    allocate(temp, factorial_cost.size(), factorial_cost[0].size());
    temp = factorial_cost;
    for (int i = 0; i < temp.size(); i++)
    {
        sort(temp[i].begin(), temp[i].end());
    }

    for (int individual = 0; individual < factorial_cost[0].size(); individual++)
    {
        int min_rank = INT_MAX;
        int best_task = -1;
        for (int task = 0; task < temp.size(); )
        { 
            int rank = find(temp[task].begin(), temp[task].end(), factorial_cost[task][individual]) - temp[task].begin() + 1;
            if (rank < min_rank)
            {
                min_rank = rank;
                best_task = task;
            }
            temp[task][rank-1] = -100000;
            task += 1;
        }
        skill_factor[individual] = best_task;
        scalar_fitness[individual] = 1.0 / min_rank;
    }
    // count_different(skill_factor);

    while (X.size() > POPULATION)
    {
        int index = min_element(scalar_fitness.begin(), scalar_fitness.end()) - scalar_fitness.begin();

        X.erase(X.begin() + index);

        for (int i = 0; i < NUMBER_TASK; i++)
        {
            factorial_cost[i].erase(factorial_cost[i].begin() + index);
        }

        skill_factor.erase(skill_factor.begin() + index);
        scalar_fitness.erase(scalar_fitness.begin() + index);
    }


}
void evaluate_individual(vector<vector<double>> X, vector<vector<double>> &factorial_cost)
{
    // Cal factorial cost
    for (int individual = 0; individual < X.size(); individual++)
    {
        for (int task = 0; task < NUMBER_TASK; task++)
        {
            factorial_cost[task][individual] = compute_cost_with_task(task, X[individual]);
        }
    }
}

void mfea()
{
    srand(time(NULL));
    int count_loop = 0;
    vector<vector<double>> X;
    vector<vector<double>> offspring;

    vector<vector<double>> factorial_cost;
    vector<int> skill_factor_parent;
    vector<int> has_two_parent;
    vector<int> skill_factor_children;
    vector<vector<double>> factorial_cost_child;
    vector<double> scalar_fitness;

    allocate(offspring, NUMBER_CHILD, D);
    allocate(factorial_cost, NUMBER_TASK, POPULATION);
    allocate(X, POPULATION, D);
    allocate(factorial_cost_child, NUMBER_TASK, NUMBER_CHILD);

    skill_factor_parent.resize(POPULATION);
    skill_factor_children.resize(NUMBER_CHILD);
    has_two_parent.resize(NUMBER_CHILD);
    scalar_fitness.resize(POPULATION);

    // Step 1
    initial_population(X);

    // Step 2
    evaluate_individual(X, factorial_cost);

    // Step 3
    compute_skill_factor(factorial_cost, skill_factor_parent, scalar_fitness);

    // Step 4

    while (count_loop < LOOP)
    {
        //
        assortative_mating(X, offspring, factorial_cost, skill_factor_parent, skill_factor_children, scalar_fitness);

        //
        evaluate_offspring(offspring, factorial_cost_child, skill_factor_children, has_two_parent);

        // Concatnate
        X.insert(X.end(), offspring.begin(), offspring.end());
        for (int i = 0; i < NUMBER_TASK; i++)
        {
            factorial_cost[i].insert(factorial_cost[i].end(), factorial_cost_child[i].begin(), factorial_cost_child[i].end());
        }
        skill_factor_parent.insert(skill_factor_parent.end(), skill_factor_children.begin(), skill_factor_children.end());

        //
        // print(factorial_cost);
        update(X, factorial_cost, skill_factor_parent, scalar_fitness);
        count_loop += 1;
    }
    // print(factorial_cost);
    for (int task = 0; task < NUMBER_TASK; task++)
    {
        print_min(factorial_cost[task]);
    }
    print(X); 
    print(factorial_cost);
    cout << "X: " << X.size() << endl;
}