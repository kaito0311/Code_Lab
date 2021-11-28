#include <bits/stdc++.h>
using namespace std;
#define SIZE_POPULATION 100
#define DIMENSIONS 51
float pm = 0.001;
#define ITERATION 10000
vector<vector<int>> vertice;

int seed = 0;
int convert(string s)
{
    stringstream conv(s);
    int x;
    conv >> x;
    return x;
}
void split(string text, string delimiter)
{
    string token;
    size_t pos = 0;
    int index = -1, x, y;
    while ((pos = text.find(delimiter)) != string ::npos)
    {
        token = text.substr(0, pos);
        // cout << token << endl;
        if (index == -1)
        {
            index = convert(token);
        }
        text.erase(0, pos + delimiter.length());
    }
    x = convert(token);
    y = convert(text);
    vertice[index].resize(3);
    vertice[index][0] = x;
    vertice[index][1] = y;
}

void readfile()
{
    int print = 0;

    ifstream readfile("data.txt");
    string text;

    vertice.resize(DIMENSIONS + 1);
    // cout << vertice.size() << endl;
    int d, b, c;
    string delimiter = " ";
    while (!readfile.eof())
    {

        getline(readfile, text);
        // cout << "file: " << text << endl;

        if (text == "NODE_COORD_SECTION")
        {
            // cout << "change" << endl;
            print = 1;
            continue;
        }
        if (print == 1)
        {
            if (text == "EOF")
                break;
            split(text, delimiter);
        }
    }
}

double cal_dis(int index1, int index2)
{
    // index1 += 1;
    // index2 += 1;
    double a = (vertice[index1][0] - vertice[index2][0]) * (vertice[index1][0] - vertice[index2][0]) + (vertice[index1][1] - vertice[index2][1]) * (vertice[index1][1] - vertice[index2][1]);
    return fabs(sqrt(a));
}

double fitness(vector<int> &individual)
{
    double sum = 0;
    for (int i = 0; i < individual.size(); i++)
    {
        sum += cal_dis(individual[i % DIMENSIONS], individual[(i + 1) % DIMENSIONS]);
    }
    return -sum;
}

void initial_population(vector<vector<int>> &population)
{
    // srand(seed++);
    for (int i = 0; i < SIZE_POPULATION; i++)
    {
        population[i].resize(DIMENSIONS);
        for (int j = 1; j <= DIMENSIONS; j++)
        {
            population[i][j - 1] = j;
        }
        random_shuffle(population[i].begin(), population[i].end());
    }
}

int choose_parent(vector<vector<int>> &population)
{
    // srand(time(NULL));
    // srand(seed++);
    // cout << seed << endl;
    int parent1 = rand() % SIZE_POPULATION;
    int parent2 = 0;
    do
    {

        parent2 = rand() % SIZE_POPULATION;
    } while (parent1 == parent2);
    return fitness(population[parent1]) > fitness(population[parent2]) ? parent1 : parent2;
}

void crossover(vector<vector<int>> &population, int parent1, int parent2)
{
    // srand(seed++);
    // cout << parent1 << " " << parent2 << endl;

    int point1 = rand() % (DIMENSIONS - 1);
    // cout << "point1 : " << point1 << endl;

    vector<int> child;
    child.resize(DIMENSIONS);

    int i = 0;
    for (i = 0; i < point1; i++)
    {
        // cout << "i: " << i << endl;
        child[i] = population[parent1][i];
    }
    for (int j = 0; j < DIMENSIONS; j++)
    {
        int k = 0;
        for (k = 0; k < i; k++)
        {
            if (population[parent2][j] == child[k])
                break;
        }
        if (k == i)
        {
            // cout << "i: " << i << endl;
            child[i] = population[parent2][j];
            i++;
        }
    }
    population.push_back(child);
}

void crossover2(vector<vector<int>> &population, int parent1, int parent2)
{
    int point1, point2;
    point1 = rand() % (DIMENSIONS - 5);
    vector<int> child1;
    vector<int> child2;

    point2 = point1 + 4;
    // Step 1: child1 se nhan gen tu point1 ->point2 cua parent2 va child2 se nhan gene tu point1 -> point2 cua parent1
    child1.resize(DIMENSIONS);
    child2.resize(DIMENSIONS);
    for (int i = point1; i < point2; i++)
    {
        child1[i] = population[parent2][i];
        child2[i] = population[parent1][i];
    }
    // Step 2:
    for (int i = 0; i < DIMENSIONS; i++)
    {
        if (i < point1 || i >= point2)
        {
            if (population[parent2].begin() + point2 == find(population[parent2].begin() + point1, population[parent2].begin() + point2, population[parent1][i]))
            {
                child1[i] = population[parent1][i];
            }
            else
                child1[i] = -1;

            if (population[parent1].begin() + point2 == find(population[parent1].begin() + point1, population[parent1].begin() + point2, population[parent2][i]))
            {
                child2[i] = population[parent2][i];
            }
            else
                child2[i] = -1;
        }
    }
    int j1 = 0;
    int j2 = 0;
    // Step 3: Fill
    for (int i = 0; i < DIMENSIONS; i++)
    {
        if (end(child1) == find(child1.begin(), child1.end(), population[parent2][i]))
        {
            while(child1[j1] != -1){
                j1 += 1;
            }
            child1[j1] = population[parent2][i];
        }

        if (end(child2) == find(child2.begin(), child2.end(), population[parent1][i]))
        {
            while(child2[j2] != -1){
                j2 += 1;
            }
            child2[j2] = population[parent1][i];
        }
    }

    // check
    if (end(child1) != find(child1.begin(), child1.end(), -1))
    {
        cout << " Loi roi ban oi" << endl;
    }
    if (end(child2) != find(child2.begin(), child2.end(), -1))
    {
        cout << " Loi :)) ";
    }

    population.push_back(child1);
    population.push_back(child2);
}

bool compare(vector<int> x1, vector<int> x2)
{

    return fitness(x1) > fitness(x2);
}

double print_population(vector<vector<int>> population, int print = 1)
{
    double average = 0;
    int count = 0;
    for (auto i = population.begin(); i != population.end(); i++)
    {
        for (int j = 0; j < DIMENSIONS; j++)
        {
            if (print == 1)
                cout << (*i)[j] << " ";
        }
        average += fitness((*i));
        count += 1;
        if (print == 1)
        {
            cout << fabs(fitness((*i))) << " ";
            cout << endl;
        }
    }
    return fabs(average) / count;
}
void mutation(vector<vector<int>> &population)
{
    // srand(seed++);
    for (int i = 0; i < SIZE_POPULATION; i++)
    {
        float x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        if (x < pm)
        {
            // cout << "mutation" << endl;
            int number = rand() % SIZE_POPULATION;
            int pos1 = rand() % DIMENSIONS;
            int pos2 = rand() % DIMENSIONS;
            int x = population[number][pos1];
            population[number][pos1] = population[number][pos2];
            population[number][pos2] = x;
        }
    }
}

int main()
{
    //
    srand(time(NULL));
    vector<vector<int>> population;
    population.resize(SIZE_POPULATION);

    readfile();
    initial_population(population);
    // compute fitness

    int iteration = 1;
    // print_population(population);
    cout << endl;
    double prev_ans = 0;
    double curr_ans = 0;
    int count =0; 
    while (iteration < ITERATION)
    {
        if (iteration % 1000 == 0)
            cout << "iteration: " << iteration << endl;

        iteration += 1;

        // Choose parent
        int parent1 = choose_parent(population);
        int parent2 = -1;
        do
        {
            parent2 = choose_parent(population);
            // cout <<"hmmm" << endl;
            // cout << parent1 << "  " << parent2 << endl;
        } while (parent2 == parent1);

        // Crossover
        crossover2(population, parent1, parent2);
        // crossover(population, parent1, parent2);
        // Mutation
        mutation(population);
        // Compute new generation
        sort(population.begin(), population.end(), compare);
        population.erase(population.begin() + SIZE_POPULATION, population.end());
        prev_ans = curr_ans;
        curr_ans =  print_population(population, 0);
        cout << curr_ans  << "  ";
        double delta = prev_ans - curr_ans;
        if(delta < 0){
        }
        else{
            if(delta < 0.1){
                count += 1;
                if(count % 10 ==0){
                    pm += 0.0001;
                }
            }
        }
        cout << pm << endl;
    }
    // print_population(population);
    // cout << count << endl;
}