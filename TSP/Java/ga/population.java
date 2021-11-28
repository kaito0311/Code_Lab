package ga;

import java.util.ArrayList;
import java.util.Random;

import javax.naming.event.NamingExceptionEvent;

public class population {
    ArrayList<individual> member = new ArrayList<individual>();

    public population(int size_population, int dimensions) {
        for (int individual = 0; individual < size_population; individual += 1) {
            member.add(individual, individualInit(dimensions));
        }
    }

    individual individualInit(int dimensions) {
        ArrayList<Integer> tmp = new ArrayList<Integer>();
        ArrayList<Integer> individual = new ArrayList<Integer>();
        for (int i = 0; i < dimensions; i++) {
            tmp.add(i);
        }
        Random rand = new Random();
        while (tmp.size() > 0) {
            int index = rand.nextInt(tmp.size());
            individual.add(tmp.get(index));
            tmp.remove(index);
        }
        return new individual(individual);

    }

    public double getAverageFitness() {
        double sum = 0;
        for (int i = 0; i < member.size(); i++) {
            sum += member.get(i).getFitness();
        }
        return sum / member.size();
    }

    void onepoint_crossover(population offspring, int parent1, int parent2){
        Random rand = new Random();
        int point1 = rand.nextInt(member.get(parent1).individual.size());
        

    }

    population create_offspring(population offspring, int number_child) {
        individual child1 = new individual();
        individual child2 = new individual();
        Random rand = new Random();

        for (int child = 0; child < number_child; child += 2) {
            int parent1 = rand.nextInt(member.size());
            int parent2 = parent1;
            while (parent1 == parent2) {
                parent2 = rand.nextInt(member.size());
            }



        }
        offspring.member.remove(offspring.member.size() - 1);
        return offspring;
    }

    public void show_population() {
        for (int index = 0; index < member.size(); index += 1) {
            member.get(index).show_individual();
        }
        System.out.println("");
    }

}
