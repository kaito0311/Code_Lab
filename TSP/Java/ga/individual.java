package ga;

import java.util.ArrayList;

public class individual {
    public ArrayList<Integer> individual = new ArrayList<Integer>();

    public individual(ArrayList<Integer> individual){
        for(int i =0; i < individual.size(); i++){
            individual.add(i, individual.get(i));
        }
    }
    public individual(){

    }
    
    public double getFitness(){
        double sum = 0; 
        for(int i = 0; i < individual.size(); i++){
            sum += utils.distance.get(individual.get(i)).get(individual.get((i+1) % individual.size()));
        }
        return sum;
    }
    public void show_individual(){
        for(int i = 0; i < individual.size(); i++){
            System.out.print(individual.get(i).toString() + " ");
        }
        System.out.println("");
    }
    
    
}
