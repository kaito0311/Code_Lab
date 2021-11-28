package ga;

import java.util.ArrayList;

public class Main {
    public static void main (String[] args) throws Exception{
       utils test = new utils();
       ArrayList<point> points = new ArrayList<point>();
       
       test.ReadFile("data1.txt", points);

       for(int i = 0; i < points.size(); i++){
           points.get(i).print();
       }

       utils.distance = test.distance(points);

    //    for(int row = 0; row < points.size(); row +=1 ){
    //        for(int col = 0; col < points.size(); col += 1){
    //            System.out.println(utils.distance.get(row).get(col));
    //        }
    //    }
        population popu = new population(2,3);
        population offspring = new population(3,3);
        popu.create_offspring(offspring, 3);
        popu.member.addAll(offspring.member);
        System.out.println(popu.member.size());
    }
    
}
