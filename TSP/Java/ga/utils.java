package ga;

import java.io.BufferedReader;
import java.io.File;

import java.io.FileReader;
import java.util.ArrayList;


public class utils {
    // String file_path = "data.txt";
    public static ArrayList<ArrayList<Double>> distance;

    public void ReadFile(String file_path, ArrayList<point> points) throws Exception {
        try{
            File f = new File(file_path);
            FileReader read_file = new FileReader(f);

            BufferedReader buffer = new BufferedReader(read_file);
            
            String line;
            while((line = buffer.readLine()) != null){
                
                String[] result = line.split(" ");
                // System.out.println(result[0] + result[1] + result[2]);
                points.add(new point(Integer.parseInt(result[0]), Integer.parseInt(result[1]), Integer.parseInt(result[2])));
            }
            
            read_file.close();
            buffer.close();
        }
        catch(Exception e){
            System.out.println("Khong doc duoc file : " + e);
        }
        
    }
    Double cal_dis_euclid(point point1,point point2){
        return Double.valueOf(Math.sqrt((point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y)* (point1.y - point2.y)));
    }

    ArrayList<ArrayList<Double>> distance(ArrayList<point> points){
        ArrayList<ArrayList<Double>> dis = new ArrayList<ArrayList<Double>>();
        for(int row = 0; row < points.size(); row+= 1){
            dis.add(new ArrayList<Double>());
            for(int col = 0; col < points.size(); col += 1){
                dis.get(row).add(col, cal_dis_euclid(points.get(row), points.get(col)));
            }
        }
        return dis;
    }

}
