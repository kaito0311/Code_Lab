package ga;

public class point {
    public int x, y, id;
    public point(){
        this.id = 0;
        this.x = 0; 
        this.y = 0;
    }
    public point(int id, int x, int y){
        this.id = id;
        this.x = x;
        this.y = y;
    }
    public void print(){
        System.out.println(id + " " + x + " " + y);
    }
    
}
