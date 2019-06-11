import java.util.ArrayList;

public class calculateUtil {
    private int toedible=0;
    private double toentrophy;

    public int importance(ArrayList<ArrayList<Character>> dat,ArrayList<Integer> lable){
        ArrayList<Double> gain=new ArrayList<>();

        for (int i=0;i< dat.size();i++){
             if(dat.get(i).get(dat.get(i).size()-1).equals('e')){
                 toedible++;
             }
        }
        double a=(double)toedible/dat.size();
        double b=(double)(dat.size()-toedible)/dat.size();
        if (a!=0.0&&b!=0) {
            toentrophy = -(a * (Math.log(a) / Math.log(2.0)) + b * (Math.log(b) / Math.log(2.0)));
        }else{
            toentrophy=0;
        }
        ArrayList<Integer> station=new ArrayList<>();
        for(int i=0;i<dat.get(0).size()-1;i++){
            if(lable.contains(i)==false){
                gain.add(toentrophy+entrophy(i,dat));
                station.add(i);
            }
        }
        double max=0;
        int number=0;
        for(int m=0;m<gain.size();m++){
            if (gain.get(m)>max){
                max=gain.get(m);
                number=m;
            }
        }
        int num=station.get(number);
        return num;
    }
    public double entrophy(int index,ArrayList<ArrayList<Character>> dat){
        double ent=0.0;
             ArrayList<Character> kind=new ArrayList<>();//record different type of one attribute
             ArrayList<ArrayList<Character>> record=new ArrayList<ArrayList<Character>>();

             kind.add(dat.get(0).get(index));

             for(int h=0;h<dat.size();h++){
                 if (kind.contains(dat.get(h).get(index))==false){
                     kind.add(dat.get(h).get(index));
                 }

             }
             for(int m=0;m<kind.size();m++){
                 ArrayList<Character> store=new ArrayList<>();
                  for(int j=0;j<dat.size();j++){
                      if(dat.get(j).get(index)==kind.get(m))
                      {
                          store.add(dat.get(j).get(dat.get(j).size()-1));

                      }
                  }
                         record.add(store);

             }
             ArrayList<Integer> re=new ArrayList<>();//record how many entity in one type
             for(int i=0;i<record.size();i++){
                 int n=0;
                   for (int m=0;m<record.get(i).size();m++){
                      n++;
                   }
                   re.add(n);
             }
                 ArrayList<Double> result = calculate(record);//each type's entrophy
                 int totol=0;
                 for (int m=0;m<record.size();m++){
                     for(int i=0;i<record.get(m).size();i++){
                         totol++;
                     }
                 }

                 for(int i=0;i<result.size();i++){
                     double c=(double)re.get(i)/totol;
                     ent=ent+c*result.get(i);
                 }
                 return ent;
    }

    //This is method to calculate the probability.
    public ArrayList<Double> calculate(ArrayList<ArrayList<Character>> record){
            ArrayList<Double> recordForPro= new ArrayList<Double>();
            for(int i=0;i<record.size();i++){
                 int edibal=0;
                int posion=0;
                double probability;
                for (int j=0;j<record.get(i).size();j++){
                       if (record.get(i).get(j)=='e'){
                          edibal++;
                       }
                    if (record.get(i).get(j)=='p'){
                        posion++;
                    }
                }

               double a=(double)edibal/(posion+edibal);
               double b=(double)posion/(posion+edibal);
               if(a!=0.0&&b!=0.0) {
                   probability = a * (Math.log(a) / Math.log(2.0)) + b * (Math.log(b) / Math.log(2.0));
               }else{
                   probability=0;
               }
               recordForPro.add(probability);

            }
           return recordForPro;
    }
}
