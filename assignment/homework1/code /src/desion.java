import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Scanner;
public class desion {
    private ArrayList<Integer> lable = new ArrayList<Integer>();
    private ArrayList<ArrayList<Character>> date = new ArrayList<ArrayList<Character>>();//all the data from mashroom.txt
    private ArrayList<ArrayList<Character>> test = new ArrayList<ArrayList<Character>>();//test case
    private ArrayList<String> stastic=new ArrayList<>();
    private static int verbose_model=0;
    private String what;
    public static void main(String args[]) throws IOException {
        String input = "/Users/crispus/Documents/mushroom_data.txt";
        System.out.println("please input a  a positive multiple of 250 that is <= 1000 to be the training set's size");
        Scanner sc = new Scanner(System.in);
        int size = sc.nextInt();
        while (size < 250 || size > 1000 || size % 250 != 0) {
            System.out.print("please reenter a positive multiple of 250 that is <= 1000 :");
            size = sc.nextInt();
        }
        System.out.println(size);
        System.out.print("Please enter a training increment (either 10, 25, or 50):");
        int increment = sc.nextInt();
        while ((increment != 10 && increment != 25 && increment != 50)==true) {
            System.out.print("please reenter  a training increment (either 10, 25, or 50):");
            increment=sc.nextInt();
        }
        System.out.println(increment);
        System.out.println("Verbose mode? yes or no");
        String verbose=sc.next();
        if(verbose.equals("yes")){
            verbose_model=1;
        }
        desion d = new desion(input, size, increment);
    }

    public desion(String path, int tSetSize, int increment) {
        int addition = increment;
        try {
            getdata(path);
        } catch (Exception e) {
            e.printStackTrace();
        }
         System.out.println("Loading Property Information from file.");
         System.out.println("Loading Data from database.");
         System.out.println();
        while (increment <=tSetSize) {
            test.clear();
            for (int i = 0; i < date.size(); i++) {
                test.add(date.get(i));
            }
            ArrayList<ArrayList<Character>> a = new ArrayList<ArrayList<Character>>();
            for (int i = 0; i < increment; i++) {
               int rd=(int)(Math.random()*test.size());
                a.add(date.get(rd));
                test.remove(rd);
            }
            System.out.println("Running with "+increment+" in training set");
            calculateAccurancy(a);
            increment = increment + addition;
        }
       if (verbose_model==1) {// print the tree's information
           print(date);
       }
       printStatic(stastic);//print statistic
    }

    //This is method to get data from file
    private void getdata(String path) throws FileNotFoundException {
        String s;
        int i = 0;
        try {
            FileInputStream inputStream = new FileInputStream(path);
            InputStreamReader reader = new InputStreamReader(inputStream, "UTF-8");
            BufferedReader br = new BufferedReader(reader);
            while ((s = br.readLine()) != null) {
                String[] c = s.split(" ");

                ArrayList<Character> ch = new ArrayList<>();
                for (int j = 0; j < c.length; j++) {
                    char[] chars = c[j].toCharArray();
                    Character character = chars[0];
                    ch.add(character);
                }
                date.add(ch);
                i++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    class treeNode {
        private int cname;
        private String sname;

        public treeNode(String s) {
            sname = s;
        }
        public treeNode(int c) { cname = c; }
        public int getCname() {
            return cname;
        }
        public String getSname() {
            return sname;
        }

        ArrayList<Character> include = new ArrayList<Character>();
        ArrayList<treeNode> node = new ArrayList<treeNode>();
    }

    private treeNode makeTree(ArrayList<ArrayList<Character>> dat) {
        if (dat.size() == 0) {
            treeNode tr = new treeNode("this is empty");
            return tr;
        }
        if (lable.size() == dat.get(0).size() - 1) {
           lable.removeAll(lable);
        }

        int index = new calculateUtil().importance(dat, lable);
        lable.add(index);
        treeNode node = new treeNode(index);
        ArrayList<Character> attribute = new ArrayList<>();
        attribute.add(dat.get(0).get(index));
        ArrayList<ArrayList<ArrayList<Character>>> totle = new ArrayList<ArrayList<ArrayList<Character>>>();
        for (int i = 1; i < dat.size(); i++) {
            if (attribute.contains(dat.get(i).get(index)) == false) {//get all the type of attribute
                attribute.add(dat.get(i).get(index));

            }
        }
        for (int i = 0; i < attribute.size(); i++) {
            ArrayList<ArrayList<Character>> everyLable = new ArrayList<ArrayList<Character>>();
            node.include.add(attribute.get(i));
            for (int n = 0; n < dat.size(); n++) {
                if (attribute.get(i).equals( dat.get(n).get(index))==true) {
                    everyLable.add(dat.get(n));
                }
            }
            totle.add(everyLable);//all the data with different attribute.
            int same;
            Character eatable = totle.get(i).get(0).get(totle.get(i).get(0).size() - 1);
            for (same = 1; same < totle.get(i).size(); same++) {
                if (eatable.equals( totle.get(i).get(same).get(totle.get(i).get(same).size() - 1))==false) {
                    break;
                }
            }
            if (same == totle.get(i).size()) {// check whether node is the end of tree
                if (totle.get(i).get(0).get(totle.get(i).get(0).size() - 1).equals('e')) {
                    treeNode td = new treeNode("eatable");
                    node.node.add(td);
                }   if (totle.get(i).get(0).get(totle.get(i).get(0).size() - 1).equals('p')) {
                    treeNode td = new treeNode("posionous");
                    node.node.add(td);
                }

            } else {
                node.node.add(makeTree(totle.get(i)));
            }
        }
        return node;
    }

    //print all the branch of tree
    public void print(ArrayList<ArrayList<Character>> dat) {
        index=0;
        System.out.println("Final desion tree is");
        System.out.println();
        treeNode node=null;
        node = makeTree(dat);//类
      ArrayList<String> path=new ArrayList<>();
        put(node,path);
    }
    private int index=0;
    public void put(treeNode node,ArrayList<String> path) {
        if(node.getSname()!=null){
            path.add("   "+node.getSname());
            System.out.print("branch"+index);
            printArrary(path);
            System.out.println();
            path.remove(path.size()-1);
            path.remove(path.size()-1);
            index++;
            return;
        }
        String s="  Attribute is:#"+node.getCname();
        path.add(s);
          for (int i=0;i<node.node.size();i++){
              String a=" "+node.include.get(i)+";";
              path.add(a);
              put(node.node.get(i),path);
          }

    }
    public void printArrary(ArrayList<String> path){
             for (int i=0;i<path.size();i++){
                 System.out.print(path.get(i));

             }
    }

    //this method is to calculate the accuracy
    public void calculateAccurancy(ArrayList<ArrayList<Character>> arrayLists) {
        //System.out.println(arrayLists.size());
        treeNode node = makeTree(arrayLists);
        int right = 0;
        for (int i = 0; i < test.size(); i++) {
            what=null;
            Character c=new Character('n');
            String string = test(node, test.get(i));
            if(what==null){
                c='n';
            }
            else if(what.equals("eatable")){
                c='e';
            }
            else if(what.equals("posionous")){
                c='p';
            }
            if (c.equals(test.get(i).get(test.get(i).size() - 1))) {
                right++;
            }
        }
        double accuracy = (double) right / test.size();
        DecimalFormat df = new DecimalFormat();
        df.setMinimumFractionDigits(4);
        String s="Training set size: "+arrayLists.size()+". Success:  "+df.format(accuracy*100)+"percent.";
        stastic.add(s);
       System.out.println("Given current tree, there are "+right+" correct classifications out of "+test.size()+"(a success rate of "+df.format(accuracy*100)+" percent)");
       lable.removeAll(lable);
       System.out.println();
    }

     public void printStatic(ArrayList<String> stastic){
        System.out.println();
        System.out.println("Statistics is:");
        System.out.println();
          for (int i=0;i<stastic.size();i++){
              System.out.println(stastic.get(i));
          }
     }
    //this method want to get the predict of the result
    public String test(treeNode node, ArrayList<Character> tes) {
        int index =node.getCname();
        String s="not found";
                for (int i = 0; i < node.include.size(); i++) {
                    if (node.include.get(i)!=tes.get(index)) {
                        continue;
                    }
                    if (node.node.get(i).getSname()!=null){
                        s=node.node.get(i).getSname();
                        what=s;//kind of result
                        return s;
                    }else {
                        test(node.node.get(i),tes);
                    }
                }
                return s;
            }
        }