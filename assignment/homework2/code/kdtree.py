from collections import namedtuple
from operator import itemgetter
from math import pow
leaves_show = 0
test_show = 0
dimension = 0
slimit=0
index=0
path2=[]
filepath="/Users/crispus/Desktop/machine learning/hw02/inputData/"


#tree
class treeNode(namedtuple("node", "location left_child right_child")):
     def __repr__(self):
        return "this is tree"


#initial of method
def buildTree():
    print("input file")
    datafile = input()
    inputlist = datafile.split(' ', 1)
    text = inputlist[0]
    global slimit
    try:
       slimit = int(inputlist[1])
    except IndexError:
         print("You need input the integer for set's minimum number after file name")
         buildTree()
    file=filepath+text
    data=dataload(file)
    node=makeTree(data, 0)
    kdtree(node)


#build the tree
def makeTree(data,depth):
    if data=="none":
        return treeNode("none", "none", "none")
    if slimit==0:
        return treeNode("none",data,"none")
    if len(data)<=slimit:
        return treeNode("none", data, "none")
    axis=depth%dimension
    if type(data[0])==float:
        return treeNode("none", data, "none")
    i=0
    s=data[0][axis]
    for n in range(len(data)):
        if data[n][axis]!=s:
            i=1
    if i ==0:
           return treeNode(data[len(data)-1], left_child=makeTree(data,depth+1),right_child=makeTree("none",depth+1))
    data=sorted(data, key=itemgetter(axis))
    mid=int(len(data)/2)
    if len(data)%2==1:
          return treeNode(
               location=data[mid],
               left_child=makeTree(data[:mid+1], depth+1),
               right_child=makeTree(data[mid+1:],depth+1) )
    else:
            return treeNode(
                location=data[mid],
                left_child=makeTree(data[:mid], depth + 1),
                right_child=makeTree(data[mid:], depth + 1))


#midle step to display the tree
def showTree(node):
    displaytree(node,path=[],path2=[],labal="b")


def displaytree(node,path,path2,labal):
    path2.append(labal)
    if type(node[2])==str:
        if node[1]!="none":
             global index
             index=index+1
             stringpath="".join(path2[1:])
             print(str(index)+".  "+stringpath+"  :  ", end='')
             min, max= findBound(node[1])
             print("Bounding Box: "+str(min)+",  "+str(max))
             print("Data in leaf:"+str(node[1]))
             print()
        if len(path2):
             path2.pop()
    else:
        path.append(node[0])
        displaytree(node[1],path,path2,labal="L")
        displaytree(node[2],path,path2,labal="R")
        path2.pop()


#method to find the bound box
def findBound(data):
    if type(data[0])==float:
        return data, data
    i=len(data[0])
    min=[]
    max=[]
    for j in range(i):
        minimum=data[0][j]
        maxmum=data[0][j]
        for n in range(len(data)):
            if data[n][j]<minimum:
                minimum=data[n][j]
            if data[n][j]>maxmum:
                maxmum=data[n][j]
        min.append(minimum)
        max.append(maxmum)
    return min, max


#interface with user
def kdtree(node):
    leaves = input("Print tree leaves? yes or no?, anything else for no:")
    if leaves=="yes":
        showTree(node)
    test = input("Test data? yes or no, anything else for no:")
    if test == "yes":
        testTree(node)
    print("wish you have a great day! Goodbye")


#to find the neareast point
def testTree(node):
    datafile=input("name of data file:")
    test_path=filepath+datafile
    data=dataload(test_path)
    for i in data:
        nearest(i, node,0)


def nearest(test_data,tree,depth):
    split=depth%dimension
    if tree[1]=="none":
        print(str(test_data)+" has no nearest neighbor (in an empty set).")
        print()
    elif type(tree[0])!=str:
            if(test_data[split]<=tree[0][split]):
                nearest(test_data,tree[1],depth+1)
            else:
                nearest(test_data,tree[2],depth+1)
    else:
              point, distance=calculate(tree[1],test_data)
              print(str(test_data)+" is in the set: ",end="")
              print(tree[1])
              print("Nearest neighbor: "+point+"  distance is: "+distance)
              print()

# load the data in file
def dataload(filename):
     data=[]
     try:
       with open(filename) as textdata:
         lines=textdata.readline()
         global dimension
         dimension=int(lines)
         for line in textdata:
             linedata=line.strip('\n').split(' ')
             data.append(linedata)
     except FileNotFoundError:
        print("Your file can not be found,please operate again")
        buildTree()
     floatchange(data)
     return data


#change the string list to float list
def floatchange(list):
    for i in range(len(list)):
        for j in range(len(list[0])):
            a = float(list[i][j])
            list[i][j] = a
    return list


#calculate the minimum distance and return the nearest point
def calculate(dataset, data):
    if type(dataset[0])==float:
        sum=0
        for i in range(len(data)):
            b = abs(data[i]-dataset[i])
            a = pow(b,dimension)
            sum=sum+a
        return str(dataset),str(pow(sum,1/dimension))
    sumall=[]
    for i in range(len(dataset)):
         sum=0
         for j in range(len(data)):
             b=abs(data[j]-dataset[i][j])
             a=pow(b,dimension)
             sum=sum+a
         sumall.append(sum)
    mins=dataset[sumall.index(min(sumall))]
    return str(mins), str(pow(min(sumall), 1/dimension))


if __name__ == '__main__':
    buildTree()