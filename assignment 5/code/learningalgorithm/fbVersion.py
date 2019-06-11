import numpy as np
from matplotlib import pyplot
from learningalgorithm.qlearning import *
Directions=["U","D","L","R"]
Gate=[]
Enviroment=[]
Rewards=[]


# initial the learning
def startLearnig():
    enviroment=loadEnviroment()
    global Enviroment
    Enviroment = enviroment
    global Rewards
    Rewards=loadReward()
    featureTable=generateFeature()
    weight=[0,0]
    fbLearning(weight,featureTable)


# load the Enviroment.
def loadEnviroment():
    enviroment=[]
    with open("/Users/crispus/Downloads/hw05/pipe_world.txt", 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip('\n')
            List=[]
            for character in line :
                 List.append(character)
                 if character=='G':
                     global Gate
                     Gate=[len(enviroment),len(List)-1]
            enviroment.append(List)
    return enviroment


# method to calculate f1 value of each state's action.
def calculateF1(state,direction):
    f1=0
    if direction==0 and state[0]-1>=0:
        newstate=[state[0]-1,state[1]]
        f1=calulatDistanceToGate(newstate)
    if direction == 1 and state[0] + 1 < len(Enviroment):
        newstate = [state[0] +1, state[1]]
        f1 = calulatDistanceToGate(newstate)
    if direction == 2 and state[1] - 1 >=0:
        newstate = [state[0], state[1]-1]
        f1 = calulatDistanceToGate(newstate)
    if direction == 3 and state[1] +1 < len(Enviroment[0]):
        newstate = [state[0], state[1]+1]
        f1 = calulatDistanceToGate(newstate)
    return f1


# method to calculate f2 value each state's action
def calculateF2(state,direction):
    f2=0
    distance1,distance2=calculateInverseDistance(state)
    if direction==0 and state[0]-1>=0:
        f2=min(distance2,distance1)
    if direction == 1 and state[0] + 1 < len(Enviroment):
        f2=min(distance1,distance2)
    if direction == 2 and state[1] - 1 >=0:
        f2=distance1
    if direction == 3 and state[1] +1 < len(Enviroment[0]):
        f2=distance2
    return f2


# this method help calculate f2
def calculateInverseDistance(state):
    if state[1]==0:
        distance1=0
        distance2 = 1 / (len(Enviroment[0]) - 1 - state[1])
    elif len(Enviroment[0])-1==state[1]:
        distance1 = 1 / state[1]
        distance2=0
    else:
        distance1=1/state[1]
        distance2=1/(len(Enviroment[0])-1-state[1])
    return distance1,distance2


# this method help to calculate f1
def calulatDistanceToGate(state):
    totalStep=Gate[0]+Gate[1]
    distance=(abs(Gate[0]-state[0])+abs(Gate[1]-state[1]))/totalStep
    return distance


# this is method to generate feature_table.
def generateFeature():
    feature_table = []
    for i in range(len(Enviroment)):
        state = []
        for j in range(len(Enviroment[0])):
            List = []
            for f in range(4):
                List.append([calculateF1([i,j],f),calculateF2([i,j],f)])
            state.append(List)
        feature_table.append(state)
    return feature_table


# this is method to get which direction in that state can go ,and return q-value of each direction
def possibalDirection(state,featureTable,weight):
    direct=[]
    possible_q=[]
    if state[0]-1>=0:
        direct.append(0)
        a=weight[0]*featureTable[state[0]][state[1]][0][0]
        b=weight[1]*featureTable[state[0]][state[1]][0][1]
        possible_q.append(a+b)
    if state[0]+1<len(Enviroment):
        direct.append(1)
        a = weight[0] * featureTable[state[0]][state[1]][1][0]
        b = weight[1] * featureTable[state[0]][state[1]][1][1]
        possible_q.append(a + b)
    if state[1]-1>=0:
        direct.append(2)
        a = weight[0] * featureTable[state[0]][state[1]][2][0]
        b = weight[1] * featureTable[state[0]][state[1]][2][1]
        possible_q.append(a + b)
    if state[1]+1<len(Enviroment[0]):
        direct.append(3)
        a = weight[0] * featureTable[state[0]][state[1]][3][0]
        b = weight[1] * featureTable[state[0]][state[1]][3][1]
        possible_q.append(a + b)
    return direct,possible_q


#   This is main part to learn weight(w1,w2). this is recursion function to update weight
def updataWeight(newState,eplision,featureTable,alpha,lable_updata,weight):
    lable=lable_updata+1
    if Enviroment[newState[0]][newState[1]] == 'M':
        return weight
    if Enviroment[newState[0]][newState[1]] == 'G':
        return weight
    if lable > len(Enviroment) * len(Enviroment[0]):
        return weight
    directions, possible_q = possibalDirection(newState, featureTable, weight)
    if np.random.random() < eplision:
        action = directions[np.random.randint(0, len(directions))]
    else:
        action = chooseAction(possible_q, directions)
    probability2 = detail(newState, action, Enviroment)
    stateChange = generateDerictiondetail(probability2)
    newState2 = updataLocation(action, stateChange, newState)
    reward = Rewards[newState2[0]][newState2[1]]
    a=weight[0]*featureTable[newState[0]][newState[1]][action][0]
    b=weight[1]*featureTable[newState[0]][newState[1]][action][1]
    delta=reward+0.9*max(possible_q)-(a+b)
    for i in range(len(weight)):
        weight[i]=weight[i]+alpha*delta*featureTable[newState[0]][newState[1]][action][i]
    weight=updataWeight(newState2,eplision,featureTable,alpha,lable_updata,weight)
    return weight


# this is method beginning test
def testWeight(weight,featureTable):
    initial_state = [0, 18]
    lable=1
    reward=0
    finalreward=stepFtable(initial_state,lable,reward,featureTable,weight)
    return finalreward


#   this is recursion method to go through environment with policy,and record reward.
def stepFtable(newState,lable,reward,featureTable,weight):
    lable=lable+1
    directions, possible_q = possibalDirection(newState, featureTable, weight)
    action = chooseAction(possible_q, directions)
    probability2 = detail(newState, action, Rewards)
    stateChange2 = generateDerictiondetail(probability2)
    newState2 = updataLocation(action,stateChange2, newState)
    reward=reward + Rewards[newState[0]][newState[1]]
    if Rewards[newState[0]][newState[1]] == 0:
        return reward
    if Rewards[newState[0]][newState[1]] == -100:
        return reward
    if lable > len(Rewards) * len(Rewards[0]):
        return reward
    reward=stepFtable(newState2,lable,reward,featureTable,weight)
    return reward


# this is method to control the alpha value,eplision value,and test.
def fbLearning(weight,featureTable):
    lable_updata = 1
    lable_eplision=0
    lable_alpha=0
    lable_test=0
    alpha=0.9
    eplision=0.9
    record=[]
    for epsiodes in range(10000):
        lable_eplision=lable_eplision+1
        lable_alpha=lable_alpha+1
        lable_test=lable_test+1
        initial_state = [0, 18]
        if lable_alpha==1000:
            alpha = alpha / (((epsiodes+1) / 1000) + 1)
            lable_alpha=0
        if lable_eplision==200:
            eplision=eplision/(((epsiodes+1)/200)+1)
            lable_eplision=0
        if lable_test==100:
            totolreward = 0
            for i in range(50):
                totolreward = totolreward +  testWeight(weight,featureTable)
            averagereward = totolreward / 50
            record.append(averagereward)
            print("the test result of average reward is :" + str(averagereward))
            lable_test=0
        weight=updataWeight(initial_state,eplision,featureTable,alpha,lable_updata,weight)
    storeInfo(weight,featureTable,record)


# this is method when we print final policy, choose what direction in each state
def chooseDirection(i,j,weight,featureTable):
    List=[]
    for d in range(4):
        a = weight[0] * featureTable[i][j][d][0]
        b = weight[1] * featureTable[i][j][d][1]
        totol=a+b
        List.append(totol)
    for i in range(len(List)):
        if List[i]==0:
            List[i]=-1000
    maxvalue=max(List)
    dirction=None
    for i in range(len(List)):
        if List[i]==maxvalue:
            dirction=i
    return Directions[dirction]


# this is method to print final policy and draw the graph for average reward
def storeInfo(weight,featureTable,record):
    enviroment = loadEnviroment()
    for i in range(len(enviroment)):
        for j in range(len(enviroment[1])):
            if enviroment[i][j] == "_" or enviroment[i][j] == "S":
                dirct=chooseDirection(i,j,weight,featureTable)
                enviroment[i][j] = dirct
    for i in enviroment:
        print(i)
    x=np.arange(1,len(record)+1)
    y=record
    pyplot.ylim((-60,-30))
    pyplot.title("feature-based version of Q-learning")
    pyplot.xlabel("rounds of test")
    pyplot.ylabel("average-reward")
    pyplot.plot(x, y)
    pyplot.show()


if __name__ == '__main__':
    startLearnig()