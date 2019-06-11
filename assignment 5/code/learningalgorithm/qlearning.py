from learningalgorithm.fbVersion import *
import numpy as np
import random
import sys
sys.setrecursionlimit(10000)
Directions=["U","D","L","R"]
Rewards=[]


# this is method to load reward in environment
def loadReward():
    reward=[]
    with open("/Users/crispus/Downloads/hw05/pipe_world.txt", 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.strip('\n')
            List=[]
            for character in line :
                 if character=='M':
                    List.append(-100)
                 elif character=='G':
                     List.append(0)
                 else:
                     List.append(-1)
            reward.append(List)
    global Rewards
    Rewards=reward
    return reward


# this is method to generate state-action table
def generateStructure(Rewards):
     state_action=[]
     for i in Rewards:
         for j in i:
            action=[0,0,0,0]
            state_action.append(action)
     return state_action


# this is method to generate which action can go,and their q-value
def possibalDirection(state,reward,state_action):
    direct=[]
    stateindex=state[0]*len(reward[0])+state[1]
    possible_q=[]
    if state[0]-1>=0:
        direct.append(0)
        possible_q.append(state_action[stateindex][0])
    if state[0]+1<len(reward):
        direct.append(1)
        possible_q.append(state_action[stateindex][1])
    if state[1]-1>=0:
        direct.append(2)
        possible_q.append(state_action[stateindex][2])
    if state[1]+1<len(reward[0]):
        direct.append(3)
        possible_q.append(state_action[stateindex][3])
    return direct,possible_q


# this is method to calculate probability to which state in one direction.
def detail(state,deriction,Rewards):
    probability=[0,0,0]
    if deriction==0 or deriction==1:
        if state[1]-1>=0 and state[1]+1<len(Rewards[1]):
            probability=[0.1,0.8,0.1]
        if state[1]-1<0 and state[1]+1>=len(Rewards[1]):
            probability=[0,1,0]
        if state[1]-1>=0 and state[1]+1>=len(Rewards[1]):
            probability=[0.1/0.9,0.8/0.9,0]
        if state[1] - 1 < 0 and state[1] + 1 < len(Rewards[1]):
            probability = [0, 0.8 / 0.9,0.1 / 0.9]
    else:
        if state[0] - 1 >= 0 and state[0] + 1 < len(Rewards):
            probability = [0.1, 0.8, 0.1]
        if state[0] - 1 < 0 and state[0] + 1 >= len(Rewards):
            probability = [0, 1, 0]
        if state[0] - 1 >= 0 and state[0] + 1 >= len(Rewards):
            probability = [0.1 / 0.9, 0.8 / 0.9, 0]
        if state[0] - 1 < 0 and state[0] + 1 < len(Rewards):
            probability = [0, 0.8 / 0.9, 0.1 / 0.9]
    return  probability


# this is the method to update 1-table
def updataQtable( newState, eplision, state_action, alpha, lable):
    if Rewards[newState[0]][newState[1]] == 0:
        return state_action
    if Rewards[newState[0]][newState[1]] == -100:
        return state_action
    if lable >= len(Rewards) * len(Rewards[0]):
        return state_action
    lable = lable + 1
    directions, possible_q = possibalDirection(newState, Rewards, state_action)
    if np.random.random() < eplision:
        action = directions[np.random.randint(0, len(directions))]
    else:
        action = chooseAction(possible_q, directions)
    probability = detail(newState, action, Rewards)
    stateChange = generateDerictiondetail(probability)
    newState2 = updataLocation(action, stateChange, newState)
    reward = Rewards[newState2[0]][newState2[1]]
    directions, possible_q = possibalDirection(newState2, Rewards, state_action)
    state_action[newState[0] * len(Rewards[0]) + newState[1]][action] = \
        state_action[newState[0] * len(Rewards[0]) + newState[1]][action] + \
        alpha * (reward + 0.9 * (max(possible_q)) -state_action[newState[0] * len(Rewards[0]) + newState[1]][action])
    state_action = updataQtable(newState2, eplision, state_action, alpha, lable)
    return state_action


# this is method to choose direction when you draw the final policy
def chooseAction(possible_q,directions):
     max=-1000
     record=[]
     for i in range(len(possible_q)):
         if possible_q[i]>=max:
             max=possible_q[i]
     for j in range(len(possible_q)):
         if possible_q[j]==max:
             record.append(j)
     action=directions[record[np.random.randint(0,len(record))]]
     return action


# this is method depend on probability to generate whether agent will slip
def generateDerictiondetail(probability):
    randomNumber = random.random()
    stateChange = -1
    lable = 0
    for i in range(len(probability)):
        lable = lable + probability[i]
        if randomNumber < lable:
            stateChange = i
            break
    return stateChange


# this is method to change agent current location
def updataLocation(action,statechange,state):
     newstate=[]
     if action==0 and statechange==0:
            newstate=[state[0]-1,state[1]-1]
     if action == 0 and statechange == 1:
             newstate = [state[0] - 1, state[1]]
     if action == 0 and statechange == 2:
             newstate = [state[0] -1, state[1]+1]
     if action == 1 and statechange == 0:
             newstate = [state[0] + 1, state[1]-1]
     if action == 1 and statechange == 1:
             newstate = [state[0] + 1, state[1]]
     if action == 1 and statechange == 2:
             newstate = [state[0] + 1, state[1]+1]
     if action == 2 and statechange == 0:
             newstate = [state[0] - 1, state[1]-1]
     if action == 2 and statechange == 1:
             newstate = [state[0], state[1]-1]
     if action == 2 and statechange == 2:
             newstate = [state[0] + 1, state[1]-1]
     if action == 3 and statechange == 0:
             newstate = [state[0] - 1, state[1] +1]
     if action == 3 and statechange == 1:
             newstate = [state[0], state[1] + 1]
     if action == 3 and statechange == 2:
             newstate = [state[0] + 1, state[1] + 1]
     return newstate


# this is method to test final policy
def testPolicy(state_action):
    initial_state = [0, 18]
    lable=1
    reward=0
    finalreward=stepQtable(initial_state,state_action,lable,reward)
    return finalreward


# this is recursion method to go through maze using learned policy.
def stepQtable( newState, state_action,lable,reward):
    lable=lable+1
    directions, possible_q = possibalDirection(newState, Rewards, state_action)
    action = chooseAction(possible_q, directions)
    probability2 = detail(newState, action, Rewards)
    stateChange2 = generateDerictiondetail(probability2)
    newState2 = updataLocation(action, stateChange2, newState)
    reward=reward + Rewards[newState[0]][newState[1]]
    if Rewards[newState[0]][newState[1]] == 0:
        return reward
    if Rewards[newState[0]][newState[1]] == -100:
        return reward
    if lable > len(Rewards) * len(Rewards[0]):
        return reward
    reward=stepQtable( newState2, state_action, lable,reward)
    return reward


# this is method to control the alpha value,eplision value,and test.
def qlearning():
    lable_updata = 1
    lable_eplision=0
    lable_alpha=0
    lable_test=0
    alpha=0.9
    eplision=0.9
    Rewards=loadReward()
    state_action = generateStructure(Rewards)
    record=[]
    for epsiodes in range(10000):
        lable_eplision=lable_eplision+1
        lable_alpha=lable_alpha+1
        lable_test=lable_test+1
        initial_state = [0, 18]
        if lable_alpha==1000:
            alpha = 1 / (((epsiodes+1) / 1000) + 1)
            lable_alpha=0
        if lable_eplision==200:
            eplision=eplision/(((epsiodes+1)/200)+1)
            lable_eplision=0
        if lable_test==100:
            totolreward=0
            for i in range(50):
                totolreward=totolreward+testPolicy(state_action)
            averagereward=totolreward/50
            record.append(averagereward)
            print("the test result of average reward is :"+str(averagereward))
            lable_test=0
        state_action=updataQtable(initial_state,eplision,state_action,alpha,lable_updata)
    savepolicy(state_action,record)


# this is method when we print final policy, choose what direction in each state
def chooseDirection(i,j,state_action):
    state=i*len(Rewards[0])+j
    maxDirection=max(state_action[state])
    maxindex=state_action[state].index(maxDirection)
    return Directions[maxindex]


# this is method to print final policy and draw the graph for average reward
def savepolicy(state_action,record):
    for i in range(len(state_action)):
        for j in range(len(state_action[0])):
            if state_action[i][j]==0:
                state_action[i][j]=-1000
    enviroment=loadEnviroment()
    for i in range(len(enviroment)):
        for j in range(len(enviroment[1])):
             if enviroment[i][j]=="_" or enviroment[i][j]=="S":
                 direction=chooseDirection(i,j,state_action)
                 enviroment[i][j]=direction
    for i in enviroment:
        print(i)
    draw(record,-100,0)
    draw(record, -1000, 0)


def draw(record,a,b):
    x = np.arange(1, len(record) + 1)
    y = record
    pyplot.ylim((a ,b))
    pyplot.title("Q-table version of Q-learning")
    pyplot.xlabel("rounds of test")
    pyplot.ylabel("average-reward")
    pyplot.plot(x, y)
    pyplot.show()


if __name__ == '__main__':
    qlearning()
