import time
import numpy as np 
import torch
import torch.nn as nn
from torch.distributions import Categorical
def op_turn(x):
    if x==1:
        return 2
    if x==2:
        return 1
    raise Exception("unkown turn")
import tictactoelib as t
a = [[0,0,0] for i in range(3)]

class ResLayer(nn.Module):
    def __init__(self,channels) -> None:
        super().__init__()
        self.one = nn.Sequential(
            nn.Conv2d(channels,channels,1),
            nn.BatchNorm2d(channels),
            nn.ReLU(channels),
            nn.Conv2d(channels,channels,1),
            nn.BatchNorm2d(channels),
        )
    def forward(self,x):
        return nn.ReLU()(self.one(x)+x)
class ResLayers(nn.Module):
    def __init__(self,channels,size) -> None:
        super().__init__()
        self.size = size
        for i in range(size):
            setattr(self,f"r{i}",ResLayer(channels))
    def forward(self,x):
        for i in range(self.size):
            x = getattr(self,f"r{i}")(x)
        return x
class AgentLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,values,winner,pred_probs,probs):
        value_error = (values-winner).pow(2)
        policy_error = torch.sum((-probs*(pred_probs+1e-6).log()),1)
        error = (value_error.flatten()+policy_error).mean()
        return error
class Agent(nn.Module):
    def __init__(self,res_layers,channels,device) -> None:
        super().__init__()
        board_channels = 3
        self.value = nn.Sequential(
            nn.Conv2d(channels,1,1),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.ReLU(),

            nn.Linear(9,1),
            nn.Tanh()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(board_channels,channels,1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Conv2d(channels,2,1),
            nn.BatchNorm2d(2),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(18,9)
        )
        self.res = ResLayers(channels,res_layers).to(device)
    def forward(self,x):
        a = self.conv(x)
        a = self.res(a)
        value = self.value(a)
        policy =Categorical(logits= self.policy(a))
        return value, policy.probs,
    def save(self,name):
        torch.save(self.state_dict(),name)
    def load(self,name):
        self.load_state_dict(torch.load(name))
device = torch.device("cuda")
current_model = Agent(15,256,device) 
current_model.to(device)
def dirichlet_noise(p):
    new_p = .75*p.cpu().detach().numpy()+ .25*np.random.dirichlet(np.full((p.shape[1],),.03),size=p.shape[0])
    return new_p
def callback(a):
    
    # print(a)
    # return 1,[1,2,3,4,5,6,7,8,9]
    global current_model
    # print(len(a))
    # print(a)
    try:
        board= torch.nn.functional.one_hot(torch.tensor(a,device=device),num_classes=3).float()

        # for i in range(100):
        value,policy = current_model(board)
        p = value.flatten().tolist(),dirichlet_noise(policy).tolist()
    except Exception as e:
        
        print(e)
        time.sleep(5)
    # print("pythonpolicy:",p[1])
    return p
def print_board(b):
    s = ""
    for i in range(3):
        for k in range(3):
            s+=["-","X","O"][b[k][i]]
            s+="\t"
        s+="\n"
    print(s)
    return s


threads=1
iterations=100
# &iterations_per_turn,&callback,&thread_count, &concurrent_games,&total_games
print(t.play_multiple_games(500000,callback,7,1,1))
a=[[0,0,0] for i in range(3)]

# a[0][0]=1
# print(a)
# a[0][1]=2
# s = time.perf_counter()
# win_state = 0
# print_board(a)
# turn =1 
# while win_state==0:
    
#     a,win_state,policy= t.play_tic_tac_toe(a,turn,iterations,callback,threads)
#     print(policy)
#     turn=op_turn(turn)
#     print_board(a)

# print("time:",time.perf_counter()-s)
# # print_board(a)
# ti = time.perf_counter()-s

# a=[[0]*3]*3
# s = time.perf_counter()
# win_state = 0
# print_board(a)
# turn = 1
# while win_state==0:
#     a,win_state,policy= t.play_tic_tac_toe(a,turn,iterations*threads,callback,1)
#     # print(policy)
#     turn=op_turn(turn)
#     # print_board(a)
# # myprint("heal")

# print("time:",time.perf_counter()-s)
# print("saved percentage:",str(int((time.perf_counter()-s)/ti*100))+"%")