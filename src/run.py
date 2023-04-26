import time
import sys
import torch.utils.data
import tictactoelib 
import numpy as np 
import torch
import torch.nn as nn
from torch.distributions import Categorical
from copy import deepcopy
import tracemalloc


O = 2
NONE=3
TIE=-1
EMPTY = 0
X= 1
device = torch.device("cpu")
def tile_to_name(tile):
    if tile == O:
        return "O"
    if tile==X:
        return "X"
    if tile==EMPTY:
        return "EMPTY"
    return "TIE"
    print(tile)
    raise Exception("classic")
def op_turn(x):
    if x==1:
        return 2
    if x==2:
        return 1
    raise Exception("unkown turn")
a = [[0,0,0] for i in range(3)]

class ResLayer(nn.Module):
    def __init__(self,channels) -> None:
        super().__init__()
        self.one = nn.Sequential(
            nn.Conv2d(channels,channels,3,padding=1),
            nn.BatchNorm2d(channels,eps=1e-5),
            nn.ReLU(channels),
            nn.Conv2d(channels,channels,3,padding=1),
            nn.BatchNorm2d(channels,eps=1e-5),
        )
    def forward(self,x):
        a= nn.ReLU()(self.one(x)+x)
        return a
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
        print("value_error",value_error,"\tpolicy_error",policy_error)
        error = (value_error.flatten()+policy_error).mean()
        return error
class Agent(nn.Module):
    def __init__(self,res_layers,channels,device) -> None:
        super().__init__()
        self.debug_id = -1
        board_channels = 3
        self.value = nn.Sequential(
            nn.Conv2d(channels,1,1,bias=False),
            nn.BatchNorm2d(1,eps=1e-5),
            nn.Flatten(),
            nn.ReLU(),

            nn.Linear(9,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Tanh()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(board_channels,channels,3,bias=False,padding=1),
            nn.BatchNorm2d(channels,eps=1e-5),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Conv2d(channels,2,1,bias=False),
            nn.BatchNorm2d(2,eps=1e-5),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(18,9)
        )
        self.res = ResLayers(channels,res_layers).to(device)
    def forward(self,x):
        a = self.conv(x)
        a = self.res(a)
        value = self.value(a)
        log_probs = self.policy(a)
        policy =Categorical(logits= log_probs)
        return value,policy.probs
    def save(self,name):
        torch.save(self.state_dict(),name)
    def load(self,name):
        try:
            self.load_state_dict(torch.load(name))
        except:
            print("WARNING: Failed to load model "+name)

class DequeDataset(torch.utils.data.Dataset):
    def __init__(self,buffer_size,*shapes) -> None:
        super().__init__()
        data_list = []
        for shape in shapes:
           data_list.append(torch.zeros((buffer_size,)+shape))
        self.data = data_list
        self.max_buffer_size= buffer_size
        self.current_idx = 0
        self.size = 0
    def __getitem__(self, index):
        arr = []
        for row in self.data:
            arr.append(row[(self.current_idx+index)%self.size])
        return arr
    def __len__(self):
        return self.size
    def append(self,*in_t):
        batch_size = in_t[0].shape[0]
        print("batch_size",batch_size)
        for i,t in enumerate(in_t):
            if t.shape[0]!=batch_size:
                raise Exception("irregular batch size")
            if batch_size>=self.max_buffer_size: raise Exception(f"appended too much (size={t.shape[0]}). Max size: {self.max_buffer_size}")
            start = self.current_idx
            end = min(self.current_idx+batch_size,self.max_buffer_size)
            replace_length = end-start 
            self.data[i][start:end] = t[:replace_length]
            
            if self.current_idx+batch_size>self.max_buffer_size:
                stop = (batch_size+self.current_idx)%self.max_buffer_size
                self.data[i][:stop] = t[replace_length:replace_length+stop]
        self.current_idx+=batch_size
        self.current_idx%=self.max_buffer_size
        
        self.size= min(self.size+batch_size,self.max_buffer_size)
        
def dirichlet_noise(p):
    return p
    new_p = .75*p.cpu().detach().numpy()+ .25*np.random.dirichlet(np.full((p.shape[1],),1),size=p.shape[0])
    return new_p


def callback(_data,models,_model_ids):
    
    # print(a)
    # return 1,[1,2,3,4,5,6,7,8,9]
    global current_model
    # print(len(a))
    # print(a)

    try:
        print("CALLBACK")
        i=0 
        """
        fake_policies= []
        fake_values = []
        for i in range(len(_data)):
            fake_policies.append([1/9]*9)
            fake_values.append(1)
        """
        data=  torch.tensor(_data) 
        print("a")
        values = torch.zeros((len(data)),1)
        print("b")
        policies= torch.zeros((len(data)),9)
        print("c")
        model_ids = torch.tensor(_model_ids)
        output_idx = 0
        for id in range(len(models)):
            idx_by_model : torch.Tensor= torch.masked_select(torch.arange(len(data)), model_ids==id).long()
            batch_data= data[idx_by_model]
            if len(batch_data)==0:
                continue
            model = models[id] 
            model.eval()
            print("real model_id:",model.debug_id)
            board= torch.nn.functional.one_hot(batch_data,num_classes=3).float()
    #        print_tensor_boards(board)
            # for i in range(100):
            model_value,model_policy = model(board)
    #        print(policy)
            noised_policy = torch.tensor(dirichlet_noise(model_policy),dtype=torch.float32)
            values[idx_by_model]= model_value
            policies[idx_by_model]= noised_policy
            
        p =values.flatten().tolist(), policies.tolist()
            # print("pythonpolicy:",p[1])
        print("finished callback")
#            print(p)
        
        return p
    except Exception as e:
        import traceback
        print("Exception during model:")    
        print(traceback.format_exc())
    
def print_board(b):
    s = ""
    for i in range(3):
        for k in range(3):
            s+=["-","X","O"][b[i][k]]
            s+="\t"
        s+="\n"
    print(s)
    return s
def invert_board(board_):
    board = deepcopy(board_ )
    for y in range(len(board)):
        for x in range(len(board[0])):
            if board[y][x]!=0:
                board[y][x] = op_turn(board[y][x])
    return board
def boards_to_tensor(a):
    print(torch.tensor(a,device=device,dtype=torch.int32))
    return torch.nn.functional.one_hot(torch.tensor(a,device=device,dtype=torch.int64),num_classes=3).float()
def print_tensor_boards(boards): 
    s = ""
    for board in boards:
        for i in range(3):
            for k in range(3):
                s+=["-","X","O"][board[i,k].argmax()]
                s+="\t"
            s+="\n"
        s+="\n\n"
    print(s)
    return s
def list_to_uniform_distribution(input_list):
    output_list = deepcopy(input_list)
    not_zero_count = 0
    for x in input_list:
        if x!=0:
            not_zero_count+=1
    for i,x in enumerate(input_list):
        if x!=0:
            output_list[i]=1/not_zero_count
    return output_list

class Board:
    def __init__(self) -> None:
        raise Exception("exists as a factory")
    @staticmethod
    def empty():
        return [[0,0,0],[0,0,0],[0,0,0]]
torch.set_printoptions(precision=4,sci_mode=False)



def get_input():
    while True:
        try:
            x=  int(input("Please input left-to-right X position (1-3):").strip())-1
            y=  int(input("Please input your top-to-bottom Y position (1-3):").strip())-1

            if x not in [0,1,2] or y not in [0,1,2]:
                print("invalid position")
                continue
            return x,y 
        except:
            continue


def play(iterations,threads,use_nn):

    current_model = Agent(3,16,device) 
    current_model.to(device)
    current_model.load("cmodel")
# &iterations_per_turn,&callback,&thread_count, &concurrent_games,&total_games, &use_nn, &return_last_move,&one_turn
#boards, values ,policies, turns (who went), is terminals
    answer = input("Input in who you want to be (X/o)")
    tracemalloc.start()
     
    if answer.strip().lower()=="o":
        player_turn = O
    else:
        player_turn = X

    winner = EMPTY
    
    board = Board.empty()
    turn = X
    while winner==EMPTY:
        print_board(board)
         
        if turn==player_turn:
            while True: 
                x,y = get_input()
                if board[y][x]!=EMPTY:
                    print("Please put your tile in an empty spot")
                    continue
                break
            board[y][x] = turn
            winner = tictactoelib.get_board_win_state(board)
            
        else:
    #&arg_model1,&arg_model2,&iterations_per_turn,&callback,&thread_count, &concurrent_games,&total_games,&use_nn_,&return_last_move,&one_turn,&starting_position,&starting_turn)){
            boards, values, policies, turns, is_terminals,winner_tally, tie_tally = tictactoelib.play_multiple_games(current_model,current_model,iterations,callback,threads,1,1,use_nn,True,True,board,op_turn(player_turn))
            if len(boards)>1:
                board = boards[1]
            print(is_terminals) 
            winner = is_terminals[1]
        turn = op_turn(turn)
        print_board(board)

    print("WINNER:",tile_to_name(winner)) 



def train(self_learn: bool,iterations,games_at_once,threads,mem_size): 
    if games_at_once< threads:
        raise Exception("games_at_once needs to be less than threads")

    current_model = Agent(3,16,device) 
    current_model.to(device)
    current_model.load("cmodel")
    current_model.debug_id ="good model" 

    old_model = Agent(3,16,device) 
    old_model.to(device)
    old_model.load("badmodel")

    old_model.debug_id ="bad model" 

    board_shape = (3,3,3)
    move_space = (9,)
    
    memory_dataset = DequeDataset(mem_size,board_shape,move_space,(1,))
    
    while True:
        #DONT USE RETURN WINNING MOVE WHEN TRAINING!!!
        inverted_boards = []
        inverted_policies = []
        inverted_values = []
#&arg_model1,&arg_model2,&iterations_per_turn,&callback,&thread_count, &concurrent_games,&total_games,&use_nn_,&return_last_move,&one_turn,&starting_position,&starting_turn)){
        boards, values, policies, turns, is_terminals,winner_tally, tie_tally = tictactoelib.play_multiple_games(current_model,old_model,iterations,callback,1,threads,games_at_once,self_learn,False,False,None,0)
        print(boards,values,policies,turns,is_terminals,winner_tally,tie_tally)
        board_dataset = 0
        #invert boards
        for board,value,policy,turn,is_terminal in zip(boards,values,policies,turns,is_terminals):
            if turn==O:
                inverted_boards.append(invert_board(board))
            else:
                inverted_boards.append(deepcopy(board))
            inverted_values.append(value)

            if is_terminal==1:
                inverted_policies.append(list_to_uniform_distribution(policy))
            else:
                inverted_policies.append(policy)

        for b,v,p,t, is_terminal in zip(boards,values,inverted_policies,turns,is_terminals):
            print_board(b) 
            print(t)
            print(v)
            print(p,sum(p))
            print(is_terminal)
            print()

        optim = torch.optim.Adam(current_model.parameters(),lr=.001)

        train_values = torch.tensor(inverted_values).view(-1,1)
        train_policies = torch.tensor(inverted_policies)
        train_boards = boards_to_tensor(inverted_boards)
        shuffle_idxs = torch.randperm(train_boards.shape[0])

        train_boards = train_boards[shuffle_idxs]
        train_values = train_values[shuffle_idxs]
        train_policies = train_policies[shuffle_idxs]

        memory_dataset.append(train_boards,train_policies,train_values)
        current_model.train()
        for epoch_idx in range(3):
            print(f"epoch {epoch_idx+1}")
            
            data_loader = torch.utils.data.DataLoader(memory_dataset,batch_size=64,pin_memory=False,shuffle=True)
            print("dataset")
            print(memory_dataset[1])

            for idx,sample in enumerate(data_loader):
                
                batch_policies = sample[1]
                batch_values = sample[2]
                batch_boards = sample[0]
                model_values, model_policies = current_model(batch_boards)
                print_tensor_boards(batch_boards)
                print("values:",model_values,"policies:",model_policies)
                print("correct values:",batch_values,"correct policies:",batch_policies)
        #        print(model_values,model_policies) 
                loss_function = AgentLoss()
                loss: torch.Tensor = loss_function(model_values,batch_values,model_policies,batch_policies)
                print("step",loss)
                print()

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(current_model.parameters(),max_norm=1)
                optim.step()
            print(loss)
            print("not !!!saving")
            print(current_model)
            current_model.save("cmodel")
            print("saved")
        a=[[0,0,0] for i in range(3)]
play(25,5,True)
#train(False,50,500,1,45000)
#train(True,1025,5,1000)
0/0
t = DequeDataset(3,(1,),(2,))
arr = (torch.tensor([[1]]),torch.tensor([[2,3]]))
t.append(*arr)
arr = (torch.tensor([[4]]),torch.tensor([[5,6]]))
t.append(*arr)



arr = (torch.tensor([[10],[20]]),torch.tensor([[30,40],[50,60]]))
t.append(*arr)
print(t.data)
print(len(t))
0/0
play(50,5,True)    

