#pragma once
#include "Python.h"
#include <memory>
#include <stdexcept>
#include <vector>
#include <condition_variable>
#include <array>
#include <atomic>
#include <shared_mutex>
#include <mutex>
using std::shared_ptr;
using std::vector;

using Action = char;

struct Pos{
    int x;
    int y;
};
enum Turn{
    EMPTY=0,
    X=1,
    O=2,
    TIE=-1,
    NONE=3
};
class TicTacToe{
public:
    TicTacToe();
    // TicTacToe(TicTacToe& t);
    // TicTacToe& operator=(TicTacToe& b);
    TicTacToe(PyObject* o,Turn t);
    void inline set_idx(Pos pos,Turn value){
        board[pos.y][pos.x]=value;
    }
    Turn inline get_idx(Pos pos){
        if(pos.y>3 || pos.x>3)
            throw std::runtime_error("Accessed illegal position");
        return board[pos.y][pos.x];
    }
    PyObject* as_list(bool invert);
    vector<Action> available_moves();
    Turn rollout(); 
    void move(Action p);
    void printBoard();
    Turn get_win_state();
    Turn board[3][3] ;
    Turn turn;
private:
    vector<Action> saved_available_moves;
    bool memo_saved;
    Turn win_state;
};
class Tree;
class Node{
public:

    Node(TicTacToe b,Turn p,Tree* head,Action a);
    void spawn_rollout(float* score_return,bool* has_run);
    float rollout();
    float selection();
    Node* highest_utc();
    void spawn_babies();
    float child_uct(int idx);
    Turn player;
    Action action;
    TicTacToe board;
    std::vector<std::unique_ptr<Node>> children;
    std::atomic<float> value;
    std::atomic<int> visits;
    // std::atomic<int> troll;
    std::atomic<int> virtual_loss;
    Tree* tree;
    std::array<float,9> policy;

private:
    mutable std::mutex node_mutex;
    std::atomic<bool> has_created_children;
    std::atomic<bool> under_shared;
    std::once_flag child_flag;

};


template<typename T>
class SafeVector{
    public:
    SafeVector(){}
    int push_back(T& t){
        std::unique_lock<std::mutex> lock(q_mutex);
        q.push_back(t);
        return q.size()-1;
    }
    void clear(){
        std::unique_lock<std::mutex> lock(q_mutex);
        q.clear();
    }
    std::mutex q_mutex;
    std::vector<T> q;
};
struct PolicyValue{
    std::array<float,9> policy;
    float value;
};
struct ModelConcurrency{
    std::condition_variable cv;
    std::mutex counter_mutex;
    std::mutex vec_mutex;
    std::mutex flag_mutex;
    int counter;
    std::vector<TicTacToe> vec;
    SafeVector<PolicyValue> ret_values;
    bool flag=false;
    int add_board(TicTacToe& t){
        std::unique_lock<std::mutex> lock(vec_mutex);
        vec.push_back(t);
        using namespace std;
        return vec.size()-1;
    }
};
typedef std::tuple<float,std::array<float,9>> (*t_net_outputs)(TicTacToe&,ModelConcurrency*);
class Tree{
public:
    Tree(TicTacToe board, Turn player,t_net_outputs net_func,PyObject* callback);
    void run(int iters,int threads);
    TicTacToe make_play();
    Node head;
    t_net_outputs get_policy_and_value;
    PyObject* callback;
    float virtual_loss_coeff=0;
    ModelConcurrency mc;
    bool done=false;
private:
    void run_thread(int iters);
};

using std::ostream;
template<typename T>
ostream& operator<< (ostream& out, const vector<T>& v) {
    out << "{";
    size_t last = v.size() - 1;
    for(size_t i = 0; i < v.size(); ++i) {
        out << v[i];
        if (i != last) 
            out << ", ";
    }
    out << "}";
    return out;
}
