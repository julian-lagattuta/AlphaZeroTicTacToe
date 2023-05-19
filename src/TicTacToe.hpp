#pragma once
#include <concepts>
#include <iostream>
#include "Python.h"
#include "abstract.h"
#include "floatobject.h"
#include "listobject.h"
#include "pystate.h"
#include <bits/iterator_concepts.h>
#include <tuple>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>
#include <condition_variable>
#include <array>
#include <atomic>
#include <shared_mutex>
#include <mutex>
#include <type_traits>


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
    NONE=3,
};

class PyGIL{
    public:
    PyGIL(){
        state= PyGILState_Ensure();
    }
    PyGIL(PyGILState_STATE _state){
        state=  _state;
    }
    ~PyGIL(){
        using namespace std;
        if(!moved){
            cout<<"freed"<<endl;
            PyGILState_Release(state);
        }else{
            cout<<"not freed"<<endl;
        }
    }
    PyGIL(PyGIL& g)=delete;
    PyGIL(PyGIL&& gil){
        gil.moved = true;
        state = gil.state;
    }
    private:
        PyGILState_STATE state;
        bool moved = false;
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
    bool safe_done = false;
private:
    mutable std::mutex node_mutex;
    std::atomic<bool> has_created_children;
    std::atomic<bool> under_shared;
    std::once_flag child_flag;

};



template<class T>
class function_call_wrapper;

template<class R, class... Args>
class function_call_wrapper<std::function<R(Args...)>>{
public:

    function_call_wrapper(std::function<R(Args...)> func,Args... args){
        f = func;
        arguments =  std::make_tuple(args...);
    }
    R operator()(){
        return std::apply(f,arguments);
    }

    std::function<R(Args...)> f;
    std::tuple<Args...> arguments;
};
#define f_vector(f) std::vector<function_call_wrapper<std::function<decltype(f)>>>


template< int I, class T >
struct nthtype;
 // base case
template< class Head, class... Tail >
struct nthtype<0, std::tuple<Head, Tail...>> {
   using type = Head;
};
template<int I >
struct nthtype<I, std::tuple<>> {
   using type = void;
};
template< int I, class Head, class... Tail >
struct nthtype<I, std::tuple<Head, Tail...>>
    : nthtype<I-1, std::tuple<Tail...>> { };
template<class T>
struct return_type;

template<class R, class... Args>
struct return_type<std::function<R(Args...)>>{
    using type = R;
};


template<class T>
struct voided_return_type;

template<class R, class... Args>
struct voided_return_type<std::function<R(Args...)>>{
    using type = R;
};
template<class... Args>
struct voided_return_type<std::function<void(Args...)>>{
    using type = void*;
};

template<class T, class B>
struct cat_tuples;

template<class... Args1,class... Args2>
struct cat_tuples<std::tuple<Args1...>,std::tuple<Args2...>>{
    using type = std::tuple<Args1...,Args2...>;
};

template<class A>
struct return_types;

template<>
struct return_types<std::tuple<>>{
    using type = std::tuple<>;
};


template<class Head, class... Tail>
struct return_types<std::tuple<Head,Tail...>>{
    using type = typename std::conditional_t<std::is_same_v<typename return_type<Head>::type, void>,
    typename cat_tuples<
            std::tuple<void*>,
            typename return_types<std::tuple<Tail...>>::type
        >::type,


     typename cat_tuples<
            std::tuple<typename return_type<Head>::type >,
            typename return_types<std::tuple<Tail...>>::type
        >::type
     >;
    
};

template<class... Args>
class WrapperVectors{
public:
    WrapperVectors(){}
    template<int id, class R, class... fArgs, std::enable_if_t<!std::is_void<R>::value,bool> = true>
    R make_call(std::function<R(fArgs...)> func, fArgs... args){
        
        std::unique_lock<std::mutex> lock(vec_mutex);
         
        auto& vec = std::get<id>(tuples);
        vec.push_back(function_call_wrapper<decltype(func)>(func,args...));;
        auto idx = vec.size()-1;
        std::unique_lock<std::mutex> flag_lock(flag_mutex);
        auto saved_flag = flag;
        lock.unlock();
        
        cv.wait(flag_lock,[saved_flag,this]{return saved_flag!=flag;});
        flag_lock.unlock();

        R ret_value = std::get<id>(ret_values)[idx];
        std::unique_lock<std::mutex> cv_lock(cv_mutex);
        counter++;
        cv_lock.unlock(); 
        cv.notify_all(); 
        return ret_value;

    }

    template<int id, class... fArgs>
    void make_call(std::function<void(fArgs...)> func, fArgs... args){
        std::unique_lock<std::mutex> lock(vec_mutex);
         
        auto vec = std::get<id>(tuples);
        vec.push_back(function_call_wrapper<decltype(func)>(func,args...));;
        auto idx = vec.size()-1;
        std::unique_lock<std::mutex> flag_lock(flag_mutex);
        auto saved_flag = flag;
        lock.unlock();
        
        cv.wait(flag_lock,[saved_flag,this]{return saved_flag!=flag;});
        flag_lock.unlock();
        //push_back(function_call_wrapper(func,args...));
    }
    //&& !std::is_void<std::invoke_result< std::tuple_element_t<I,std::tuple<Args...>> > >::value
    
    template<int I=0, std::enable_if_t<I < sizeof...(Args) && !std::is_void_v<typename return_type<typename nthtype<I,std::tuple<Args...>>::type>::type >  ,bool > = true>
    int flush_iter(){

        bool must_release = false;
        if(std::get<I>(tuples).size()>0 && std::get<I>(tuples)[0].f.template target<decltype(PyObject_CallObject)>() == PyObject_CallObject){
            must_release = true;
            PyGILState_Release(state);
        }

        std::get<I>(ret_values).clear();
        for(auto k : std::get<I>(tuples)){
            std::get<I>(ret_values).push_back(k());
        }

        if(must_release){
            state = PyGILState_Ensure();
        }
        int ret_value = std::get<I>(tuples).size() + flush_iter<I+1>();
        std::get<I>(tuples).clear();
        return ret_value;
    }
    template<int I, std::enable_if_t<I == sizeof...(Args), bool> = true>
    int flush_iter(){
        return 0;
    }
    
    template<int I=0, std::enable_if_t<I < sizeof...(Args) && std::is_void_v<typename return_type<typename nthtype<I,std::tuple<Args...>>::type >::type >  ,bool> = true> 
    int flush_iter(){
        std::get<I>(ret_values).clear();
        for(auto k : std::get<I>(tuples)){
                     
            k();
        }
        int ret_value = std::get<I>(tuples).size() + flush_iter<I+1>();
        std::get<I>(tuples).clear();
        return ret_value;
    }
    
    void flush(){
        
        
        
        state = PyGILState_Ensure(); 

        std::unique_lock<std::mutex> vec_lock(vec_mutex);
        std::unique_lock<std::mutex> flag_lock(flag_mutex);


        int batch_size = flush_iter<0>();

        PyGILState_Release(state);

        flag = !flag;


        counter =0;

        cv.notify_all();



        flag_lock.unlock();

        auto temp_counter = &counter;
        std::unique_lock<std::mutex> cv_lock(cv_mutex);
        cv.wait(cv_lock,[temp_counter,batch_size]{return *temp_counter==batch_size;});
        cv_lock.unlock();
    

    }


    PyGILState_STATE state;
    std::atomic<int> counter;
    bool flag = false;
    std::mutex cv_mutex;
    std::mutex flag_mutex;
    std::mutex vec_mutex;
    std::condition_variable cv;
    std::tuple<std::vector<function_call_wrapper<Args>>...> tuples;
    std::tuple<std::vector<typename voided_return_type<Args>::type>...> ret_values;
     
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

#define dtype(a) std::function<decltype(a)>

PyObject* CPPPack(PyObject* a, PyObject* b, PyObject* c,PyObject* d);
void function_Py_DECREF(PyObject* o);
struct ModelConcurrency{
    struct {
        f_vector(PyList_New) list_new;
        f_vector(PyList_Append) list_append;
        f_vector(PyList_SetItem) list_setitem;
        f_vector(PyList_Size) list_size;
        f_vector(function_Py_DECREF) list_decref;
        f_vector(PyLong_FromLong) long_fromlong;
        f_vector(PyFloat_FromDouble) float_fromdouble;
        f_vector(PyLong_AsLong) long_aslong;
        std::vector<PyObject*> list_new_ret_values;
        std::vector<PyObject*> long_fromlong_ret_values;
        std::vector<PyObject*> float_fromdouble_ret_values;
        std::vector<long> long_aslong_ret_values;
        std::vector<long> list_size_ret_values;
        int counter ;
        bool flag=false;
        std::mutex flag_mutex;
        std::mutex vec_mutex;
        std::condition_variable cv;
        std::mutex cv_mutex;
        
    } function_wrappers;


    WrapperVectors<dtype(PyList_New),dtype(PyList_Append),dtype(PyList_SetItem),dtype(PyList_Size),dtype(function_Py_DECREF),dtype(PyLong_FromLong),dtype(PyFloat_FromDouble),dtype(PyLong_AsLong),dtype(PyObject_CallObject),dtype(CPPPack),dtype(PyFloat_AsDouble)> fwrappers;
    std::atomic<int> winner_tally;
    std::atomic<int> tie_tally;

    std::condition_variable cv;
    std::mutex counter_mutex;
    std::mutex vec_mutex;
    std::mutex flag_mutex;
    int counter;
    std::vector<TicTacToe> vec;
    std::vector<int> model_ids;
    std::vector<PyObject*> models;
    SafeVector<PolicyValue> ret_values;
    std::vector<float> alphas;
    bool flag=false;
    bool done= false;
    int add_board(TicTacToe& t,int model,float alpha){
        std::unique_lock<std::mutex> lock(vec_mutex);
        vec.push_back(t);
        model_ids.push_back(model);
        alphas.push_back(alpha);
	int idx = vec.size()-1;
        using namespace std;
    	std::unique_lock lk(flag_mutex);
    	bool current_flag = flag;
	lock.unlock();
	
    	cv.wait(lk,[current_flag,this]{return this->flag!=current_flag;});
	lk.unlock();
        return idx;
    }
};
typedef std::tuple<float,std::array<float,9>> (*t_net_outputs)(TicTacToe&,std::shared_ptr<ModelConcurrency>,int,float);
class Tree{
public:
    Tree(TicTacToe b,Turn p,t_net_outputs net_func, PyObject* _callback,std::shared_ptr<ModelConcurrency> model_concurrency,int _model_id,int is_training);

    Tree(TicTacToe b,Turn p,std::shared_ptr<ModelConcurrency> model_concurrency,int is_training);
    void run_dependent(int iters,int threads,std::shared_ptr<ModelConcurrency> mc);
    void run_independent(int iters,int threads);
    void run_thread(int i,std::atomic<int>* iter_count);
    TicTacToe make_play();
    Node head;
    t_net_outputs get_policy_and_value;
    PyObject* callback;
    float virtual_loss_coeff=.1;
    std::shared_ptr<ModelConcurrency> mc;
    bool done=false;
    bool use_nn = false;
    int is_training = 1;
    int model_id;
    float exploration_constant = 1.414;
private:
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


void send_to_model(PyObject* agent_function,std::shared_ptr<ModelConcurrency> mc);
bool send_to_python(std::shared_ptr<ModelConcurrency> mc);


Turn opposite_turn(Turn t);

void f_Py_DECREF(PyObject* o,std::shared_ptr<ModelConcurrency> mc);
PyObject* board_to_list(TicTacToe& board,bool invert,std::shared_ptr<ModelConcurrency> mc);
PyObject* event_PyObject_CallObject(PyObject* o,PyObject* args ,std::shared_ptr<ModelConcurrency> mc);
long event_PyLong_AsLong(PyObject* l,std::shared_ptr<ModelConcurrency> mc);
PyObject* event_PyFloat_FromDouble(double  l,std::shared_ptr<ModelConcurrency> mc);
PyObject* event_PyLong_FromLong(long l,std::shared_ptr<ModelConcurrency> mc);
int event_PyList_Size(PyObject* list,std::shared_ptr<ModelConcurrency> mc);
void event_PyList_Append(PyObject* list,PyObject* o ,std::shared_ptr<ModelConcurrency> mc);
PyObject* event_PyList_New(long i,std::shared_ptr<ModelConcurrency> mc);
double event_PyFloat_AsDouble(PyObject* i,std::shared_ptr<ModelConcurrency> mc);
PyObject* CPPPack(PyObject* a, PyObject* b, PyObject* c,PyObject* d);
PyObject* event_PyTuple_Pack(std::shared_ptr<ModelConcurrency> mc, PyObject* a, PyObject* b, PyObject* c, PyObject* d);
void event_PyList_SetItem(PyObject* list,long index,PyObject* o ,std::shared_ptr<ModelConcurrency> mc);
