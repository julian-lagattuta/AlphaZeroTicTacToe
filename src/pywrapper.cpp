#include "TicTacToe.hpp"
#include "Python.h"
#include <iostream>
#include <condition_variable>
#include <tuple>


#include <stdexcept>
#include <exception>
#include <typeinfo>

struct __cxa_exception {
    std::type_info *inf;
};
struct __cxa_eh_globals {
    __cxa_exception *exc;
};
extern "C" __cxa_eh_globals* __cxa_get_globals();
const char* what_exc() {
    __cxa_eh_globals* eh = __cxa_get_globals();
    if (eh && eh->exc && eh->exc->inf)
        return eh->exc->inf->name();
    return NULL;
}
using namespace std;
// SafeVector<TicTacToe> boardVector;
std::tuple<float,std::array<float,9>> agent_callback(TicTacToe& board,std::shared_ptr<ModelConcurrency> mc){
    int idx = mc->add_board(board);    

    
    
    std::unique_lock lk(mc->flag_mutex);
    bool current_flag = mc->flag;
    mc->cv.wait(lk,[current_flag,mc]{return mc->flag!=current_flag;});
    lk.unlock();
    // std::cout<<"add"<<endl;
    //received data
    std::array<float,9> policy = mc->ret_values.q.at(idx).policy;
    float value = mc->ret_values.q.at(idx).value;

    std::unique_lock l(mc->counter_mutex);
    mc->counter++;  
    l.unlock();
    mc->cv.notify_all(); 
    return make_tuple(value,policy); 
}
template <class T, std::size_t N>
ostream& operator<<(ostream& o, const array<T, N>& arr)
{
    copy(arr.cbegin(), arr.cend(), ostream_iterator<T>(o, " "));
    return o;
}


void play_game(int iterations_per_turn,int threads,PyObject* callback,PyObject* policies,PyObject* values,PyObject* boards,std::mutex* list_mutex,std::shared_ptr<ModelConcurrency> mc){

    TicTacToe board = TicTacToe();
    std::vector<int> board_idxs;
    std::vector<bool> inverts;
    using std::cout;
    while(board.get_win_state()==Turn::EMPTY){ 
        Tree t(board,board.turn,agent_callback,callback,mc);
        t.run_dependent(iterations_per_turn,threads,mc);
            // code that could cause exception
        auto move = t.make_play();
        std::unique_lock<std::mutex> policies_lock(*list_mutex);
        cout<<endl;
        PyObject* policy_list = PyList_New(9);
        for(int i = 0;i<9;i++){
            PyList_SetItem(policy_list,i,PyFloat_FromDouble(0));
        }
        for(int i=0;i<t.head.children.size();i++){
            // cout<<t.head.children[i]->value.load()/t.head.children[i]->visits.load()<<" "<<t.head.children[i]->visits.load()<<", ";
            
            PyList_SetItem(policy_list,t.head.children[i]->action,PyFloat_FromDouble(float(t.head.children[i]->visits.load())/float(threads*(iterations_per_turn))));
        }
        cout<<"ref count "<<policy_list->ob_refcnt<<endl;
        PyList_Append(policies,policy_list);

        Py_DECREF(policy_list);
        cout<<"after count "<<policy_list->ob_refcnt<<endl;
        bool inverted = move.turn==Turn::X;
        auto board_as_list = move.as_list(inverted);

        PyList_Append(boards,board_as_list);
        
        PyList_Append(values,PyLong_FromLong(0));

        inverts.push_back(inverted);
        board_idxs.push_back(PyList_Size(policies)-1);
        
        board= move;


        Py_DECREF(board_as_list);
    }
    std::cout<<"acquiring list_mutex"<<endl;
    std::unique_lock<std::mutex> policies_lock(*list_mutex);
    std::cout<<"acquired list_mutex"<<endl;
    auto winner =  board.get_win_state();
    int x_value = winner==Turn::X? 1 : winner==Turn::O ? -1 : 0;
    int o_value = -x_value;
    int i =0;
    for(auto idx:board_idxs){
        auto value = inverts.at(i) ? o_value: x_value;
        PyList_SetItem(values,idx,PyFloat_FromDouble(value));
        i++;
    }
    
}

static PyObject* play_multiple_games(PyObject* self, PyObject* args){
    
    
    Py_ssize_t iterations_per_turn;
    PyObject* callback;
    Py_ssize_t thread_count;
    Py_ssize_t concurrent_games;
    Py_ssize_t rounds_per_game;
    Py_ssize_t total_games;
    if(!PyArg_ParseTuple(args,"nOnnn",&iterations_per_turn,&callback,&thread_count, &concurrent_games,&total_games)){

        return NULL;
    }
    std::mutex list_mutex;
    PyObject* policies= PyList_New(0);
    PyObject* values= PyList_New(0);
    PyObject* boards = PyList_New(0);
    std::atomic<int> game_counter;

    auto model_concurrency = std::make_shared<ModelConcurrency>();

    Py_BEGIN_ALLOW_THREADS;
    auto model_thread = std::thread(&model_thread_func,model_concurrency,callback,1);
    std::vector<std::thread> threads;
    auto play_game_loop =[iterations_per_turn,thread_count,callback,policies,values,boards,&list_mutex,model_concurrency,&game_counter,total_games]{
        for(;game_counter.load()<total_games;game_counter.fetch_add(1)){
            play_game(iterations_per_turn,thread_count,callback,policies,values,boards,&list_mutex,model_concurrency);
        }
    };

    for(int k = 0;k<concurrent_games;k++){
        threads.push_back(std::thread(play_game_loop));
    }
    for(auto& t : threads){
        t.join();
    }
    model_concurrency->done=true;
    model_thread.join();
    Py_END_ALLOW_THREADS; 

    //board, winner
    return Py_BuildValue("OOO",boards,values,policies);


}
static PyObject* play(PyObject* self,PyObject* args){
    using std::cout;
    PyObject* board_list; 
    Py_ssize_t turn;
    Py_ssize_t iterations;
    PyObject* callback;
    Py_ssize_t threads;
    if(!PyArg_ParseTuple(args,"OnnOn",&board_list,&turn,&iterations,&callback,&threads)){

        return NULL;
    }
    TicTacToe board = TicTacToe(board_list,static_cast<Turn>(turn));
    
    Tree t(board,board.turn,agent_callback,callback);
    try
    {
        t.run_independent(iterations,threads);
        // code that could cause exception
    }
    catch (const std::exception &exc)
    {
        cout<<"burrhh"<<endl;
        // catch anything thrown within try block that derives from std::exception
        std::cerr << exc.what();
    }

    auto move = t.make_play();
    cout<<endl;
    PyObject* policy_list = PyList_New(9);
    for(int i = 0;i<9;i++){
        PyList_SetItem(policy_list,i,PyFloat_FromDouble(0));
    }
    for(int i=0;i<t.head.children.size();i++){
        cout<<t.head.children[i]->value.load()/t.head.children[i]->visits.load()<<" "<<t.head.children[i]->visits.load()<<", ";
        
    PyList_SetItem(policy_list,t.head.children[i]->action,PyFloat_FromDouble(float(t.head.children[i]->visits.load())/float(threads*(iterations))));
    }
    cout<<endl;
     
    //board, winner
    return Py_BuildValue("OnO",move.as_list(false),move.get_win_state(),policy_list);

}


static PyMethodDef methods[] = {
    {"play_tic_tac_toe", play, METH_VARARGS, "duhh"},
    {"play_multiple_games",play_multiple_games,METH_VARARGS,"the heck?!"},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef tictactoelib={
    PyModuleDef_HEAD_INIT,
    "tictactoelib",
    "A fast tic tac toe implementation",
    -1,
    methods,

};
PyMODINIT_FUNC PyInit_tictactoelib(void){
    std::set_terminate(__gnu_cxx::__verbose_terminate_handler);
        // std::cout<<"init threads"<<endl;
    // PyImport_ImportModule("threading"),
    return PyModule_Create(&tictactoelib);
}
