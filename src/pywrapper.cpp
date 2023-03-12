#include "TicTacToe.hpp"
#include "listobject.h"
#include "pyerrors.h"
#include <ctime>
#include <Python.h>
#include <chrono>
#include <functional>
#include <mutex>
#include <type_traits>
#include <iostream>
#include <condition_variable>
#include <tuple>

#include <future>
#include <stdexcept>
#include <exception>
#include <typeinfo>
// struct __cxa_exception {
//     std::type_info *inf;
// };
// struct __cxa_eh_globals {
//     __cxa_exception *exc;
// };
// extern "C" __cxa_eh_globals* __cxa_get_globals();
// const char* what_exc() {
//     __cxa_eh_globals* eh = __cxa_get_globals();
//     if (eh && eh->exc && eh->exc->inf)
//         return eh->exc->inf->name();
//     return NULL;
// }
using namespace std;
// SafeVector<TicTacToe> boardVector;
std::tuple<float,std::array<float,9>> agent_callback(TicTacToe& board,std::shared_ptr<ModelConcurrency> mc){
    int idx = mc->add_board(board);    

    
    
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
/*
void add_to_lists(PyObject* policies, PyObject* values, PyObject* boards, Tree& t,int iterations_per_turn,int threads,TicTacToe& board, vector<int> board_idxs){

    	PyGILState_STATE state;
    	state = PyGILState_Ensure();

        PyObject* policy_list = PyList_New(9);
        for(int i = 0;i<9;i++){
            PyList_SetItem(policy_list,i,PyFloat_FromDouble(0));
        }
        for(int i=0;i<t.head.children.size();i++){
            
            PyList_SetItem(policy_list,t.head.children[i]->action,PyFloat_FromDouble(float(t.head.children[i]->visits.load())/float(threads*(iterations_per_turn))));
        }
        cout<<"ref count "<<policy_list->ob_refcnt<<endl;
        PyList_Append(policies,policy_list);

        Py_DECREF(policy_list);
        cout<<"after count "<<policy_list->ob_refcnt<<endl;
        bool inverted =board.turn==Turn::X;
        auto board_as_list =board.as_list(inverted);

        PyList_Append(boards,board_as_list);
        
        PyList_Append(values,PyLong_FromLong(0));

        inverts.push_back(inverted);
        board_idxs.push_back(PyList_Size(policies)-1);
        
        board= move;


        Py_DECREF(board_as_list);

    	PyGILState_Release(state);
}*/
/*
template<class T>
struct event_utils;

tempmlate<class R, class... Args>
struct event_utils<std::function<R(Args...)>>{
    static R send_to_thread(std::shared_ptr<ModelConcurrency> mc,std::function<R(Args...)> func, Args... arguments){
        function_call_wrapper<decltype(func)>(func,arguments...);
    }
};*/

void event_PyList_Append(PyObject* list,PyObject* o ,std::shared_ptr<ModelConcurrency> mc){
    auto& fw = mc->function_wrappers;
    std::unique_lock<std::mutex> lock(fw.vec_mutex);
    mc->function_wrappers.list_append.push_back(function_call_wrapper<std::function<decltype(PyList_Append)>>(PyList_Append,list,o)) ;
    std::unique_lock<std::mutex> flag_lock(fw.flag_mutex);
    auto saved_flag = fw.flag;
    lock.unlock();
    
    fw.cv.wait(flag_lock,[saved_flag,mc]{return saved_flag!=mc->function_wrappers.flag;});
    flag_lock.unlock();

}
void event_PyList_SetItem(PyObject* list,int index,PyObject* o ,std::shared_ptr<ModelConcurrency> mc){
    auto& fw = mc->function_wrappers;
    std::unique_lock<std::mutex> lock(fw.vec_mutex);
    mc->function_wrappers.list_setitem.push_back(function_call_wrapper<std::function<decltype(PyList_SetItem)>>(PyList_SetItem,list,index,o));
    std::unique_lock<std::mutex> flag_lock(fw.flag_mutex);
    auto saved_flag = fw.flag;
    lock.unlock();
    
    fw.cv.wait(flag_lock,[saved_flag,mc]{return saved_flag!=mc->function_wrappers.flag;});
    flag_lock.unlock();

}
PyObject* event_PyList_New(int i,std::shared_ptr<ModelConcurrency> mc){
    auto& fw = mc->function_wrappers;
    std::unique_lock<std::mutex> lock(fw.vec_mutex);
     
    mc->function_wrappers.list_new.push_back(function_call_wrapper<std::function<decltype(PyList_New)>>(PyList_New,i)) ;
    auto idx = fw.list_new.size()-1;
    std::unique_lock<std::mutex> flag_lock(fw.flag_mutex);
    auto saved_flag = fw.flag;
    lock.unlock();
    
    fw.cv.wait(flag_lock,[saved_flag,mc]{return saved_flag!=mc->function_wrappers.flag;});
    flag_lock.unlock();

    auto ret_value = fw.list_new_ret_values[idx];
    std::unique_lock<std::mutex> cv_lock(fw.cv_mutex);
    fw.counter++;
    cv_lock.unlock(); 
    fw.cv.notify_all(); 
    return ret_value;


}
PyObject* board_to_list(TicTacToe& board,bool invert,std::shared_ptr<ModelConcurrency> mc){
    PyObject* new_list =event_PyList_New(3,mc);
    for(int y =0;y<3;y++){
        auto line = PyList_New(3);
        PyList_SetItem(new_list,y,line);
        for(int x=0;x<3;x++){
            auto p = board.get_idx({x,y});
            if(invert){ 
                if(p==Turn::X){

                    event_PyList_SetItem(line,x,PyLong_FromLong(Turn::O),mc);
                }else if(p==Turn::O){

                    event_PyList_SetItem(line,x,PyLong_FromLong(Turn::X),mc);
                }else{

                    event_PyList_SetItem(line,x,PyLong_FromLong(p),mc);
                }

            }else{

                    event_PyList_SetItem(line,x,PyLong_FromLong(p),mc);
            }
        }
    }
    return new_list;
}
void play_game(int iterations_per_turn,int threads,PyObject* callback,PyObject* policies,PyObject* values,PyObject* boards,std::mutex* list_mutex,std::shared_ptr<ModelConcurrency> mc,bool use_nn){

    TicTacToe board = TicTacToe();
    std::vector<int> board_idxs;
    std::vector<bool> inverts;
    using std::cout;
    while(board.get_win_state()==Turn::EMPTY){ 
        Tree t(board,board.turn,agent_callback,callback,use_nn,mc);
        t.run_dependent(iterations_per_turn,threads,mc);
            // code that could cause exception
        auto move = t.make_play();
        std::unique_lock<std::mutex> policies_lock(*list_mutex);
        PyObject* policy_list =event_PyList_New(9,mc);
        for(int i = 0;i<9;i++){
            event_PyList_SetItem(policy_list,i,PyFloat_FromDouble(0),mc);
        }
        for(int i=0;i<t.head.children.size();i++){
            // cout<<t.head.children[i]->value.load()/t.head.children[i]->visits.load()<<" "<<t.head.children[i]->visits.load()<<", ";
            
            PyList_SetItem(policy_list,t.head.children[i]->action,PyFloat_FromDouble(float(t.head.children[i]->visits.load())/float(threads*(iterations_per_turn)-1)));
        }
        auto timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
        event_PyList_Append(policies,policy_list,mc);
        Py_DECREF(policy_list);
        bool inverted = move.turn==Turn::X;
        PyObject* board_as_list= board_to_list(move,false,mc);
        event_PyList_Append(boards,board_as_list,mc);
        
        event_PyList_Append(values,PyLong_FromLong(0),mc);


        inverts.push_back(inverted);
        board_idxs.push_back(PyList_Size(policies)-1);
        
        board= move;


        Py_DECREF(board_as_list);

    }

    std::unique_lock<std::mutex> policies_lock(*list_mutex);
    auto winner =  board.get_win_state();
    int x_value = winner==Turn::X? 1 : winner==Turn::O ? -1 : 0;
    int o_value = -x_value;
    int i =0;
    for(auto idx:board_idxs){
        auto value = inverts.at(i) ? o_value: x_value;
        event_PyList_SetItem(values,idx,PyFloat_FromDouble(value),mc);
        i++;
    } 
}

static PyObject* play_multiple_games(PyObject* self, PyObject* args){
    
    
    Py_ssize_t iterations_per_turn;
    PyObject* callback;
    Py_ssize_t thread_count;
    Py_ssize_t concurrent_games;
    Py_ssize_t total_games;
    bool use_nn;
    if(!PyArg_ParseTuple(args,"nOnnnp",&iterations_per_turn,&callback,&thread_count, &concurrent_games,&total_games,&use_nn)){


        return NULL;
    }
    std::mutex list_mutex;
    PyObject* policies= PyList_New(0);
    PyObject* values= PyList_New(0);
    PyObject* boards = PyList_New(0);
    std::atomic<int> game_counter;

    auto model_concurrency = std::make_shared<ModelConcurrency>();

    //Py_BEGIN_ALLOW_THREADS;
    std::vector<std::future<void>> futures;
    auto play_game_loop =[iterations_per_turn,thread_count,callback,policies,values,boards,&list_mutex,model_concurrency,&game_counter,total_games,use_nn]{
        for(;game_counter.fetch_add(1)<total_games;){
            play_game(iterations_per_turn,thread_count,callback,policies,values,boards,&list_mutex,model_concurrency,use_nn);
        }
    };

    for(int k = 0;k<concurrent_games;k++){
        futures.push_back(std::async(std::launch::async,play_game_loop));
    }
    bool break_while  = false;

    while(!break_while){
        send_to_model(callback,model_concurrency); 
        send_to_python(model_concurrency); 
        if(use_nn){
            std::this_thread::sleep_for(10ms);
        }else{
            std::this_thread::sleep_for(1ms);
        }

        break_while = true;
        for(auto& f : futures){
            if(f.wait_for(0ms) != std::future_status::ready){
                break_while = false;
                break;
            }
        }
    }
    //Py_END_ALLOW_THREADS; 

    //board, winner
    return Py_BuildValue("OOO",boards,values,policies);


}
/*
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
*/
static PyMethodDef methods[] = {
    {"play_multiple_games",play_multiple_games,METH_VARARGS,"the heck?!"},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef ticmod{
    PyModuleDef_HEAD_INIT,
    "ticlib",
    "A fast tic tac toe implementation",
    -1,
    methods,
    NULL,
    NULL,
    NULL,
    NULL

};

PyMODINIT_FUNC PyInit_tictactoelib(void){
    // std::set_terminate(__gnu_cxx::__verbose_terminate_handler);
        // std::cout<<"init threads"<<endl;
    // PyImport_ImportModule("threading"),
    // cout<<"my bruh"<<endl;
    if (!PyEval_ThreadsInitialized())
    {
        PyEval_InitThreads();
    }

    return PyModule_Create(&ticmod);
}
