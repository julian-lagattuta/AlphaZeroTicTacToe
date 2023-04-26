#include "TicTacToe.hpp"
#include "ceval.h"
#include "floatobject.h"
#include "listobject.h"
#include "longobject.h"
#include "modsupport.h"
#include "object.h"
#include "pyerrors.h"
#include "pystate.h"
#include <ctime>
#include <Python.h>
#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
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
std::tuple<float,std::array<float,9>> agent_callback(TicTacToe& board,std::shared_ptr<ModelConcurrency> mc,int model_id){
    int idx = mc->add_board(board,model_id);    

    
    
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

long event_PyLong_AsLong(PyObject* l,std::shared_ptr<ModelConcurrency> mc){
    
    

    auto& fw = mc->function_wrappers;
    std::unique_lock<std::mutex> lock(fw.vec_mutex);
     
    mc->function_wrappers.long_aslong.push_back(function_call_wrapper<std::function<decltype(PyLong_AsLong)>>(PyLong_AsLong,l)) ;
    auto idx = fw.long_aslong.size()-1;
    std::unique_lock<std::mutex> flag_lock(fw.flag_mutex);
    auto saved_flag = fw.flag;
    lock.unlock();
    
    fw.cv.wait(flag_lock,[saved_flag,mc]{return saved_flag!=mc->function_wrappers.flag;});
    flag_lock.unlock();

    auto ret_value = fw.long_aslong_ret_values[idx];
    std::unique_lock<std::mutex> cv_lock(fw.cv_mutex);
    fw.counter++;
    cv_lock.unlock(); 
    fw.cv.notify_all(); 
    return ret_value;


}
PyObject* event_PyFloat_FromDouble(double l,std::shared_ptr<ModelConcurrency> mc){
    auto& fw = mc->function_wrappers;
    std::unique_lock<std::mutex> lock(fw.vec_mutex);
     
    mc->function_wrappers.float_fromdouble.push_back(function_call_wrapper<std::function<decltype(PyFloat_FromDouble)>>(PyFloat_FromDouble,l)) ;
    auto idx = fw.float_fromdouble.size()-1;
    std::unique_lock<std::mutex> flag_lock(fw.flag_mutex);
    auto saved_flag = fw.flag;
    lock.unlock();
    
    fw.cv.wait(flag_lock,[saved_flag,mc]{return saved_flag!=mc->function_wrappers.flag;});
    flag_lock.unlock();

    auto ret_value = fw.float_fromdouble_ret_values[idx];
    std::unique_lock<std::mutex> cv_lock(fw.cv_mutex);
    fw.counter++;
    cv_lock.unlock(); 
    fw.cv.notify_all(); 
    return ret_value;


}
PyObject* event_PyLong_FromLong(long l,std::shared_ptr<ModelConcurrency> mc){


    auto& fw = mc->function_wrappers;
    std::unique_lock<std::mutex> lock(fw.vec_mutex);
     
    mc->function_wrappers.long_fromlong.push_back(function_call_wrapper<std::function<decltype(PyLong_FromLong)>>(PyLong_FromLong,l)) ;
    auto idx = fw.long_fromlong.size()-1;
    std::unique_lock<std::mutex> flag_lock(fw.flag_mutex);
    auto saved_flag = fw.flag;
    lock.unlock();
    
    fw.cv.wait(flag_lock,[saved_flag,mc]{return saved_flag!=mc->function_wrappers.flag;});
    flag_lock.unlock();

    auto ret_value = fw.long_fromlong_ret_values[idx];
    std::unique_lock<std::mutex> cv_lock(fw.cv_mutex);
    fw.counter++;
    cv_lock.unlock(); 
    fw.cv.notify_all(); 
    return ret_value;


}
int event_PyList_Size(PyObject* list,std::shared_ptr<ModelConcurrency> mc){

    auto& fw = mc->function_wrappers;
    std::unique_lock<std::mutex> lock(fw.vec_mutex);
     
    mc->function_wrappers.list_size.push_back(function_call_wrapper<std::function<decltype(PyList_Size)>>(PyList_Size,list)) ;
    auto idx = fw.list_size.size()-1;
    std::unique_lock<std::mutex> flag_lock(fw.flag_mutex);
    auto saved_flag = fw.flag;
    lock.unlock();
    
    fw.cv.wait(flag_lock,[saved_flag,mc]{return saved_flag!=mc->function_wrappers.flag;});
    flag_lock.unlock();

    auto ret_value = fw.list_size_ret_values[idx];
    std::unique_lock<std::mutex> cv_lock(fw.cv_mutex);
    fw.counter++;
    cv_lock.unlock(); 
    fw.cv.notify_all(); 
    return ret_value;


}
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
TicTacToe list_to_board(PyObject* board_list){

    TicTacToe t;
    
    for(auto y =0;y<3;y++){
        PyObject* row =PyList_GetItem(board_list,y);
        for(auto x= 0;x<3;x++){
                t.set_idx({x,y}, static_cast<Turn>(PyLong_AsLong(PyList_GetItem(row,x))));
        }
    }
    return t;
}
TicTacToe list_to_board(PyObject* board_list,std::shared_ptr<ModelConcurrency> mc){

    TicTacToe t;
    
    for(auto y =0;y<3;y++){
        PyObject* row =PyList_GetItem(board_list,y);
        for(auto x= 0;x<3;x++){
                t.set_idx({x,y}, static_cast<Turn>(event_PyLong_AsLong(PyList_GetItem(row,x),mc)));
        }
    }
    return t;
}
PyObject* board_to_list(TicTacToe& board,bool invert,std::shared_ptr<ModelConcurrency> mc){
    PyObject* new_list =event_PyList_New(3,mc);
    for(int y =0;y<3;y++){
        auto line = event_PyList_New(3,mc);
        PyList_SetItem(new_list,y,line);
        for(int x=0;x<3;x++){
            auto p = board.get_idx({x,y});
            if(invert){ 
                if(p==Turn::X){

                    event_PyList_SetItem(line,x,event_PyLong_FromLong(Turn::O,mc),mc);
                }else if(p==Turn::O){

                    event_PyList_SetItem(line,x,event_PyLong_FromLong(Turn::X,mc),mc);
                }else{

                    event_PyList_SetItem(line,x,event_PyLong_FromLong(p,mc),mc);
                }

            }else{

                    event_PyList_SetItem(line,x,event_PyLong_FromLong(p,mc),mc);
            }
        }
    }
    return new_list;
}
void play_game(bool one_turn, int iterations_per_turn,int threads,bool return_last_move,PyObject* callback,PyObject* policies,PyObject* values,PyObject* boards,PyObject* turns,PyObject* is_terminals,std::mutex* list_mutex,std::shared_ptr<ModelConcurrency> mc,bool use_nn,PyObject* starting_position,int starting_turn){
    TicTacToe board = TicTacToe();



    if(one_turn){
        board = list_to_board(starting_position,mc); 
        board.turn = static_cast<Turn>(starting_turn);
    }
    std::vector<int> board_idxs;
    std::vector<bool> inverts;
    std::unique_lock<std::mutex> policies_lock(*list_mutex);

    bool inverted = Turn::X==board.turn;
    PyObject* board_list = board_to_list(board, inverted, mc);
    event_PyList_Append(boards,board_list,mc);
    
    f_Py_DECREF(board_list,mc);
    auto py_turn = event_PyLong_FromLong(static_cast<int>(opposite_turn(board.turn)),mc);
    inverts.push_back(inverted);
    event_PyList_Append(turns,py_turn,mc);
    
    f_Py_DECREF(py_turn,mc);
    
    auto py_value = event_PyLong_FromLong(0,mc);
    event_PyList_Append(values,py_value,mc);

    f_Py_DECREF(py_value,mc);
    
    auto py_zero = event_PyLong_FromLong(0,mc);
    event_PyList_Append(is_terminals,py_zero,mc);

    f_Py_DECREF(py_zero,mc);


    board_idxs.push_back(event_PyList_Size(values,mc)-1);
    

    policies_lock.unlock();
    
    int model_id=std::chrono::system_clock::now().time_since_epoch().count()%2;
    int x_model_id = model_id;
    cout<<"x model_id: "<<x_model_id<<endl;
    using std::cout;
    do{
        Tree t(board,board.turn,agent_callback,callback,mc,model_id);
        t.use_nn = use_nn;
        t.run_dependent(iterations_per_turn,threads,mc);

        cout<<"values: ";
        for(auto& child: t.head.children){
            cout<<child->value<<" ";
        }
        cout<<endl;
            // code that could cause exception
        auto move = t.make_play();

        std::unique_lock<std::mutex> _policies_lock(*list_mutex);

        PyObject* policy_list =event_PyList_New(9,mc);
        for(int i = 0;i<9;i++){
            event_PyList_SetItem(policy_list,i,event_PyFloat_FromDouble(0,mc),mc);
        }
        int sum= 0;
        for(int i=0;i<t.head.children.size();i++){
            // cout<<t.head.children[i]->value.load()/t.head.children[i]->visits.load()<<" "<<t.head.children[i]->visits.load()<<", ";
            sum+=t.head.children[i]->visits.load();
            PyList_SetItem(policy_list,t.head.children[i]->action,event_PyFloat_FromDouble(float(t.head.children[i]->visits.load())/float(iterations_per_turn-1),mc));
        }
        cout<<"total iterations: "<<sum<<endl;

        event_PyList_Append(policies,policy_list,mc);
        f_Py_DECREF(policy_list,mc);

        if(return_last_move ||  move.get_win_state()==Turn::EMPTY){
            bool inverted = move.turn==Turn::X;
            PyObject* board_as_list= board_to_list(move,false,mc);
            
            event_PyList_Append(boards,board_as_list,mc);
            f_Py_DECREF(board_as_list,mc);
            
            py_turn = event_PyLong_FromLong(static_cast<int>(board.turn),mc);
            event_PyList_Append(turns,py_turn,mc);
    
            f_Py_DECREF(py_turn,mc);

            py_zero  = event_PyLong_FromLong(0,mc);
            event_PyList_Append(values,py_zero,mc);

            event_PyList_Append(is_terminals,event_PyLong_FromLong(0,mc),mc);

            f_Py_DECREF(py_zero,mc);

            inverts.push_back(inverted);
            board_idxs.push_back(event_PyList_Size(values,mc)-1);


        }
        board= move;

        model_id++;
        model_id%=2;

    }while(board.get_win_state()==Turn::EMPTY && !one_turn);
   policies_lock.lock(); 
    auto winner =  board.get_win_state();
    if(winner!=Turn::TIE){
        auto inc_amount = winner==Turn::X? 1 : winner==Turn::O ? -1 : 0;
        inc_amount*=x_model_id==0? 1 : -1;
        mc->winner_tally+=inc_amount; 
    }else{
        mc->tie_tally++;
    }
    int x_value = winner==Turn::X? 1 : winner==Turn::O ? -1 : 0;
    int o_value = -x_value;
    int i =0;
    for(auto idx:board_idxs){
        auto value = inverts.at(i) ? o_value: x_value;
        event_PyList_SetItem(values,idx,event_PyFloat_FromDouble(value,mc),mc);
        i++;
    }
    event_PyList_SetItem(is_terminals,board_idxs.at(board_idxs.size()-1),event_PyLong_FromLong(winner,mc),mc);
    if(board_idxs.size()>1 && return_last_move)
        event_PyList_SetItem(is_terminals,board_idxs[board_idxs.size()-2],event_PyLong_FromLong(1,mc),mc);

}
void test_func(){
    auto a= PyGILState_Ensure(); 
    PyGILState_Release(a);
}
static PyObject* play_multiple_games(PyObject* self, PyObject* args){
    
    PyObject* arg_model1; 
    PyObject* arg_model2; 
    Py_ssize_t iterations_per_turn;
    PyObject* callback;
    PyObject* starting_position;
    Py_ssize_t thread_count;
    Py_ssize_t concurrent_games;
    Py_ssize_t total_games;
    int return_last_move;
    int use_nn_;
    int one_turn;
    int starting_turn;
    if(!PyArg_ParseTuple(args,"OOnOnnnpppOl",&arg_model1,&arg_model2,&iterations_per_turn,&callback,&thread_count, &concurrent_games,&total_games,&use_nn_,&return_last_move,&one_turn,&starting_position,&starting_turn)){


        return NULL;
    }
    PyObject* model1=  arg_model1;
    PyObject* model2 = arg_model2==Py_None ? model1 : arg_model2;
    cout<<"using use_nn "<<use_nn_<<endl;
    if(!use_nn_&& (model1!=Py_None || model2!=Py_None)){
         cout<<"WARNING: inputted model when use_nn is false"<<endl;
    }else if(use_nn_&& model1==Py_None){
        cout<<"ERROR: passed no model when use_nn is true"<<endl;
    }
    std::mutex list_mutex;
    PyObject* policies= PyList_New(0);
    PyObject* values= PyList_New(0);
    PyObject* boards = PyList_New(0);
    PyObject* turns = PyList_New(0);
    PyObject* is_terminals = PyList_New(0);
    PyObject* model_ids= PyList_New(0);

    std::atomic<int> game_counter;

    auto model_concurrency = std::make_shared<ModelConcurrency>();
    model_concurrency->models.push_back(model1);
    model_concurrency->models.push_back(model2);
    
    std::vector<std::future<void>> futures;
    auto play_game_loop =[iterations_per_turn,thread_count,callback,policies,values,boards,&list_mutex,model_concurrency,&game_counter,total_games,use_nn_,turns,return_last_move,is_terminals,one_turn,starting_position,starting_turn,model_ids](int model_id){
        for(;game_counter.fetch_add(1)<total_games;){
            play_game(one_turn,iterations_per_turn,thread_count,return_last_move,callback,policies,values,boards,turns,is_terminals,&list_mutex,model_concurrency,use_nn_,starting_position,starting_turn);
        }
    };

    for(int k = 0;k<concurrent_games;k++){
        futures.push_back(std::async(std::launch::async,play_game_loop,k%2));
    }
    bool break_while  = false;
    int futures_left = futures.size();
    while(!break_while){
        send_to_model(callback,model_concurrency);
        while(send_to_python(model_concurrency)){
            int delay_timer = 50;
            while(send_to_python(model_concurrency));
            while(--delay_timer);
            cout<<"b";
        }
        if(use_nn_){
            if(futures_left==1){
                std::this_thread::sleep_for(1ms);
            }else{
                std::this_thread::sleep_for(1ms*futures_left);
            }
        }else{
            std::this_thread::sleep_for(0ms);
        }
        
        futures_left = futures.size();
        break_while = true;
        for(auto& f : futures){
            if(f.wait_for(0ms) != std::future_status::ready){
                break_while = false;
                futures_left--;
            }
        }
    }

    //board, winner
    PyObject* return_value = Py_BuildValue("OOOOOll",boards,values,policies,turns,is_terminals,model_concurrency->winner_tally.load(),model_concurrency->tie_tally.load());
    Py_DECREF(boards);
    Py_DECREF(values);
    Py_DECREF(policies);
    Py_DECREF(turns);
    Py_DECREF(is_terminals);
    cout<<"boards ref count: "<<boards->ob_refcnt<<endl; 
    return return_value;

}


static PyObject* PyBoard_win_state(PyObject* self, PyObject* args){
    PyObject* board_list;
    if(!PyArg_ParseTuple(args,"O",&board_list)){
        return NULL;
    }

    auto size = PyList_Size(board_list); 
    if(size!=3){
        cout<<"ok please input ONE tic tac toe board"<<endl;
        return NULL;
    }
    auto t = list_to_board(board_list);
    auto winner = t.get_win_state();
    return Py_BuildValue("l",winner);
}
static PyMethodDef methods[] = {
    {"play_multiple_games",play_multiple_games,METH_VARARGS,"the heck?!"},
    {"get_board_win_state",PyBoard_win_state,METH_VARARGS,"the heck?!"},
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
