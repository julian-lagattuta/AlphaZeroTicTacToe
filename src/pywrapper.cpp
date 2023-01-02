#include "TicTacToe.hpp"
#include "Python.h"
#include <iostream>
#include <condition_variable>
#include <tuple>
using namespace std;
// SafeVector<TicTacToe> boardVector;
std::tuple<float,std::array<float,9>> agent_callback(TicTacToe& board,ModelConcurrency* mc){
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
static PyObject* play(PyObject* self,PyObject* args){
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
    t.run(iterations,threads);
        // code that could cause exception
    }
    catch (const std::exception &exc)
    {
        cout<<"burrhh"<<endl;
        // catch anything thrown within try block that derives from std::exception
        std::cerr << exc.what();
    }

    auto move = t.make_play();
    PyObject* policy_list = PyList_New(9);
    for(int i = 0;i<9;i++){
        PyList_SetItem(policy_list,i,PyFloat_FromDouble(0));
    }
    for(int i=0;i<t.head.children.size();i++){
        cout<<t.head.children[i]->value.load()/t.head.children[i]->visits.load()<<" "<<t.head.children[i]->visits.load()<<", ";
        
        PyList_SetItem(policy_list,t.head.children[i]->action,PyFloat_FromDouble(float(t.head.children[i]->visits.load())/float(threads*(iterations))));
    }
    cout<<endl;
     
    cout<<"policy:"<<t.head.policy<<endl;
    //board, winner
    return Py_BuildValue("OnO",move.as_list(false),move.get_win_state(),policy_list);

}


static PyMethodDef methods[] = {
    {"play_tic_tac_toe", play, METH_VARARGS, "duhh"},
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
