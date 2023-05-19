#include <iterator>
#include <random>
#include "TicTacToe.hpp" 
#include "floatobject.h"
#include "listobject.h"
#include "longobject.h"
#include "object.h"
#include "pystate.h"
#include <chrono>
#include <stdexcept>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <array>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
using namespace std;
void function_Py_DECREF(PyObject* o){
    Py_DECREF(o);
}

void f_Py_DECREF(PyObject* o ,std::shared_ptr<ModelConcurrency> mc){

    mc->fwrappers.make_call<4>(std::function(function_Py_DECREF),o);
    
}
PyObject* event_PyObject_CallObject(PyObject* o,PyObject* args ,std::shared_ptr<ModelConcurrency> mc){
    return mc->fwrappers.make_call<8>(std::function(PyObject_CallObject),o,args);
}
long event_PyLong_AsLong(PyObject* l,std::shared_ptr<ModelConcurrency> mc){
    return mc->fwrappers.make_call<7>(std::function(PyLong_AsLong),l);
}
PyObject* event_PyFloat_FromDouble(double  l,std::shared_ptr<ModelConcurrency> mc){
    return mc->fwrappers.make_call<6>(std::function(PyFloat_FromDouble),l);
}
PyObject* event_PyLong_FromLong(long l,std::shared_ptr<ModelConcurrency> mc){
    return mc->fwrappers.make_call<5>(std::function(PyLong_FromLong),l);
}
int event_PyList_Size(PyObject* list,std::shared_ptr<ModelConcurrency> mc){
    return mc->fwrappers.make_call<3>(std::function(PyList_Size),list);
}
void event_PyList_Append(PyObject* list,PyObject* o ,std::shared_ptr<ModelConcurrency> mc){
    mc->fwrappers.make_call<1>(std::function(PyList_Append),list,o);
}
PyObject* event_PyList_New(long i,std::shared_ptr<ModelConcurrency> mc){
    return mc->fwrappers.make_call<0>(std::function(PyList_New),i);
}
double event_PyFloat_AsDouble(PyObject* i,std::shared_ptr<ModelConcurrency> mc){
    return mc->fwrappers.make_call<10>(std::function(PyFloat_AsDouble),i);
}
PyObject* CPPPack(PyObject* a, PyObject* b, PyObject* c,PyObject* d){
    return PyTuple_Pack(4,a,b,c,d);
}
PyObject* event_PyTuple_Pack( PyObject* a, PyObject* b, PyObject* c, PyObject* d,std::shared_ptr<ModelConcurrency> mc){
    return mc->fwrappers.make_call<9>(std::function(CPPPack),a,b,c,d);
}
void event_PyList_SetItem(PyObject* list,long index,PyObject* o ,std::shared_ptr<ModelConcurrency> mc){
    mc->fwrappers.make_call<2>(std::function(PyList_SetItem),list,index,o);
    
}
Turn opposite_turn(Turn t){
    if(t==Turn::X){
        return Turn::O;
    }if(t==Turn::O){
        return Turn::X;
    }
    std::cout<<"ERROR UNKOWN TURN1!!"<<std::endl;
    return EMPTY;
}
TicTacToe::TicTacToe(){
    for(int i = 0;i<3;i++){
        for(int k =0;k<3;k++){
           set_idx({i,k},Turn::EMPTY);
        }
    } 
    turn = Turn::X;
    win_state=Turn::NONE;
    memo_saved = false;

}

TicTacToe::TicTacToe(PyObject* o,Turn t){
    

    for(int i =0;i<3;i++){
        for(int j=0;j<3;j++){
            set_idx({i,j},static_cast<Turn>(PyLong_AsLong(PyList_GetItem(PyList_GetItem(o,j),i))));
        }
    }
    memo_saved = false;
    win_state = Turn::NONE;
    turn =t;
}
vector<Action> TicTacToe::available_moves(){
    if(memo_saved){
        return saved_available_moves;
    }
    vector<Action> pos;
    for(int i =0;i<3;i++){
        for(int j = 0;j<3;j++){
            if(get_idx({j,i})==Turn::EMPTY)
                pos.push_back(i*3+j);
        }
    }
    saved_available_moves = pos;
    memo_saved = true;
    return pos;
}

Turn TicTacToe::get_win_state(){
    if(win_state!=Turn::NONE){
        return win_state;
    }
    for(int i =0;i<3;i++){
        if(get_idx({0,i}) != Turn::EMPTY && get_idx({0,i})==get_idx({1,i}) && get_idx({1,i})==get_idx({2,i})){
            auto k =get_idx({0,i});
            win_state = k;
            return k; 
        }
    }
    for(int i =0;i<3;i++){
        if(get_idx({i,0}) != Turn::EMPTY && get_idx({i,0})==get_idx({i,1}) && get_idx({i,1})==get_idx({i,2})){
            auto k =get_idx({i,0});
            win_state =k;
            return k;
        }
    }
    if(get_idx({1,1})!=Turn::EMPTY&& get_idx({0,0})==get_idx({1,1}) && get_idx({1,1})==get_idx({2,2})){
        auto k = get_idx({0,0});
        win_state = k;
        return k;
    }if(get_idx({1,1})!=Turn::EMPTY&& get_idx({0,2})==get_idx({1,1}) && get_idx({1,1})==get_idx({2,0})){
        auto k = get_idx({1,1});
        win_state = k;
        return k;
    }
    for(int i = 0;i<3;i++){
        for(int j=0;j<3;j++){
            if(get_idx({i,j})==Turn::EMPTY){
                auto k = get_idx({i,j});
                win_state = k;
                return k;
            }
        }
    }
    win_state = Turn::TIE;
    return Turn::TIE;
}


void TicTacToe::move(Action p){
    if(get_idx({p%3,p/3})!=Turn::EMPTY){
        throw std::runtime_error("Tried to write to illegal position");
    }
    set_idx({p%3,p/3},turn);
    turn = turn==Turn::X ? Turn::O : Turn::X;
    win_state = Turn::NONE;
    memo_saved = false;
}


using namespace std;
void TicTacToe::printBoard(){
    for(int i =0;i<3;i++){
        for(int j = 0;j<3;j++){
            const char* a[] = {"-","X","O"};
            cout<<a[get_idx({j,i})]<<'\t';
        }
        cout<<endl;
    }
}

Turn TicTacToe::rollout(){
    
    auto copy = *this;
    while(copy.get_win_state()==Turn::EMPTY){
        auto av = copy.available_moves();
        
        copy.move(av[rand()%av.size()]);
    }
    
    return copy.get_win_state();
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
PyObject* TicTacToe::as_list(bool invert){
    auto state =PyGILState_Ensure();
    PyObject* new_list = PyList_New(3);
    for(int y =0;y<3;y++){
        auto line = PyList_New(3);
        PyList_SetItem(new_list,y,line);
        for(int x=0;x<3;x++){
            auto p = get_idx({x,y});
            if(invert){ 
                if(p==Turn::X){

                    PyList_SetItem(line,x,PyLong_FromLong(Turn::O));
                }else if(p==Turn::O){

                    PyList_SetItem(line,x,PyLong_FromLong(Turn::X));
                }else{

                    PyList_SetItem(line,x,PyLong_FromLong(p));
                }

            }else{

                    PyList_SetItem(line,x,PyLong_FromLong(p));
            }
        }
    }
    return new_list;
}

Node::Node(TicTacToe b,Turn p,Tree* h,Action a):action(a),tree(h),visits(0){
    player = p;
    board = b;
    has_created_children.store(false);
    under_shared.store(false);
    // spawning_children=false;
}
float Node::rollout(){
    if(!tree->use_nn ){
        Turn state = board.rollout();
        if(state==player)
            return 1;
        if(state==Turn::TIE)
            return 0;
        return -1;
    }
    if(board.get_win_state()!=Turn::EMPTY){
        if(board.get_win_state()==player) return 1;
        if(board.get_win_state()==Turn::TIE) return 0;
        return -1;
    }
    float alpha = 0;
    if(tree->is_training){
        alpha = &tree->head==this? 1: .01; 
    }else{
        alpha = 0;
    }
    auto value_policy  = tree->get_policy_and_value(board,tree->mc,tree->model_id,alpha);
    
    float value = std::get<0>(value_policy);
    if(player==board.turn){
        value*=-1;
    }
    
    auto ret_policy = std::get<1>(value_policy);
    std::array<float,9> net_policy;
    
    auto av = board.available_moves();
    float sum =0;
    for(int i = 0;i<9;i++){
        net_policy[i]=0;
    }
    for(int i = 0;i<av.size();i++){
        net_policy[av[i]]=ret_policy[av[i]]; 
        sum+=ret_policy[av[i]];
    }
#ifdef DEBUG
    cout<<"value: "<<value<<"\npolicy: ";
#endif
    for(int i =0;i<net_policy.size();i++){
        net_policy[i]/=sum;
#ifdef DEBUG
        cout<<net_policy[i]<<" ";
#endif
    }
#ifdef DEBUG
    cout<<endl;
    board.printBoard();
    cout<<endl;
#endif
    policy = net_policy;
    return value;

    
}
float Node::child_uct(int idx){
    float mult = board.turn!=player? -1:1;
    auto& child = children[idx];
    float cv = child->visits.load();
    if(cv==0){
        cv=1e-4;
    }

    float v = child->value.load()*mult;
    float sv = visits.load(); 
    float uct = v/cv+sqrt(2*log(sv)/cv);
    return  uct;
}
Node* Node::highest_utc(){

    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    srand(seed);
    float max_uct = -INFINITY;
    auto highest_node = children[0].get();
    float sv = visits.load();
    assert(sv!=0);
    
    float mult = board.turn!=player? -1:1;
    int idx =0 ;
    int chosen = 0;
    std::vector<Node*> unvisited;
    bool found_unvisited = false;
    for(auto& child: children){
        if(child->visits.load()==0&&!tree->use_nn ){
            unvisited.push_back(child.get());
            found_unvisited=true;
            continue;
        }
        if(found_unvisited){
            continue;
        }

        float cv = child->visits.load();
        if(cv==0){
            cv=1e-4;
        }

        float v = child->value.load()*mult;
        float uct = 0;

        auto virtual_loss_value= child->virtual_loss.load()*tree->virtual_loss_coeff;
        if(tree->use_nn){
            uct = v/cv+tree->exploration_constant*policy[child->action]*sqrt(visits.load())/(1+child->visits)-virtual_loss_value;
        }else{
            
            uct = v/cv+tree->exploration_constant*sqrt(log(sv)/cv)-child->virtual_loss.load()*tree->virtual_loss_coeff;
        }
        if(uct>max_uct ){
            max_uct=uct;
            highest_node = child.get();
            chosen = idx;
        }
        idx++;
    }
    if(found_unvisited&& !tree->use_nn){
        return unvisited.at(rand()%(unvisited.size()));
    }
    return highest_node;
}
void Node::spawn_babies(){
    auto av=  board.available_moves();
    for(Action p : av){
        TicTacToe b = board;
        b.move(p);
        auto n = std::make_unique<Node>(b,player,tree,p);
        children.push_back(std::move(n));
    }
}
void Node::spawn_rollout(float* score_return,bool* has_run){
    
    *has_run = true;
    auto score = rollout();
    spawn_babies();   
    value.fetch_add(score);

    // virtual_loss--;
    // spawning_children=false;
    // spawning_children.store(false);
    // spawning_children.notify_all();
    has_created_children.store(true);
    *score_return = score;
}
float Node::selection(){
    virtual_loss+=tree->virtual_loss_coeff;
    auto winner = board.get_win_state();
    if(winner!=Turn::EMPTY){
        float score = rollout();
        value.fetch_add(score);
        virtual_loss-= tree->virtual_loss_coeff;
        visits++;
        return score;
    }
    
    if(!safe_done){
        std::unique_lock lock(node_mutex);
        float ret_score=0;
        bool has_run = false;
        std::call_once(child_flag,&Node::spawn_rollout,*this,&ret_score,&has_run);
        if(has_run){
            virtual_loss-=tree->virtual_loss_coeff;
            
            visits++;
            safe_done=true;
            return ret_score;
        }
    }
    assert(children.size()!=0); 
    // std::shared_lock slock(node_mutex);
    // under_shared.store(true);
    assert(children.size()!=0); 
    Node* highest = highest_utc();
    // Node* highest=  children[0].get();
    auto score = highest->selection();

    value.fetch_add(score);
    virtual_loss-=tree->virtual_loss_coeff;
    visits++;
    under_shared.store(false);
    return score;
}
Tree::Tree(TicTacToe b,Turn p,std::shared_ptr<ModelConcurrency> model_concurrency,int _is_training): head(b,p,this,0), use_nn(false), is_training(_is_training) {
    if(model_concurrency.get()==nullptr){
        mc = make_shared<ModelConcurrency>();
    }else{
        mc = model_concurrency;
    }
}
Tree::Tree(TicTacToe b,Turn p,t_net_outputs net_func, PyObject* _callback,std::shared_ptr<ModelConcurrency> model_concurrency,int _model_id, int _is_training): head(b,p,this,0), get_policy_and_value(net_func), callback(_callback), use_nn(true), is_training(_is_training) {

    if(model_concurrency.get()==nullptr){

        mc = make_shared<ModelConcurrency>();
    }else{
        mc = model_concurrency;
    }
    model_id= _model_id;
}
void Tree::run_thread(int i,std::atomic<int>* iter_count){
    srand(time(NULL));
    for(;iter_count->fetch_add(1)<i;){
        head.selection();
    }
}
bool send_to_python(std::shared_ptr<ModelConcurrency> mc){

    PyGILState_STATE state = PyGILState_Ensure(); 
    auto& fw = mc->function_wrappers;

    std::unique_lock<std::mutex> vec_lock(fw.vec_mutex);
    std::unique_lock<std::mutex> flag_lock(fw.flag_mutex);



    fw.list_new_ret_values.clear();
    fw.long_fromlong_ret_values.clear();
    fw.long_aslong_ret_values.clear();
    fw.float_fromdouble_ret_values.clear();
    fw.list_size_ret_values.clear();

    if(fw.list_append.size()+fw.list_size.size()+fw.list_setitem.size()+fw.list_new.size()+fw.list_decref.size()+fw.long_fromlong.size()+fw.long_aslong.size()+fw.float_fromdouble.size()==0){
        PyGILState_Release(state);
        return false;
    }

    for(auto& f : fw.list_append){
        f();
    }
    for(auto& f: fw.list_decref){
        f();
    }
    for(auto& f: fw.list_new){
        auto pyobj = f();
        fw.list_new_ret_values.push_back(pyobj); 
    }

    for(auto& f: fw.float_fromdouble){
        auto pyobj = f();
        fw.float_fromdouble_ret_values.push_back(pyobj); 
    }
    for(auto& f: fw.long_fromlong){
        auto pyobj = f();
        fw.long_fromlong_ret_values.push_back(pyobj); 
    }
    for(auto& f: fw.long_aslong){
        auto pyobj = f();
        fw.long_aslong_ret_values.push_back(pyobj); 
    }
    for(auto& f: fw.list_size){
        auto pyobj = f();
        fw.list_size_ret_values.push_back(pyobj); 
    }
    for(auto& f: fw.list_setitem){
        f();
    }

    PyGILState_Release(state);

    fw.flag = !fw.flag;

    fw.list_setitem.clear();
    fw.list_new.clear();
    fw.list_append.clear();
    fw.list_size.clear();
    fw.list_decref.clear();
    fw.long_fromlong.clear();
    fw.long_aslong.clear();
    fw.float_fromdouble.clear();

    fw.counter =0;

    int batch_size = fw.list_size_ret_values.size()+fw.list_new_ret_values.size()+fw.long_aslong_ret_values.size()+fw.long_fromlong_ret_values.size()+fw.float_fromdouble_ret_values.size();
    fw.cv.notify_all();



    flag_lock.unlock();

    auto temp_counter = &fw.counter;
    std::unique_lock<std::mutex> cv_lock(fw.cv_mutex);
    fw.cv.wait(cv_lock,[temp_counter,batch_size]{return *temp_counter==batch_size;});
    cv_lock.unlock();
    return true;
    
}

void delete_pyobject(PyObject*& o){
    if(o->ob_refcnt!=1){
        cout<<"WARNING: DELETE PYOBJECT NOT REF CNT 1: "<<o->ob_refcnt<<endl;
    }
    o->ob_refcnt=1;
    Py_DECREF(o);
    o=nullptr;
}
void send_to_model(PyObject* agent_function,std::shared_ptr<ModelConcurrency> mc){

    std::vector<bool> inverts;
    

    std::unique_lock <std::mutex> vec_lock(mc->vec_mutex);
    std::unique_lock <std::mutex> flag_lock(mc->flag_mutex);
    mc->counter=0;
    
    
    auto batch_size = mc->vec.size();
    if(batch_size==0){
        mc->flag=!mc->flag;
        return;
    }
    PyObject* pylist = event_PyList_New(mc->vec.size(),mc);
    auto& boards = mc->vec;
    for(int i = 0;i<batch_size;i++){
        auto& board =boards[i];

        
        auto invert =board.turn==Turn::X;

        inverts.push_back(invert);
        event_PyList_SetItem(pylist,i,board_to_list(board,invert,mc),mc); 
    }
    PyObject* values; //tensors
    PyObject* policies; //tensors
                        //
    PyObject* models = event_PyList_New(mc->models.size(),mc);
    int i =0;
    for(auto& model : mc->models){
        event_PyList_SetItem(models,i,model,mc);
        Py_INCREF(model);
        i++;
    }
    assert(mc->model_ids.size()==batch_size);
    PyObject* model_ids = event_PyList_New(mc->model_ids.size(),mc);
    PyObject* alphas= event_PyList_New(mc->alphas.size(),mc);

    i=0;
    for(auto id: mc->model_ids){
        event_PyList_SetItem(model_ids,i,event_PyLong_FromLong(id,mc),mc);
        i++;
    }
    i=0;
    for(auto alpha: mc->alphas){
        event_PyList_SetItem(alphas, i, event_PyFloat_FromDouble(alpha,mc),mc);
        i++;
    }
    PyObject* parameters = event_PyTuple_Pack(pylist,models,model_ids,alphas,mc);


    PyObject* result = event_PyObject_CallObject(agent_function,parameters,mc);
    

    f_Py_DECREF(parameters,mc);

    f_Py_DECREF(model_ids,mc);
    f_Py_DECREF(models,mc);
    f_Py_DECREF(pylist,mc); 

    f_Py_DECREF(alphas,mc);
    if(!PyArg_ParseTuple(result,"OO",&values,&policies)){
        cout<<"trouble parsing rip"<<endl;
    }
    for(int i = 0;i<batch_size;i++){
        auto policy_value = PolicyValue();
        mc->ret_values.push_back(policy_value);
        for(int k=0;k<9;k++){
            mc->ret_values.q[i].policy[k]=event_PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(policies,i),k),mc);
        }
        mc->ret_values.q[i].value = PyFloat_AsDouble(PyList_GetItem(values,i));
    }
    f_Py_DECREF(result,mc);

    mc->vec.clear();
    mc->model_ids.clear();
    mc->alphas.clear();
    mc->flag=!mc->flag;


    mc->cv.notify_all();
    flag_lock.unlock();
    std::unique_lock waiting_lock(mc->counter_mutex);
    auto& k = mc->counter;
    auto temp_p = &mc->counter;
    mc->cv.wait(waiting_lock, [temp_p,batch_size]{return (*temp_p)==batch_size;});
    mc->ret_values.clear();
    
    assert((*temp_p)==batch_size);

}
void Tree::run_dependent(int i, int thread_count,std::shared_ptr<ModelConcurrency> mc){
    try {
    std::atomic<int> iter_counter;
    // std::thread(&Tree::run_thread,this,i).join();
    std::vector<std::thread> threads;
    for(int k =0;k<thread_count-1;k++){
        threads.push_back(std::thread(&Tree::run_thread,this,i,&iter_counter));
        // std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
        std::thread(&Tree::run_thread,this,i,&iter_counter).join();
    for(auto& t:threads){
        t.join();
    }
#ifdef DEBUG
    std::cout<<"done!"<<iter_counter.load()<<endl;
#endif
    }catch (const std::exception &exc)
    {
        std::cerr << exc.what();
    }

    }
TicTacToe Tree::make_play(){
    auto max_node = head.children[0].get();

    for(int i =1;i<head.children.size();i++){
        if(head.children[i]->visits>max_node->visits){
            max_node = head.children[i].get();
        }
    }
    return max_node->board;
}
