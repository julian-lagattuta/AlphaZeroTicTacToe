
# AlphaZero TicTacToe
An implementation of the AlphaZero algorithm for Tic Tac Toe + parallel MCTS written in C++.

This is a work in progress. As of now, only parallel MCTS is working.

# Building
First build the pyd file written in C++ to run in Python.

    mkdir build
    cd build
    cmake .. -G "[generator of choice. preferably not MSVC because it is made by Microsoft :)]"
    cmake --build .
    
   Run src/run.py in python
