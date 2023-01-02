
# AlphaZero TicTacToe
An implementation of the AlphaZero algorithm for Tic Tac Toe + parallel MCTS written in C++.

This is a work in progress. As of now, only parallel MCTS is working.

# Building

First, edit CMakeLists.txt:

 - Replace the Python include directory with your Python include directory.
 - Replace the Python lib file with your Python lib file



Build the pyd file written in C++ to run in Python.

    mkdir build
    cd build
    cmake .. -G "[generator that uses g++/clang]"
    cmake --build .
    
   Run src/run.py in python
