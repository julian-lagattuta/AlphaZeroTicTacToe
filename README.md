
# AlphaZero TicTacToe

This is a work in progress. An implementation of the AlphaZero AI algorithm for TicTacToe. Currently, supports parallelized Monte Carlo Tree Search for maximum performance. As of now, only parallel MCTS is working. The Python code calls into the C++ code, which calls into Python when it needs to use the neural network. AlphaZero, developed by OpenAI, could be the world’s best chess and Go player. The algorithm is a cross between neural networks and Monte Carlo Tree Search. Making this to learn  more about AI, C++, and the Python C API. Now working on its ability to train and on Python C API issues. 
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
