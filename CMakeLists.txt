project(PY)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -lpthread -g -Wall  "   )

# cmake with -DCMAKE_BUILD_TYPE=Debug
set(SOURCES 
    src/TicTacToe.cpp
    src/pywrapper.cpp
)
include_directories( C:/Users/Julian/AppData/Local/Programs/Python/Python37/include )
add_library( tictactoelib SHARED ${SOURCES})
target_link_libraries(tictactoelib C:/Users/Julian/AppData/Local/Programs/Python/Python37/libs/python37.lib)



add_custom_command(TARGET tictactoelib POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tictactoelib> ../src/tictactoelib.pyd
)