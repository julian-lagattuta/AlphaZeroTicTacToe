#to compile this yourself, follow the comments in this file
#the settings are currently set for a python3.8 on Linux


project(PY)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++2a -lpthread -g -Wall  "   )


set(SOURCES 
    src/TicTacToe.cpp
    src/pywrapper.cpp
)

#if you want to, uncomment the next line to enable debug output which contains cool stuff if you want to see whats happening
add_compile_definitions(DEBUG)


# Change the include directory to your python include directory
include_directories( /usr/include/python3.8/)
add_library( tictactoelib SHARED ${SOURCES})


# Change the lib file  to your python lib file
target_link_libraries(tictactoelib /usr/local/lib/python3.8) 



add_custom_command(TARGET tictactoelib POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tictactoelib> ../src/tictactoelib.so
)
