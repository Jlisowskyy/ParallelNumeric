#!/bin/bash
VAR1="GccOutput"

if [[ $1 != "" ]] 
then 
    VAR1=$1
fi

time g++ -O3 -mavx -mfma -march=native -std=c++20 main.cpp Src/*.cpp -o $VAR1
exit $?