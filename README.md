
# MyMathInterpreter

## Table of Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
4. [Roadmap](#roadmap)
5. [License](#license)
## Introduction

MyMathInterpreter is an educational project that aims to implement a simple C-like language capable of fast numerical operations. 
This project serves as an exploration into the world of compilers and interpreters.

## Motivation

The primary objective of this project is to deepen my understanding of compilers and interpreters. 
Programming languages have always fascinated me, and I believe 
that knowledge in this area is not only essential when implementing such systems but also beneficial for any programmer. 
Predicting and understanding compiler/interpreter actions is crucial for becoming a proficient programmer.

## Getting Started

### Prerequisites

- just any c++20 compatible compiler

### Installation

Begin by cloning the repository to your local machine:
```shell
git clone https://github.com/Jlisowskyy/MyMathInterpreter ; cd MyMathInterpreter
```

Tailor the compilation process to your preferences by checking and modifying compilation flags. 
Open the following file:
```shell 
vim include/globalValues.h
```

Compile the program using your preferred C++ compiler, for instance, gcc:
```shell
g++ -O3 -march=native -std=c++20 -o MyMathInterpreter main.cpp src/*
```

Execute the interpreter to test its functionality (Please note: CLI is not yet implemented):
```shell
./MyMathInterpreter [FILENAME]
```

## Roadmap

As the project unfolds, it is currently in its early stages, progressing at a slowed pace due to ongoing university duties. 
The envisioned future features include:

- [x] Working lexer
- [x] Simple expression evaluations
- [ ] Hierarchical operators and expression evaluation
- [ ] Introduce functions and classes, possibly pre-implemented classes without defining options
- [ ] Integration with my own numerical library
- [ ] Potential implementation of multithreading capabilities
- [ ] Unix-like CLI for enhanced user interaction
- [ ] Interactive interpreter within CLI environment
- [ ]  Explore the possibility of compilation capabilities, though this is anticipated towards the project's end

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
