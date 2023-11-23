# ParallelNumeric

## Table of Contents
1. [Introduction](#introduction)
    - [Details](#details)
2. [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
3. [Roadmap](#roadmap)
4. [License](#license)
## Introduction

ParallelNumeric is an educational library project that implements diverse multi-threaded numerical solutions. 

### Details

It offers a straightforward interface for numerical operations encapsulated within two main classes: Vector and Matrix. 
The primary aim of this project was to delve into performance-focused programming 
and explore the tools and technologies employed to achieve optimal solutions.
The library leverages OpenMP and Intel intrinsics,
harnessing AVX2 and multiple CPU threads to obtain maximally cache-friendly algorithms.
Presently, only static linking is supported,
and CPU details detection is not also implemented, so the library is well-fitted
only for intel 13600k CPU.

## Getting Started

### Prerequisites

- just any c++20 compatible compiler

### Installation

Begin by cloning the repository into your project directory:
```shell
git clone https://github.com/Jlisowskyy/ParallelNumeric 
```

To include headers in your project, use a relative path, for instance, to include all types:
```C
#include "ParallelNumeric/Include/Wrappers/OptimalOperations.hpp"
```

Compile the program using your preferred C++ compiler, for instance, gcc:
```shell
g++ -O3 -march=native -fopenmp -std=c++20 YOUR_FLAGS -o YOUR_PROGRAM_NAME main.cpp  YOUR_SOURCES ParallelNumeric/Src/*
```
## Roadmap

As the project unfolds, it is currently in its early stages, progressing at a slowed pace due to ongoing university duties. 
The envisioned future features include:

- [x] Working Matrix Multiplication
- [x] All basic Vector operations
- [ ] Perform tests against state-of-the-art solutions
- [ ] All basic Matrix, Matrix&Vector operations
- [ ] Static linking
- [ ] Global resource management strategy
- [ ] Detailed documentation
- [ ] Provide simple examples
- [ ] More complex numerical algorithms and operations

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.
