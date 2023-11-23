$MyPath = Get-Location
Set-Location /Users/Jlisowskyy/Desktop/Projekty/ParallelNumeric/asm
g++ -std=c++20 -mfma -mavx -mavx2 -O3 -fopenmp -S -masm=intel -march=native ../main.cpp  ../Src/Debuggers.cpp ../Src/ResourceManager.cpp ../Src/ErrorCodes.cpp ../Src/Matrix.cpp ../Src/NumericalCore.cpp ../Src/ParallelNumeric.cpp ../Src/MatrixMultiplication.cpp ../Src/Vector.cpp
Set-Location ${MyPath}