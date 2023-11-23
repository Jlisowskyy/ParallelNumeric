$Path = "C:\Users\Jlisowskyy\Desktop\Projekty\ParallelNumeric\asm\CompiledAsm"
$OldPath = Get-Location

Set-Location $Path
g++ -masm=intel -march=native -c (Get-ChildItem ../*.s).FullName
g++ (Get-ChildItem *.o) -fopenmp -mavx -mavx2 -mfma -march=native -o ../../CompiledAsm.exe
Set-Location $OldPath