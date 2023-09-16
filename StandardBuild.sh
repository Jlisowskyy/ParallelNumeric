#!/bin/bash

OutputName="CompOut"
rm $OutputName

echo Build time measures:
./Scripts/BuildGcc.bash $OutputName

if [ $? -eq 0 ]
then
    echo Execution time measures:
    time ./${OutputName}
fi