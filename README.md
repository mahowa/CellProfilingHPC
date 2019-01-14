# CellProfilingHPC
Final for High performance computing 2018

## To install:
run install_packages.sh  *Must have root access -- type 'sh install_packages.sh' to execute


## To Run:
mpirun -n 4 python cell_profiler.py 4

* change integer value after -n to adjust how many mpi tasks to use
* change last int value to adjust how many openmp threads to use

For ease there is a run.sh file that can be modified. type 'sh run.sh' to execute



To change the files used you will need to manually change them in cell_profiler.py or rename the files to what the program wants
