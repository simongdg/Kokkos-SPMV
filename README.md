# Kokkos-SPMV
Sparce Matrix-vector Multiplication program use for Kokkos-tools development

Dependencies: <br />
1)clone Trilinos(https://github.com/trilinos/Trilinos) <br />
2)cd Trilinos/packages/tpetra/kernels/src/impl
3)touch TpetraKernels_config.h
4)cd ../../../../
5)rm -r kokkos
6)clone kokkos from here: https://github.com/simongdg/kokkos
7)cd kokkos
8)clone kokkos-tools from here: https://github.com/simongdg/kokkos-tools
9)cd kokkos-tools/src/tools/autoTunner-kernel-timer (and/or other tools if you want)
10)make (this should produce a .so file)


Compiling:
1)clone this repo
2)Enable autotuning by commenting in/out line 719 xor 720 in test_crsmatrix.cpp 
1)Open the Makefile
2)Modify the TRILINOS_PATH to point to where Trilinos reside
3)Modify the KOKKOS_ARCH according to your architecture, options can be found in the Kokkos programming guide
4)make -j8 KOKKOS_DEVICES=Cuda. Other possible KOKKOS_DEVICES can be found in the kokkos programming guide


Running:
1)export KOKKOS_PROFILE_LIBRARY=Trilinos/packages/kokkos/kokkos-tools/src/tools/autoTunner-kernel-timer/kp_kernel_autoTuner.so
2)srun ./test_matvec.cuda-new -fb YOUR/MATRICES/HOME/PATH/DIMACS10/road_central/road_central.mtx
or
2)python test.py matrices.txt (make sure you modify the mm_path in the python scrip before you use it)
or
2)python test2.py matrices.txt matrices_param_K20x.txt (same as above, example parameter file is provided in this repo)

NOTE: the matrices should be in binary from. 


Generating binary matrix from Matrix Market:
1)Download a matrix in Matrix Market fromat
2)srun ./test_matvec.cuda-new -f matrixName.mtx
3)The above command will generate two files matrixName.mtx_col and matrixName.mtx_row
4)Manually copy the header of the matrixName.mtx onto a file called matrixName.mtx_descr

