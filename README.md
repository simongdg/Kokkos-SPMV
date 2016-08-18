# Kokkos-SPMV
Sparce Matrix-vector Multiplication program use for Kokkos-tools development

Dependencies: <br />
1)clone Trilinos(https://github.com/trilinos/Trilinos) <br />
2)cd Trilinos/packages/tpetra/kernels/src/impl <br />
3)touch TpetraKernels_config.h <br />
4)cd ../../../../ <br />
5)rm -r kokkos <br />
6)clone kokkos from here: https://github.com/simongdg/kokkos <br />
7)cd kokkos <br />
8)clone kokkos-tools from here: https://github.com/simongdg/kokkos-tools <br />
9)cd kokkos-tools/src/tools/autoTunner-kernel-timer (and/or other tools if you want) <br />
10)make (this should produce a .so file) <br />
<br />
<br />

Compiling: <br />
1)clone this repo <br />
2)Enable autotuning by commenting in/out line 719 xor 720 in test_crsmatrix.cpp <br />
1)Open the Makefile <br />
2)Modify the TRILINOS_PATH to point to where Trilinos reside <br />
3)Modify the KOKKOS_ARCH according to your architecture, options can be found in the Kokkos programming guide <br />
4)make -j8 KOKKOS_DEVICES=Cuda. Other possible KOKKOS_DEVICES can be found in the kokkos programming guide <br />
<br />
<br />

Running: <br />
1)export KOKKOS_PROFILE_LIBRARY=Trilinos/packages/kokkos/kokkos-tools/src/tools/autoTunner-kernel-timer/kp_kernel_autoTuner.so <br />
2)srun ./test_matvec.cuda-new -fb YOUR/MATRICES/HOME/PATH/DIMACS10/road_central/road_central.mtx <br />
or <br />
2)python test.py matrices.txt (make sure you modify the mm_path in the python scrip before you use it) <br />
or <br />
2)python test2.py matrices.txt matrices_param_K20x.txt (same as above, example parameter file is provided in this repo) <br />
<br />
NOTE: the matrices should be in binary from. <br />
<br />
<br />

Generating binary matrix from Matrix Market: <br />
1)Download a matrix in Matrix Market fromat <br />
2)srun ./test_matvec.cuda-new -f matrixName.mtx <br />
3)The above command will generate two files matrixName.mtx_col and matrixName.mtx_row <br />
4)Manually copy the header of the matrixName.mtx onto a file called matrixName.mtx_descr <br />

