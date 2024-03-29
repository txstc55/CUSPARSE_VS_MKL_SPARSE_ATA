# CUSPARSE_VS_MKL_SPARSE_ATA

This code tests the performance of ATA with the two major library: [Math Kernel Library(MKL)](https://software.intel.com/en-us/mkl) and [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html).

For MKL, we will use the [mkl_sparse_sypr](https://software.intel.com/en-us/mkl-developer-reference-c-mkl-sparse-sypr) routine to compute ATA. Note this routine is normally for computing ATBA. However, we can set the B matrix to be a diagonal unit matrix to perform the two-stage of [mkl_sparse_syrk](https://software.intel.com/node/e8ee46bf-389a-4809-823d-98333a0ec0eb). See the note at descrB.

For cuSPARSE, we will use [cusparseDcsrgemm](https://docs.nvidia.com/cuda/cusparse/index.html#csrgemm) routine and [csrmm2](https://docs.nvidia.com/cuda/cusparse/index.html#csrmm2) routine. For the fist one two matrices are both matrices are sparse while the second routine has one sparse matrix with one dense matrix.

For csrmm2, the sparse matrix is converted to dense then store the data in column major as a double array. The C matrix is initialized to all 0, alpha is 1 and beta is 0, and the sparse matrix's operation is set to CUSPARSE_OPERATION_TRANSPOSE.

User has to supply three arguments: number of rows, number of columns and average number of non-zeros in matrix A. All matrices generated will be type of double.

User has to link MKL and cuSPARSE library manually, this repo only provides the code. For best performance, please turn on the parallel flag for MKL.
