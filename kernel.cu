
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Eigen/Sparse>
#include "util.h"
#include <iostream>
#include <cusparse.h>
// error check macros
#define CUSPARSE_CHECK(x) {cusparseStatus_t _c=x; if (_c != CUSPARSE_STATUS_SUCCESS) {printf("cusparse fail: %d, line: %d\n", (int)_c, __LINE__); exit(-1);}}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

int main(int argc, char* argv[])
{
	srand(1);
	std::cout << "Args: " << argv[1] << ", " << argv[2] << ", " << argv[3] << "\n";
	std::cout << "Generating sparse matrix\n";
	unsigned int n_row, n_col, entry_per_row;
	n_row = atoi(argv[1]);
	n_col = atoi(argv[2]);
	entry_per_row = atoi(argv[3]);
	Eigen::SparseMatrix<double, Eigen::RowMajor> A = generate_sparse_matrix_average<double, Eigen::RowMajor>(n_row, n_col, entry_per_row);
	Eigen::SparseMatrix<double, Eigen::RowMajor> AT = A.transpose();
	std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> sparse_matrix_vector;
	sparse_matrix_vector.push_back(A);
	// std::cout << a << "\n";

	std::cout << "Performing AtA for ground truth\n";
	Eigen::SparseMatrix<double, Eigen::RowMajor> ATA = A.transpose() * A;
	// std::cout<<ata<<"\n";

	std::cout << "Using mkl\n";
	sparse_matrix_t R1_mkl_t;
	PROFILE_MKL_MULTI_SYRK(sparse_matrix_vector, R1_mkl_t);
	auto mkl_info = extract_value(R1_mkl_t);
	auto R1_mkl = ConstructSparseMatrix(std::get<0>(mkl_info), std::get<1>(mkl_info), std::get<2>(mkl_info), (std::get<5>(mkl_info)).data(), (std::get<3>(mkl_info)).data(), (std::get<4>(mkl_info)).data());

	// std::cout << "Ground Truth: \n" << ATA << "\n\n";
	// std::cout << "MKL Result: \n" << R1_mkl << "\n\n";

	// ======================================================================================//
	// setting items for A in cusparse                                                       //
	// ======================================================================================//
	cusparseStatus_t stat;
	cusparseHandle_t hndl;
	// getting the necessary pointers
	int* csrRowPtrA,* csrRowPtrC, * csrColIndA,  * csrColIndC;
	int* h_csrRowPtrA,  * h_csrRowPtrC, * h_csrColIndA,  * h_csrColIndC;
	double* csrValA, * csrValC, * h_csrValA, * h_csrValC;

	// setting up paramaters
	int nnzA, nnzC;
	int m, n, k;
	m = A.cols();
	n = A.rows();
	k = A.cols();
	// get nnz
	nnzA = A.nonZeros();
	// malloc the array
	h_csrRowPtrA = (int*)malloc((n + 1) * sizeof(int));
	h_csrColIndA = (int*)malloc(nnzA * sizeof(int));
	h_csrValA = (double*)malloc(nnzA * sizeof(double));
	if ((h_csrRowPtrA == NULL) || (h_csrColIndA == NULL) || (h_csrValA == NULL))
	{
		printf("malloc fail\n"); return -1;
	}


	// setting up the values for matrix A in csr format
	for (unsigned int i = 0; i < A.rows()+1; i++) {
		h_csrRowPtrA[i] = A.outerIndexPtr()[i];
	}
	for (unsigned int i = 0; i < A.nonZeros(); i++) {
		h_csrColIndA[i] = A.innerIndexPtr()[i];
		h_csrValA[i] = A.valuePtr()[i];
	}
	cudaMalloc(&csrRowPtrA, (n + 1) * sizeof(int));
	cudaMalloc(&csrColIndA, nnzA * sizeof(int));
	cudaMalloc(&csrValA, nnzA * sizeof(double));
	cudaCheckErrors("cudaMalloc fail");
	cudaMemcpy(csrRowPtrA, h_csrRowPtrA, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csrColIndA, h_csrColIndA, nnzA * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csrValA, h_csrValA, nnzA * sizeof(double), cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy fail");


	// setting up descr
	cusparseMatDescr_t descrA, descrC;
	CUSPARSE_CHECK(cusparseCreate(&hndl));
	stat = cusparseCreateMatDescr(&descrA);
	CUSPARSE_CHECK(stat);
	stat = cusparseCreateMatDescr(&descrC);
	CUSPARSE_CHECK(stat);
	stat = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	CUSPARSE_CHECK(stat);
	stat = cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
	CUSPARSE_CHECK(stat);
	stat = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
	CUSPARSE_CHECK(stat);
	stat = cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
	CUSPARSE_CHECK(stat);
	cusparseOperation_t transA = CUSPARSE_OPERATION_TRANSPOSE;
	cusparseOperation_t transB = CUSPARSE_OPERATION_NON_TRANSPOSE;



	// ======================================================================================//
	// cusparse gemm first stage, alloc space                                                //
	// ======================================================================================//
	// figure out size of C
	int baseC;
	// nnzTotalDevHostPtr points to host memory
	int* nnzTotalDevHostPtr = &nnzC;
	stat = cusparseSetPointerMode(hndl, CUSPARSE_POINTER_MODE_HOST);
	CUSPARSE_CHECK(stat);
	cudaMalloc((void**)&csrRowPtrC, sizeof(int) * (m + 1));
	cudaCheckErrors("cudaMalloc fail");
	stat = cusparseXcsrgemmNnz(hndl, transA, transB, m, n, k,
		descrA, nnzA, csrRowPtrA, csrColIndA,
		descrA, nnzA, csrRowPtrA, csrColIndA,
		descrC, csrRowPtrC, nnzTotalDevHostPtr);
	CUSPARSE_CHECK(stat);
	if (NULL != nnzTotalDevHostPtr) {
		nnzC = *nnzTotalDevHostPtr;
	}
	else {
		cudaMemcpy(&nnzC, csrRowPtrC + m, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&baseC, csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
		cudaCheckErrors("cudaMemcpy fail");
		nnzC -= baseC;
	}
	cudaMalloc((void**)&csrColIndC, sizeof(int) * nnzC);
	cudaMalloc((void**)&csrValC, sizeof(double) * nnzC);
	cudaCheckErrors("cudaMalloc fail");

	// ======================================================================================//
	// cusparse gemm second stage, perform A*B                                               //
	// ======================================================================================//
	auto elapsed = benchmarkTimer([&]() {
		for (unsigned int i = 0; i < 100; i++) {
			stat = cusparseDcsrgemm(hndl, transA, transB, m, n, k,
				descrA, nnzA,
				csrValA, csrRowPtrA, csrColIndA,
				descrA, nnzA,
				csrValA, csrRowPtrA, csrColIndA,
				descrC,
				csrValC, csrRowPtrC, csrColIndC);
		}
	});
	std::cout << "CUSPARSE SYRK: " << elapsed << " us\n";
	CUSPARSE_CHECK(stat);
	// copy c back to our memory from gpu
	h_csrRowPtrC = (int*)malloc((m + 1) * sizeof(int));
	h_csrColIndC = (int*)malloc(nnzC * sizeof(int));
	h_csrValC = (double*)malloc(nnzC * sizeof(double));
	if ((h_csrRowPtrC == NULL) || (h_csrColIndC == NULL) || (h_csrValC == NULL))
	{
		printf("malloc fail\n"); return -1;
	}
	cudaMemcpy(h_csrRowPtrC, csrRowPtrC, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_csrColIndC, csrColIndC, nnzC * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_csrValC, csrValC, nnzC * sizeof(double), cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy fail");


	// reconstruct R1_cusparse using the main memory
	auto R1_cusparse = ConstructSparseMatrix(m, k, nnzC, h_csrValC, h_csrRowPtrC, h_csrColIndC);





	// ======================================================================================//
	// Allocate dense A in column major                                                      //
	// ======================================================================================//
	Eigen::MatrixXd A_dense_matrix = Eigen::MatrixXd(A);
	// std::cout << A_dense_matrix << "\n";
	double* A_dense, *h_A_dense, *C_dense, *h_C_dense;
	// for the dense matrix of A
	h_A_dense = (double*)malloc(A_dense_matrix.rows() * A_dense_matrix.cols() * sizeof(double));
	// for the result
	h_C_dense = (double*)malloc(A_dense_matrix.rows() * A_dense_matrix.cols() * sizeof(double));

	// append entries according to column order
	unsigned int d_count = 0;
	for (unsigned int i = 0; i < A_dense_matrix.cols(); i++) {
		for (unsigned int j = 0; j < A_dense_matrix.rows(); j++) {
			h_A_dense[d_count] = A_dense_matrix(j, i);
			d_count++;
		}
	}

	cudaMalloc(&A_dense, (A_dense_matrix.rows() * A_dense_matrix.cols()) * sizeof(double));
	cudaMalloc((void**)&C_dense, (A_dense_matrix.cols() * A_dense_matrix.cols()) * sizeof(double));
	cudaCheckErrors("cudaMalloc fail");
	cudaMemcpy(A_dense, h_A_dense, (A_dense_matrix.rows() * A_dense_matrix.cols()) * sizeof(double), cudaMemcpyHostToDevice);
	cudaCheckErrors("cudaMemcpy fail");
	cudaMemset((void*)C_dense, 0, (A_dense_matrix.rows() * A_dense_matrix.cols()) * sizeof(double));
	cudaCheckErrors("cudaMemset fail");

	// ======================================================================================//
	// perform csrmm2                                                                        //
	// ======================================================================================//
	const double lbd_bt = k;
	const double ldc = m;
	double alpha = 1.0;
	double beta = 0.0;
	elapsed = benchmarkTimer([&]() {
		for (unsigned int i = 0; i < 100; i++) {
			stat = cusparseDcsrmm2(hndl,
				CUSPARSE_OPERATION_TRANSPOSE,
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				m, n, k, nnzA, &alpha,
				descrA, csrValA, csrRowPtrA, csrColIndA,
				A_dense, lbd_bt,
				&beta, C_dense, ldc);
		}
	});
	std::cout << "CUSPARSE CSRMM SYRK: " << elapsed << " us\n";
	CUSPARSE_CHECK(stat);


	cudaMemcpy(h_C_dense, C_dense, m * n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaCheckErrors("cudaMemcpy fail");

	d_count = 0;
	Eigen::MatrixXd R2_cusparse(m, n);
	for (unsigned int i = 0; i < m; i++) {
		for (unsigned int j = 0; j < n; j++) {
			R2_cusparse(i, j) = h_C_dense[d_count];
			d_count++;
		}
	}



	// std::cout << "Ground Truth: \n" << ATA << "\n\n";
	// std::cout << "MKL Result: \n" << R1_mkl << "\n\n";
	// std::cout << "CUSPARSE RESULT: \n" << R1_cusparse << "\n\n";
	// std::cout << "CUSPARSE CSRMM2 RESULT: \n" << R2_cusparse << "\n\n";
	std::cout << "Checking corretness of cusparse by checking the norm of difference (we assume the correctness of MKL SYPR already):\n";
	std::cout << (ATA - R1_cusparse).norm() << "\n";
	std::cout << (ATA - R2_cusparse).norm() << "\n";

    return 0;
}

