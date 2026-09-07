/* ==========================================================================
 * GPU-Accelerated Linear Programming Engine (SIH / Enterprise Optimization)
 * Algorithm: Primal-Dual Hybrid Gradient (PDHG), restarted, Ruiz-equilibrated
 * Architecture: CUDA C++ (cuSPARSE, cuBLAS), CUDA Graph-captured inner loop
 *
 * ==========================================================================
 * UPDATED VERSION -- GPU/frontend integration patch
 * --------------------------------------------------------------------------
 * Only change vs. the original file: the solver now takes a second CLI
 * argument, an output JSON path, and writes its result there as JSON
 * instead of printing a CSV summary line to stdout and dumping x to a
 * separate solution_x.txt file:
 *
 *     gpu_solver.exe input.txt output.json      (Windows -- built as .exe)
 *     ./gpu_solver   input.txt output.json      (Linux/Mac -- no extension)
 *
 * The JSON field names/shape match frontend_cuda_crusaders_v2.py's
 * run_real_solver_pipeline() EXACTLY, as read from that file directly:
 * it already has the subprocess call `[_GPU_SOLVER_PATH, gpu_input_file,
 * output_file]` and `gpu_json.get(...)` calls for "status" (must equal
 * "OPTIMAL"), "objective_value", "iterations", and
 * "performance"."gpu_compute_time_ms" (nested) -- this file was checked
 * against that code, not guessed. No solver math, kernel, preconditioning,
 * or convergence logic was touched -- see "UPDATED" comments below for the
 * only edited regions (argument parsing + the final JSON-output block).
 * ========================================================================== */

#define USE_FP64 0   // <-- flip this and recompile to compare wall-clock time

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <cstdlib>
#include <cctype>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#if USE_FP64
  typedef double real_t;
  #define CUDA_REAL_TYPE CUDA_R_64F
  #define cublasXdot   cublasDdot
  #define cublasXnrm2  cublasDnrm2
  #define fabsX        fabs
  #define fmaxX        fmax
  #define fminX        fmin
  #define strtoX       strtod
#else
  typedef float real_t;
  #define CUDA_REAL_TYPE CUDA_R_32F
  #define cublasXdot   cublasSdot
  #define cublasXnrm2  cublasSnrm2
  #define fabsX        fabsf
  #define fmaxX        fmaxf
  #define fminX        fminf
  #define strtoX       strtof
#endif

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ \
                      << " status=" << status << "\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__ \
                      << " status=" << status << "\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void update_dual_kernel(real_t* y, const real_t* Ax, const real_t* b,
                                   const real_t* sigma, const int* row_type, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        real_t new_y = y[i] + sigma[i] * (Ax[i] - b[i]);
        if (row_type[i] == 1) {
            y[i] = new_y;
        } else if (row_type[i] == 0) {
            y[i] = fmaxX(new_y, (real_t)0.0);
        } else if (row_type[i] == -1) {
            y[i] = fminX(new_y, (real_t)0.0);
        } else {
            y[i] = new_y;
        }
    }
}

__global__ void update_primal_kernel(const real_t* x_old, real_t* x_new,
                                     const real_t* ATy, const real_t* c,
                                     const real_t* tau, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
        real_t val = x_old[j] - tau[j] * (ATy[j] + c[j]);
        val = fmaxX(val, (real_t)0.0);
        x_new[j] = val;
    }
}

__global__ void extrapolate_kernel(real_t* x_old, const real_t* x_new, real_t* x_bar, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
        real_t current = x_new[j];
        real_t past = x_old[j];
        x_bar[j] = (real_t)2.0 * current - past;
        x_old[j] = current;
    }
}

__global__ void compute_primal_violation(const real_t* Ax, const real_t* b, const int* row_type, real_t* violation, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        real_t diff = Ax[i] - b[i];
        if (row_type[i] == 1) violation[i] = diff;
        else if (row_type[i] == 0) violation[i] = (diff > (real_t)0.0) ? diff : (real_t)0.0;
        else if (row_type[i] == -1) violation[i] = (diff < (real_t)0.0) ? diff : (real_t)0.0;
        else violation[i] = diff;
    }
}

__global__ void compute_dual_violation(const real_t* ATy, const real_t* c, real_t* violation, int N) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < N) {
        real_t diff = ATy[j] + c[j];
        violation[j] = (diff < (real_t)0.0) ? -diff : (real_t)0.0;
    }
}

// Memory-mapped parsing with DOUBLE-PRECISION Ruiz Equilibration (Zero runtime GPU cost)
void read_and_compress_matrix(const char* filename, int& M, int& N, int& nnz,
                              std::vector<real_t>& c, std::vector<real_t>& b,
                              std::vector<real_t>& csr_values, std::vector<int>& csr_col_idx, std::vector<int>& csr_row_ptr,
                              std::vector<real_t>& csc_values, std::vector<int>& csc_row_idx, std::vector<int>& csc_col_ptr,
                              std::vector<real_t>& tau_vec, std::vector<real_t>& sigma_vec,
                              std::vector<int>& row_type,
                              std::vector<real_t>& col_scale_out, real_t& scale_b, std::vector<real_t>& c_orig) {

    int fd = open(filename, O_RDONLY);
    if (fd < 0) { std::cerr << "Failed to open " << filename << "\n"; exit(EXIT_FAILURE); }
    struct stat sb;
    fstat(fd, &sb);
    const char* p = (const char*)mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) { std::cerr << "mmap failed\n"; exit(EXIT_FAILURE); }
    const char* end = p + sb.st_size;

    auto skip_ws = [&]() { while (p < end && std::isspace(*p)) ++p; };
    auto read_int = [&]() { skip_ws(); char* next; int v = std::strtol(p, &next, 10); p = next; return v; };
    auto read_real = [&]() { skip_ws(); char* next; real_t v = strtoX(p, &next); p = next; return v; };

    M = read_int(); N = read_int(); nnz = read_int();

    c.resize(N);
    for (int i = 0; i < N; ++i) c[i] = read_real();
    c_orig = c;

    b.resize(M);
    for (int i = 0; i < M; ++i) b[i] = read_real();

    row_type.resize(M);
    for (int i = 0; i < M; ++i) row_type[i] = read_int();

    real_t max_b = 0, max_c = 0;
    for (real_t val : b) max_b = std::max(max_b, std::abs(val));
    for (real_t val : c) max_c = std::max(max_c, std::abs(val));
    scale_b = (max_b > (real_t)1e-12) ? ((real_t)1.0 / max_b) : (real_t)1.0;
    real_t scale_c = (max_c > (real_t)1e-12) ? ((real_t)1.0 / max_c) : (real_t)1.0;
    for (real_t& val : b) val *= scale_b;
    for (real_t& val : c) val *= scale_c;

    csr_row_ptr.assign(M + 1, 0);
    std::vector<int> raw_row(nnz), raw_col(nnz);
    std::vector<real_t> raw_val(nnz);

    for (int i = 0; i < nnz; ++i) {
        raw_row[i] = read_int(); raw_col[i] = read_int(); raw_val[i] = read_real();
        csr_row_ptr[raw_row[i] + 1]++;
    }
    for (int i = 0; i < M; ++i) csr_row_ptr[i + 1] += csr_row_ptr[i];

    munmap((void*)((const char*)p), sb.st_size);
    close(fd);

    // Double-precision internal preconditioning pass
    std::vector<double> col_scale_d(N, 1.0);
    std::vector<double> row_scale_d(M, 1.0);
    std::vector<double> raw_val_d(nnz);
    for(int i=0; i<nnz; ++i) raw_val_d[i] = static_cast<double>(raw_val[i]);

    for (int pass = 0; pass < 10; ++pass) {
        std::vector<double> row_norm(M, 0.0), col_norm(N, 0.0);
        for (int i = 0; i < nnz; ++i) {
            double scaled = std::abs(raw_val_d[i] * row_scale_d[raw_row[i]] * col_scale_d[raw_col[i]]);
            row_norm[raw_row[i]] = std::max(row_norm[raw_row[i]], scaled);
            col_norm[raw_col[i]] = std::max(col_norm[raw_col[i]], scaled);
        }
        for (int i = 0; i < M; ++i) if (row_norm[i] > 1e-15) row_scale_d[i] /= std::sqrt(row_norm[i]);
        for (int j = 0; j < N; ++j) if (col_norm[j] > 1e-15) col_scale_d[j] /= std::sqrt(col_norm[j]);
    }

    std::vector<double> scaled_row_sums_d(M, 0.0), scaled_col_sums_d(N, 0.0);
    for (int i = 0; i < nnz; ++i) {
        raw_val_d[i] *= row_scale_d[raw_row[i]] * col_scale_d[raw_col[i]];
        scaled_row_sums_d[raw_row[i]] += std::abs(raw_val_d[i]);
        scaled_col_sums_d[raw_col[i]] += std::abs(raw_val_d[i]);
        raw_val[i] = static_cast<real_t>(raw_val_d[i]);
    }
    for (int i = 0; i < M; ++i) b[i] *= static_cast<real_t>(row_scale_d[i]);
    for (int j = 0; j < N; ++j) c[j] *= static_cast<real_t>(col_scale_d[j]);

    col_scale_out.resize(N);
    for(int j=0; j<N; ++j) col_scale_out[j] = static_cast<real_t>(col_scale_d[j]);

    std::vector<int> current_row_head = csr_row_ptr;
    csr_values.resize(nnz); csr_col_idx.resize(nnz);
    for (int i = 0; i < nnz; ++i) {
        int r = raw_row[i]; int dest = current_row_head[r]++;
        csr_values[dest] = raw_val[i]; csr_col_idx[dest] = raw_col[i];
    }

    csc_col_ptr.assign(N + 1, 0); csc_row_idx.resize(nnz); csc_values.resize(nnz);
    for (int i = 0; i < nnz; i++) csc_col_ptr[csr_col_idx[i] + 1]++;
    for (int j = 0; j < N; j++) csc_col_ptr[j + 1] += csc_col_ptr[j];

    std::vector<int> current_col_head = csc_col_ptr;
    for (int i = 0; i < M; i++) {
        for (int k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; k++) {
            int j = csr_col_idx[k]; int dest = current_col_head[j]++;
            csc_values[dest] = csr_values[k]; csc_row_idx[dest] = i;
        }
    }

    tau_vec.resize(N); sigma_vec.resize(M);
    for (int j = 0; j < N; j++) tau_vec[j] = static_cast<real_t>(0.99 / (scaled_col_sums_d[j] > 1e-15 ? scaled_col_sums_d[j] : 1.0));
    for (int i = 0; i < M; i++) sigma_vec[i] = static_cast<real_t>(0.99 / (scaled_row_sums_d[i] > 1e-15 ? scaled_row_sums_d[i] : 1.0));

    real_t damping = (real_t)0.9;
    for (real_t& t : tau_vec) t *= damping;
    for (real_t& s : sigma_vec) s *= damping;
}

/* UPDATED: JSON helpers -- write a double as a JSON number, falling back
 * to null for NaN/Inf (which are not valid JSON) so a DIVERGED result
 * still produces a well-formed, frontend-parseable file. */
static std::string json_num(double v) {
    if (std::isnan(v) || std::isinf(v)) return "null";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.10g", v);
    return std::string(buf);
}

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char ch : s) {
        if (ch == '"' || ch == '\\') out += '\\';
        out += ch;
    }
    return out;
}

int main(int argc, char** argv) {
    // UPDATED: now requires an output JSON path as the second argument --
    // gpu_solver.exe input.txt output.json
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.txt> <output.json>\n";
        return EXIT_FAILURE;
    }
    const char* filename = argv[1];
    const char* out_json_path = argv[2];

    int M = 0, N = 0, nnz = 0;
    std::vector<real_t> c, b, csr_values, csc_values;
    std::vector<int> csr_col_idx, csr_row_ptr, csc_row_idx, csc_col_ptr;
    std::vector<real_t> tau_vec, sigma_vec;
    std::vector<int> row_type;
    std::vector<real_t> col_scale, c_orig;
    real_t scale_b = 1.0;

    auto t_parse_start = std::chrono::steady_clock::now();
    read_and_compress_matrix(filename, M, N, nnz, c, b, csr_values, csr_col_idx, csr_row_ptr,
                             csc_values, csc_row_idx, csc_col_ptr, tau_vec, sigma_vec, row_type,
                             col_scale, scale_b, c_orig);
    auto t_parse_end = std::chrono::steady_clock::now();
    double parse_time_ms = std::chrono::duration<double, std::milli>(t_parse_end - t_parse_start).count();
    std::cerr << "Parse time: " << (parse_time_ms / 1000.0) << " seconds\n";

    // Lazy context initialization warmup
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

    cublasHandle_t cublas_handle = nullptr;
    cusparseHandle_t cusparse_handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    CUSPARSE_CHECK(cusparseSetStream(cusparse_handle, stream));

    real_t *d_csr_values, *d_b, *d_c;
    int *d_csr_col_idx, *d_csr_row_ptr;
    real_t *d_x, *d_y, *d_x_bar;

    CUDA_CHECK(cudaMalloc((void**)&d_csr_values, nnz * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_csr_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_csr_row_ptr, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, M * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_c, N * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, M * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_x_bar, N * sizeof(real_t)));

    CUDA_CHECK(cudaMemcpy(d_csr_values, csr_values.data(), nnz * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_col_idx, csr_col_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_row_ptr, csr_row_ptr.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), M * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, c.data(), N * sizeof(real_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(d_x, 0, N * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(real_t)));
    CUDA_CHECK(cudaMemset(d_x_bar, 0, N * sizeof(real_t)));

    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, M, N, nnz,
                                     d_csr_row_ptr, d_csr_col_idx, d_csr_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_REAL_TYPE));

    real_t* d_csc_values;
    int *d_csc_row_idx, *d_csc_col_ptr;
    CUDA_CHECK(cudaMalloc((void**)&d_csc_values, nnz * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_csc_row_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_csc_col_ptr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_csc_values, csc_values.data(), nnz * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csc_row_idx, csc_row_idx.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csc_col_ptr, csc_col_ptr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t matAT;
    CUSPARSE_CHECK(cusparseCreateCsr(&matAT, N, M, nnz,
                                     d_csc_col_ptr, d_csc_row_idx, d_csc_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_REAL_TYPE));

    cusparseDnVecDescr_t vecX, vecY, vecX_bar;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, N, d_x, CUDA_REAL_TYPE));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, M, d_y, CUDA_REAL_TYPE));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX_bar, N, d_x_bar, CUDA_REAL_TYPE));

    real_t *d_x_new, *d_Ax, *d_ATy;
    real_t *d_primal_violation, *d_dual_violation;

    CUDA_CHECK(cudaMalloc((void**)&d_x_new, N * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_Ax, M * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_ATy, N * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_primal_violation, M * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_dual_violation, N * sizeof(real_t)));

    cusparseDnVecDescr_t vecAx, vecATy;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecAx, M, d_Ax, CUDA_REAL_TYPE));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecATy, N, d_ATy, CUDA_REAL_TYPE));

    size_t bufferSizeA = 0, bufferSizeAT = 0;
    void *dBufferA = nullptr, *dBufferAT = nullptr;
    real_t alpha = 1.0, beta = 0.0;

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX_bar, &beta, vecAx, CUDA_REAL_TYPE,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeA));

    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matAT, vecY, &beta, vecATy, CUDA_REAL_TYPE,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeAT));

    CUDA_CHECK(cudaMalloc(&dBufferA, bufferSizeA));
    CUDA_CHECK(cudaMalloc(&dBufferAT, bufferSizeAT));

#if CUDART_VERSION >= 12040
    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX_bar, &beta, vecAx, CUDA_REAL_TYPE,
        CUSPARSE_SPMV_ALG_DEFAULT, dBufferA));

    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matAT, vecY, &beta, vecATy, CUDA_REAL_TYPE,
        CUSPARSE_SPMV_ALG_DEFAULT, dBufferAT));
#endif

    real_t *d_tau, *d_sigma;
    int *d_row_type;
    CUDA_CHECK(cudaMalloc((void**)&d_tau, N * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_sigma, M * sizeof(real_t)));
    CUDA_CHECK(cudaMalloc((void**)&d_row_type, M * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_tau, tau_vec.data(), N * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sigma, sigma_vec.data(), M * sizeof(real_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_type, row_type.data(), M * sizeof(int), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksM = (M + threadsPerBlock - 1) / threadsPerBlock;
    int blocksN = (N + threadsPerBlock - 1) / threadsPerBlock;

    int ITERS_PER_GRAPH;
    int DIAG_INTERVAL;
    int MAX_ITER;

    if (nnz < 100000) {
        MAX_ITER = 50000;
        DIAG_INTERVAL = 400;
        ITERS_PER_GRAPH = 200;
    } else if (nnz <= 1000000) {
        MAX_ITER = 120000;
        DIAG_INTERVAL = 1000;
        ITERS_PER_GRAPH = 500;
    } else {
        MAX_ITER = 200000;
        DIAG_INTERVAL = 2000;
        ITERS_PER_GRAPH = 1000;
    }
    int GRAPHS_PER_DIAG = DIAG_INTERVAL / ITERS_PER_GRAPH;
    const int RESTART_INTERVAL = 200; // Decoupled fixed restart cadence

    // ---------------------------------------------------------------------
    // CUDA GRAPH CAPTURE WITH DECOUPLED INTERNAL RESTART NODES
    // ---------------------------------------------------------------------
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int k = 0; k < ITERS_PER_GRAPH; ++k) {
        CUSPARSE_CHECK(cusparseSpMV(
            cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX_bar, &beta, vecAx, CUDA_REAL_TYPE,
            CUSPARSE_SPMV_ALG_DEFAULT, dBufferA));
        update_dual_kernel<<<blocksM, threadsPerBlock, 0, stream>>>(d_y, d_Ax, d_b, d_sigma, d_row_type, M);

        CUSPARSE_CHECK(cusparseSpMV(
            cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matAT, vecY, &beta, vecATy, CUDA_REAL_TYPE,
            CUSPARSE_SPMV_ALG_DEFAULT, dBufferAT));
        update_primal_kernel<<<blocksN, threadsPerBlock, 0, stream>>>(d_x, d_x_new, d_ATy, d_c, d_tau, N);

        extrapolate_kernel<<<blocksN, threadsPerBlock, 0, stream>>>(d_x, d_x_new, d_x_bar, N);

        // Internal restart embedded safely at capture time (every 200 iterations)
        if ((k + 1) % RESTART_INTERVAL == 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_x_bar, d_x, N * sizeof(real_t), cudaMemcpyDeviceToDevice, stream));
        }
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    double norm_b = 0.0, norm_c = 0.0;
    for (real_t val : b) norm_b += (double)val * (double)val;
    for (real_t val : c) norm_c += (double)val * (double)val;
    norm_b = std::sqrt(norm_b);
    norm_c = std::sqrt(norm_c);
    double primal_denom = 1.0 + norm_b;
    double dual_denom = 1.0 + norm_c;

    double final_gap = 1.0;
    double final_primal_res = 1.0;
    double final_dual_res = 1.0;
    int actual_iterations = MAX_ITER;
    int iter = 0;

    while (iter < MAX_ITER) {
        for (int g = 0; g < GRAPHS_PER_DIAG && iter < MAX_ITER; ++g) {
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
            iter += ITERS_PER_GRAPH;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // --- Double-precision Diagnostics ---
        real_t primal_cost = 0.0, dual_dot = 0.0;
        CUBLAS_CHECK(cublasXdot(cublas_handle, N, d_c, 1, d_x, 1, &primal_cost));
        CUBLAS_CHECK(cublasXdot(cublas_handle, M, d_b, 1, d_y, 1, &dual_dot));

        real_t dual_profit = -dual_dot;
        double gap = std::abs((double)primal_cost - (double)dual_profit) /
                     (1.0 + std::abs((double)primal_cost) + std::abs((double)dual_profit));

        CUSPARSE_CHECK(cusparseSpMV(
            cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecAx, CUDA_REAL_TYPE,
            CUSPARSE_SPMV_ALG_DEFAULT, dBufferA));

        compute_primal_violation<<<blocksM, threadsPerBlock, 0, stream>>>(d_Ax, d_b, d_row_type, d_primal_violation, M);
        compute_dual_violation<<<blocksN, threadsPerBlock, 0, stream>>>(d_ATy, d_c, d_dual_violation, N);

        real_t primal_residual = 0.0, dual_residual = 0.0;
        CUBLAS_CHECK(cublasXnrm2(cublas_handle, M, d_primal_violation, 1, &primal_residual));
        CUBLAS_CHECK(cublasXnrm2(cublas_handle, N, d_dual_violation, 1, &dual_residual));

        double rel_primal_residual = (double)primal_residual / primal_denom;
        double rel_dual_residual = (double)dual_residual / dual_denom;

        final_gap = gap;
        final_primal_res = rel_primal_residual;
        final_dual_res = rel_dual_residual;

        std::cerr << "iter=" << iter << " gap=" << gap
                  << " primal_res=" << rel_primal_residual
                  << " dual_res=" << rel_dual_residual << "\n";

        if (std::isnan(gap) || std::isinf((double)primal_cost) || std::isinf((double)dual_profit) ||
            std::isnan(rel_primal_residual) || std::isinf(rel_primal_residual) ||
            std::isnan(rel_dual_residual) || std::isinf(rel_dual_residual)) {
            actual_iterations = iter;
            break;
        }

        if (gap < 1e-3 && rel_primal_residual < 1e-3 && rel_dual_residual < 1e-3) {
            actual_iterations = iter;
            break;
        }
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_time_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start, stop));

    std::vector<real_t> h_x(N);
    CUDA_CHECK(cudaMemcpy(h_x.data(), d_x, N * sizeof(real_t), cudaMemcpyDeviceToHost));

    double objective_value = 0.0;
    for (int j = 0; j < N; j++) {
        h_x[j] = h_x[j] * col_scale[j] / scale_b;
        objective_value += (double)c_orig[j] * (double)h_x[j];
    }

    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUSPARSE_CHECK(cusparseDestroy(cusparse_handle));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::string status;
    if (std::isnan(final_gap) || std::isinf(final_gap) ||
        std::isnan(final_primal_res) || std::isinf(final_primal_res)) {
        status = "DIVERGED";
    } else if (actual_iterations < MAX_ITER) {
        // UPDATED: the frontend's run_real_solver_pipeline() checks
        // gpu_json.get("status") == "OPTIMAL" verbatim to set
        // SolveResult.converged -- must match that string exactly.
        status = "OPTIMAL";
    } else {
        status = "MAX_ITER_REACHED";
    }

    // Diagnostic line kept on stdout for logs/debugging -- the JSON file
    // written below is the actual machine-readable contract with the
    // frontend.
    std::cout.precision(10);
    std::cout << status << "," << std::fixed << objective_value << "," << actual_iterations << ","
              << kernel_time_ms << "," << final_gap << ","
              << final_primal_res << "," << final_dual_res << "\n";

    /* ---------------------------------------------------------------
     * UPDATED: write the JSON result file the frontend consumes,
     * replacing the old plain-text solution_x.txt dump. Field names
     * below match frontend_cuda_crusaders_v2.py's run_real_solver_pipeline()
     * EXACTLY (it reads: status, objective_value, iterations,
     * performance.gpu_compute_time_ms, primal_residual, dual_residual) --
     * see that function's gpu_json.get(...) calls:
     *   status                     "OPTIMAL" | "MAX_ITER_REACHED" | "DIVERGED"
     *                              (frontend checks == "OPTIMAL" for converged)
     *   objective_value            objective at x, MINIMIZE convention
     *                              (frontend applies was_maximize / objective_constant
     *                              correction itself from the converter's .meta.json,
     *                              same as it does for the CPU solver path)
     *   iterations                 PDHG iterations actually run
     *   performance.gpu_compute_time_ms   GPU kernel time (CUDA event based) --
     *                              frontend reads this nested field specifically
     *   performance.parse_time_ms  input-file parse + preconditioning time (extra,
     *                              not read by the frontend but harmless to include)
     *   gap, primal_residual, dual_residual   final convergence diagnostics
     *   n_row, n_col               problem dimensions (post gpu_format.py canonicalization)
     *   x                          solution vector, length n_col (frontend currently
     *                              ignores this and reports x as zeros -- see its
     *                              comment "the compiled binary reports the
     *                              objective/residuals, not x" -- included anyway
     *                              since it's harmless and lets a future frontend
     *                              change pick it up for free)
     * --------------------------------------------------------------- */
    std::ofstream out_json(out_json_path);
    if (!out_json) {
        std::cerr << "Failed to open output JSON file: " << out_json_path << "\n";
        return EXIT_FAILURE;
    }
    out_json << "{\n";
    out_json << "  \"status\": \"" << json_escape(status) << "\",\n";
    out_json << "  \"objective_value\": " << json_num(objective_value) << ",\n";
    out_json << "  \"iterations\": " << actual_iterations << ",\n";
    out_json << "  \"performance\": {\n";
    out_json << "    \"gpu_compute_time_ms\": " << json_num((double)kernel_time_ms) << ",\n";
    out_json << "    \"parse_time_ms\": " << json_num(parse_time_ms) << "\n";
    out_json << "  },\n";
    out_json << "  \"gap\": " << json_num(final_gap) << ",\n";
    out_json << "  \"primal_residual\": " << json_num(final_primal_res) << ",\n";
    out_json << "  \"dual_residual\": " << json_num(final_dual_res) << ",\n";
    out_json << "  \"n_row\": " << M << ",\n";
    out_json << "  \"n_col\": " << N << ",\n";
    out_json << "  \"x\": [";
    for (int j = 0; j < N; j++) {
        if (j > 0) out_json << ", ";
        out_json << json_num((double)h_x[j]);
    }
    out_json << "]\n";
    out_json << "}\n";
    out_json.close();

    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroySpMat(matAT));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX_bar));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecAx));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecATy));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));

    cudaFree(d_csr_values); cudaFree(d_csr_col_idx); cudaFree(d_csr_row_ptr);
    cudaFree(d_csc_values); cudaFree(d_csc_row_idx); cudaFree(d_csc_col_ptr);
    cudaFree(d_b); cudaFree(d_c); cudaFree(d_x); cudaFree(d_y); cudaFree(d_x_bar);
    cudaFree(dBufferA); cudaFree(dBufferAT); cudaFree(d_tau); cudaFree(d_sigma); cudaFree(d_row_type);
    cudaFree(d_x_new); cudaFree(d_Ax); cudaFree(d_ATy);
    cudaFree(d_primal_violation); cudaFree(d_dual_violation);

    return 0;
}
