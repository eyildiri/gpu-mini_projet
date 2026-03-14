#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include <iostream>
#include <math.h>

using DataT = float;

#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__); \
        goto Error; \
    } \
} while (0)

enum Operation
{
    OP_ADD = 0,
    OP_MUL = 1
};

const char* opToString(Operation op)
{
    switch (op) {
    case OP_ADD: return "add";
    case OP_MUL: return "mul";
    default:     return "unknown";
    }
}

// ------------------------------------------------------------
// KERNEL
// Chaque thread traite j éléments voisins.
// k augmente l'intensité de calcul.
// ------------------------------------------------------------
__global__ void vectorOpKernel(DataT* d_c, const DataT* d_a, const DataT* d_b,
    int N, int j, int k, int op)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * j;

    for (int t = 0; t < j; t++) {
        int idx = start + t;
        if (idx >= N) return;

        DataT va = d_a[idx];
        DataT vb = d_b[idx];
        DataT res = va;

        if (op == OP_ADD) {
            for (int iter = 0; iter < k; iter++) {
                res = res + vb;
            }
        }
        else {
            for (int iter = 0; iter < k; iter++) {
                res = res * vb;
            }
        }

        d_c[idx] = res;
    }
}

// ------------------------------------------------------------
// CPU reference
// ------------------------------------------------------------
void vectorOpCPU(DataT* h_c, const DataT* h_a, const DataT* h_b,
    int N, int k, Operation op)
{
    for (int i = 0; i < N; i++) {
        DataT va = h_a[i];
        DataT vb = h_b[i];
        DataT res = va;

        if (op == OP_ADD) {
            for (int iter = 0; iter < k; iter++) {
                res = res + vb;
            }
        }
        else {
            for (int iter = 0; iter < k; iter++) {
                res = res * vb;
            }
        }

        h_c[i] = res;
    }
}

// ------------------------------------------------------------
// Vérification CPU / GPU
// ------------------------------------------------------------
bool checkCuda(const DataT* h_cpu, const DataT* h_gpu, int N)
{
    const float absEps = 1e-5f;
    const float relEps = 1e-4f;

    for (int i = 0; i < N; i++) {
        float ref = h_cpu[i];
        float got = h_gpu[i];
        float diff = fabsf(ref - got);
        float tol = absEps + relEps * fabsf(ref);

        if (diff > tol) {
            printf("Mismatch at %d : cpu=%f gpu=%f diff=%f tol=%f\n",
                i, ref, got, diff, tol);
            return false;
        }
    }

    return true;
}

// ------------------------------------------------------------
// GPU device info
// ------------------------------------------------------------
void printDeviceInfo()
{
    int device = 0;
    cudaDeviceProp prop;

    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed!\n");
        return;
    }

    std::cout << "GPU_NAME," << prop.name << std::endl;
    std::cout << "GPU_CC," << prop.major << "." << prop.minor << std::endl;
    std::cout << "GPU_SM_COUNT," << prop.multiProcessorCount << std::endl;
    std::cout << "GPU_MAX_THREADS_PER_BLOCK," << prop.maxThreadsPerBlock << std::endl;
    std::cout << "GPU_WARP_SIZE," << prop.warpSize << std::endl;
    std::cout << "GPU_GLOBAL_MEM_GB," << (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << std::endl;
    std::cout << "GPU_SHARED_MEM_PER_BLOCK_KB," << (double)prop.sharedMemPerBlock / 1024.0 << std::endl;
    std::cout << "GPU_REGS_PER_BLOCK," << prop.regsPerBlock << std::endl;
}

// ------------------------------------------------------------
// Fonction style template
// - retourne cudaError_t
// - alloue/cop ie/lance/copie retour/nettoie
// - mesure le temps moyen du kernel
// ------------------------------------------------------------
cudaError_t vectorOpWithCuda(DataT* h_c,
    const DataT* h_a,
    const DataT* h_b,
    int N,
    int threadsPerBlock,
    int j,
    int k,
    Operation op,
    int numRuns,
    float* avgKernelTimeMs)
{
    DataT* d_a = 0;
    DataT* d_b = 0;
    DataT* d_c = 0;

    cudaEvent_t start = 0;
    cudaEvent_t stop = 0;

    cudaError_t cudaStatus;

    *avgKernelTimeMs = 0.0f;

    // Choose which GPU to run on
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        goto Error;
    }

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&d_a, N * sizeof(DataT));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_a failed!\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_b, N * sizeof(DataT));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_b failed!\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_c, N * sizeof(DataT));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_c failed!\n");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers
    cudaStatus = cudaMemcpy(d_a, h_a, N * sizeof(DataT), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_a failed!\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(d_b, h_b, N * sizeof(DataT), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_b failed!\n");
        goto Error;
    }

    cudaStatus = cudaEventCreate(&start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate(start) failed!\n");
        goto Error;
    }

    cudaStatus = cudaEventCreate(&stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventCreate(stop) failed!\n");
        goto Error;
    }

    // TP2: nombre de threads logiques si 1 thread traite j éléments
    //int logicalThreads = (N + j - 1) / j;
    //int blocks = (logicalThreads + threadsPerBlock - 1) / threadsPerBlock;
    int logicalThreads = (N + j - 1) / j;
    dim3 block_size((logicalThreads + threadsPerBlock - 1) / threadsPerBlock);
    dim3 thread_size(threadsPerBlock);

    // Warm-up : on lance une fois le kernel pour éviter d'inclure dans le temps mesuré les éventuels overheads de la première exécution (compilation JIT, etc.)
    vectorOpKernel <<<block_size, thread_size>>> (d_c, d_a, d_b, N, j, k, (int)op);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Warm-up launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize after warm-up failed!\n");
        goto Error;
    }

    // Average benchmark
    cudaStatus = cudaEventRecord(start);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord(start) failed!\n");
        goto Error;
    }

    for (int run = 0; run < numRuns; run++) {
        vectorOpKernel <<<block_size, thread_size >>> (d_c, d_a, d_b, N, j, k, (int)op);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
    }

    cudaStatus = cudaEventRecord(stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventRecord(stop) failed!\n");
        goto Error;
    }

    cudaStatus = cudaEventSynchronize(stop);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaEventSynchronize(stop) failed!\n");
        goto Error;
    }

    {
        float totalMs = 0.0f;
        cudaStatus = cudaEventElapsedTime(&totalMs, start, stop);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaEventElapsedTime failed!\n");
            goto Error;
        }

        *avgKernelTimeMs = totalMs / (float)numRuns;
    }

    // Copy output vector from GPU buffer to host memory
    cudaStatus = cudaMemcpy(h_c, d_c, N * sizeof(DataT), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy d_c -> h_c failed!\n");
        goto Error;
    }

Error:
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return cudaStatus;
}

// ------------------------------------------------------------
// Un cas complet : CPU + GPU + vérification + métriques
// ------------------------------------------------------------
bool runOneCase(const char* experiment,
    Operation op,
    int N,
    int threadsPerBlock,
    int j,
    int k,
    int numRuns)
{
    bool ok = false;

    DataT* h_a = nullptr;
    DataT* h_b = nullptr;
    DataT* h_cpu = nullptr;
    DataT* h_gpu = nullptr;

    h_a = new DataT[N];
    h_b = new DataT[N];
    h_cpu = new DataT[N];
    h_gpu = new DataT[N];

    if (!h_a || !h_b || !h_cpu || !h_gpu) {
        fprintf(stderr, "Host allocation failed!\n");
        goto Cleanup;
    }

    for (int i = 0; i < N; i++) {
        h_a[i] = 0.5f + 0.001f * (float)(i % 100);
        h_b[i] = 0.90f + 0.0005f * (float)(i % 50);
        h_cpu[i] = 0.0f;
        h_gpu[i] = 0.0f;
    }

    // ----------------------------
    // CPU benchmark: warm-up + moyenne
    // ----------------------------
    const int cpuRuns = 50;

    // Warm-up CPU
    vectorOpCPU(h_cpu, h_a, h_b, N, k, op);

    double totalCpuTimeUs = 0.0;

    for (int run = 0; run < cpuRuns; run++) {
        auto cpuStart = std::chrono::high_resolution_clock::now();
        vectorOpCPU(h_cpu, h_a, h_b, N, k, op);
        auto cpuStop = std::chrono::high_resolution_clock::now();

        totalCpuTimeUs +=
            (double)std::chrono::duration_cast<std::chrono::microseconds>(cpuStop - cpuStart).count();
    }

    double cpuTimeUs = totalCpuTimeUs / (double)cpuRuns;

    // On garde le dernier résultat CPU dans h_cpu pour la vérification
    vectorOpCPU(h_cpu, h_a, h_b, N, k, op);

    // ----------------------------
    // GPU benchmark
    // ----------------------------
    {
        float avgKernelTimeMs = 0.0f;

        cudaError_t cudaStatus = vectorOpWithCuda(
            h_gpu, h_a, h_b, N, threadsPerBlock, j, k, op, numRuns, &avgKernelTimeMs
        );

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "vectorOpWithCuda failed!\n");
            goto Cleanup;
        }

        ok = checkCuda(h_cpu, h_gpu, N);

        double gpuTimeUs = (double)avgKernelTimeMs * 1000.0;
        double speedup = (gpuTimeUs > 0.0) ? (cpuTimeUs / gpuTimeUs) : 0.0;

        // 2 lectures + 1 écriture par élément
        double bytesMoved = 3.0 * (double)N * sizeof(DataT);
        double memoryThroughputGBs = bytesMoved / (double)avgKernelTimeMs / 1e6;

        // k opérations par élément
        double operations = (double)N * (double)k;
        double computeThroughputGOPS = operations / (double)avgKernelTimeMs / 1e6;

        double computeIntensity = operations / bytesMoved;

        std::cout
            << experiment << ","
            << opToString(op) << ","
            << "float" << ","
            << N << ","
            << threadsPerBlock << ","
            << j << ","
            << k << ","
            << cpuTimeUs << ","
            << gpuTimeUs << ","
            << speedup << ","
            << memoryThroughputGBs << ","
            << computeThroughputGOPS << ","
            << computeIntensity << ","
            << (ok ? 1 : 0)
            << std::endl;
    }

Cleanup:
    delete[] h_a;
    delete[] h_b;
    delete[] h_cpu;
    delete[] h_gpu;

    return ok;
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main()
{
    // IMPORTANT : utiliser Release x64
    // CSV columns:
    // experiment,operation,datatype,N,threads_per_block,j,k,cpu_us,gpu_us,speedup,memory_GBs,compute_GOPS,compute_intensity_OPS_per_byte,correct

    printDeviceInfo();

    std::cout
        << "experiment,operation,datatype,N,threads_per_block,j,k,"
        << "cpu_us,gpu_us,speedup,memory_GBs,compute_GOPS,compute_intensity_OPS_per_byte,correct"
        << std::endl;

    const int numRuns = 100;
    const int threadsPerBlock = 256;

    // 1) Performance vs N
    const int N_values[] = {
        1 << 10,
        1 << 12,
        1 << 14,
        1 << 16,
        1 << 18,
        1 << 20,
        1 << 22,
        1 << 23
    };

    for (int opi = 0; opi < 2; opi++) {
        Operation op = (opi == 0) ? OP_ADD : OP_MUL;

        for (int i = 0; i < (int)(sizeof(N_values) / sizeof(N_values[0])); i++) {
            runOneCase("vsN", op, N_values[i], threadsPerBlock, 1, 1, numRuns);
        }
    }

    // 2) Performance vs j
    const int fixedN_for_j = 1 << 22;
    const int j_values[] = { 1, 2, 4, 8, 16 };

    for (int opi = 0; opi < 2; opi++) {
        Operation op = (opi == 0) ? OP_ADD : OP_MUL;

        for (int i = 0; i < (int)(sizeof(j_values) / sizeof(j_values[0])); i++) {
            runOneCase("vsJ", op, fixedN_for_j, threadsPerBlock, j_values[i], 1, numRuns);
        }
    }

    // 3) Roofline / intensité vs k
    const int fixedN_for_k = 1 << 22;
    const int fixedJ_for_k = 1;
    const int k_values[] = { 1, 2, 4, 8, 16, 32, 64 };

    for (int opi = 0; opi < 2; opi++) {
        Operation op = (opi == 0) ? OP_ADD : OP_MUL;

        for (int i = 0; i < (int)(sizeof(k_values) / sizeof(k_values[0])); i++) {
            runOneCase("roofline", op, fixedN_for_k, threadsPerBlock, fixedJ_for_k, k_values[i], numRuns);
        }
    }

    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        return 1;
    }

    return 0;
}