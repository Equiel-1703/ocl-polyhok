#include <stdio.h>
#include <time.h>
#include <chrono>

__device__ float anon_ajh07a72e0(float *mat1, float *mat2, int m, int x, int y)
{
    // Fix: this was an int started at 0, I replaced it with float started at 0.0
    float sum = 0.0;
    for (int i = 0; i < m; i += 1)
    {
        sum = (sum + (mat1[((x * m) + i)] * mat2[((i * m) + y)]));
    }

    return (sum);
}

extern "C" __global__ void map2xy2D_kernel(float *arr1, float *arr2, int par, float *resp, int size)
{
    int row = ((blockIdx.y * blockDim.y) + threadIdx.y);
    int col = ((blockIdx.x * blockDim.x) + threadIdx.x);
    if (((col < size) && (row < size)))
    {
        resp[((row * size) + col)] = anon_ajh07a72e0(arr1, arr2, par, row, col);
    }
}

int main(int argc, char const *argv[])
{

    int value = atoi(argv[1]);

    int m = value;

    cudaError_t j_error;

    float *a = (float *)malloc(m * m * sizeof(float));
    float *b = (float *)malloc(m * m * sizeof(float));

    srand(time(0));

    // Fix: indexes were going out of bounds and starting at 1 instead of 0
    for (int i = 0; i < m * m; ++i)
    {
        a[i] = rand() % 1000;
    }

    for (int i = 0; i < m * m; ++i)
    {
        b[i] = rand() % 1000;
    }

    float *d_a, *d_b, *d_c;

    int block_size = 16;
    int grid_rows = (m + block_size - 1) / block_size;
    int grid_cols = (m + block_size - 1) / block_size;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);

    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    auto chrono_start = std::chrono::high_resolution_clock::now();

    cudaMalloc((void **)&d_a, sizeof(float) * m * m);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 1: %s\n", cudaGetErrorString(j_error));
    cudaMalloc((void **)&d_b, sizeof(float) * m * m);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 2: %s\n", cudaGetErrorString(j_error));
    cudaMalloc((void **)&d_c, sizeof(float) * m * m);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 3: %s\n", cudaGetErrorString(j_error));

    auto h2d_start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(d_a, a, sizeof(float) * m * m, cudaMemcpyHostToDevice);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 4: %s\n", cudaGetErrorString(j_error));
    cudaMemcpy(d_b, b, sizeof(float) * m * m, cudaMemcpyHostToDevice);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 5: %s\n", cudaGetErrorString(j_error));

    auto h2d_end = std::chrono::high_resolution_clock::now();

    map2xy2D_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, m, d_c, m);

    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 6: %s\n", cudaGetErrorString(j_error));

    cudaDeviceSynchronize();

    auto kernel_end = std::chrono::high_resolution_clock::now();

    // Copiando resultados pra A, já que não precisamos mais das matrizes originais
    cudaMemcpy(a, d_c, sizeof(float) * m * m, cudaMemcpyDeviceToHost);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 7: %s\n", cudaGetErrorString(j_error));

    auto chrono_end = std::chrono::high_resolution_clock::now();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::chrono::duration<double, std::milli> chrono_total = chrono_end - chrono_start;
    std::chrono::duration<double, std::milli> chrono_h2d = h2d_end - h2d_start;
    std::chrono::duration<double, std::milli> chrono_kernel = kernel_end - h2d_end;
    std::chrono::duration<double, std::milli> chrono_d2h = chrono_end - kernel_end;

    printf("cuda\t%d\t%3.1f\n", m, time);
    printf("-------------------------\n");
    printf("Threads per block: %d x %d\n", block_size, block_size);
    printf("Grid size: %d x %d\n", grid_cols, grid_rows);
    printf("-------------------------\n");
    printf("Total time (cuda events): %3.5f ms\n", time);
    printf("Total time (chrono): %3.5f ms\n", chrono_total.count());
    printf("H2D time (chrono): %3.5f ms\n", chrono_h2d.count());
    printf("Kernel time (chrono): %3.5f ms\n", chrono_kernel.count());
    printf("D2H time (chrono): %3.5f ms\n", chrono_d2h.count());

    free(a);
    free(b);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
