#include <stdio.h>
#include <time.h>

void cpu_mm(float *h_a, float *h_b, float *h_result, int m, int n, int k)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            // Fix: this was an int started at 0, I replaced it with float started at 0.0
            float tmp = 0.0;
            for (int h = 0; h < n; ++h)
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

void checkElementsAre(float *gpu, float *cpu, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (gpu[i] != cpu[i])
        {
            printf("FAIL: gpu[%d] - %0.0f does not equal cpu = %0.0f\n", i, gpu[i], cpu[i]);
            exit(1);
        }
    }
    printf("SUCCESS! All values computed correctly.\n");
}

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
    float *c = (float *)malloc(m * m * sizeof(float));
    // float *cpu_result = (float*) malloc(m*m*sizeof(float));

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

    cudaMemcpy(d_a, a, sizeof(float) * m * m, cudaMemcpyHostToDevice);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 4: %s\n", cudaGetErrorString(j_error));
    cudaMemcpy(d_b, b, sizeof(float) * m * m, cudaMemcpyHostToDevice);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 5: %s\n", cudaGetErrorString(j_error));

    map2xy2D_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, m, d_c, m);

    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 6: %s\n", cudaGetErrorString(j_error));

    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, sizeof(float) * m * m, cudaMemcpyDeviceToHost);
    j_error = cudaGetLastError();
    if (j_error != cudaSuccess)
        printf("Error 7: %s\n", cudaGetErrorString(j_error));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("cuda\t%d\t%3.1f\n", m, time);

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
