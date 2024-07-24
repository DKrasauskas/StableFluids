#pragma once

struct CSR {
    float* data;
    int* row, * column;
    int n;
};

__global__ void matrix_vector_product(float* result, CSR* matrix, float* vec) {
    uint id = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;
    for (int i = matrix->row[id]; i < matrix->row[id + 1]; i++) {
        sum += matrix->data[i] * vec[matrix->column[i]];
    }
    result[id] = sum;
}
__global__ void vector_vector_product(float* res, float* vec_a, float* vec_b, float split) {
    uint id = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;
    for (int i = id * split; i < (id + 1) * split; i++) {
        sum += vec_a[i] * vec_b[i];
    }
    atomicAdd(res, sum);
}
__global__ void vector_vector_add(float* res, float* vec_a, float* vec_b, float constant) {
    uint id = threadIdx.x + blockIdx.x * blockDim.x;
    res[id] = vec_a[id] + constant * vec_b[id];
}
__global__ void vector_vector_copy(float* res, float* vec_a) {
    uint id = threadIdx.x + blockIdx.x * blockDim.x;
    res[id] = vec_a[id];
}

//TODO -> gpu only
//__global__ void conjugateGradient(CSR* matrix, float* d, float* q, float* r, float* x, float* b, int ndim) {
//    float alpha, beta, del0, d_dot_q;
//    float del_new = 0;
//    float del_old = 0;
//    matrix_vector_product<<<ndim, 1>>>(q, matrix, x);
//    vector_vector_add << <ndim, 1 >> > (d, b, q, -1);
//    vector_vector_copy << <ndim, 1 >> > (r, d);
//    vector_vector_product << <ndim, 1 >> > (&del_new, d, d, 1);
//    del_old = del_new;
//    del0 = del_new;
//    for (int i = 0; i < ndim; i++) {
//        if (del_new <= 0.001f * del0) break;
//        matrix_vector_product <<<ndim, 1 >>> (q, matrix, x);
//        d_dot_q = 0;
//        vector_vector_product <<<ndim, 1 >>> (&d_dot_q, d, q, 1);
//        alpha = del_new / d_dot_q;
//        vector_vector_add <<<ndim, 1 >>> (x, x, d, alpha);
//        vector_vector_add <<<ndim, 1 >>> (r, r, q, -1 *alpha);
//        del_old = del_new;
//        vector_vector_product <<<ndim, 1 >>> (&del_new, r, r, 1);
//        beta = del_new / del_old;
//        vector_vector_add <<<ndim, 1 >> > (d, r, d,  beta);
//    }
//}

void dot(float* res, float** vec_a, float** vec_b, int n_dim) {
    int split = 100;
    int block = 100;
    vector_vector_product << <block, 1 >> > (res, (*vec_a), (*vec_b), 100);
    cudaDeviceSynchronize();
}
// result = vec_a + constant * vec_b
void add(float** res, float** vec_a, float** vec_b, int n_dim, float constant = 1) {
    int grid_dim = n_dim > 1000 ? 100 : 1;
    vector_vector_add << <n_dim , 1 >> > ((*res), (*vec_a), (*vec_b), constant);
    cudaDeviceSynchronize();
}
void matrix_vector(float** res, CSR* matrix, float** vec_b, int n_dim) {
    int grid_dim = n_dim > 1000 ? 100 : 1;
    matrix_vector_product << <n_dim ,1 >> > ((*res), matrix, (*vec_b));
    cudaDeviceSynchronize();
}
void debug_csr(CSR* matrix) {
    for (int i = 0; i < matrix->n; i++) {
        float* row = (float*)calloc(matrix->n, sizeof(float) * matrix->n);
        for (int k = matrix->row[i]; k < matrix->row[i + 1]; k++) {
            row[matrix->column[k]] = matrix->data[k];
        }
        for (int k = 0; k < matrix->n; k++) {
            std::cout << row[k] << " ";
        }
        free(row);
        std::cout << "\n";
    }
}
void LinearOperator_laplace(CSR* matrix, int n, float width) {
    cudaMallocManaged(&matrix->data, sizeof(float) * n * n * 5);
    cudaMallocManaged(&matrix->row, sizeof(int) * (n * n + 1));
    cudaMallocManaged(&matrix->column, sizeof(float) * n * n * 5);
    int index = 0;
    for (int i = 0; i < n * n; i++) {
        matrix->row[i] = index;
        if (i >= n && i != n - 1) {
            matrix->column[index] = i - n;
            matrix->data[index] = -1 / (width * width);
            index += 1;
        }
        if (i >= 1 && i % n != 0) {
            matrix->column[index] = i - 1;
            matrix->data[index] = -1 / (width * width);
            index += 1;
        }
        matrix->column[index] = i;
        matrix->data[index] = 4 / (width * width);
        index += 1;
        if (i + n < n * n) {
            matrix->column[index] = i + n;
            matrix->data[index] = -1 / (width * width);
            index += 1;
        }
        if (i + 1 < n * n && (i + 1) % n != 0) {
            matrix->column[index] = i + 1;
            matrix->data[index] = -1 / (width * width);
            index += 1;
        }
    }
    matrix->row[n * n] = index;
    matrix->n = n * n;
}
void LinearOperator_divergence(CSR* matrix, int n, int dof) {
    auto data = (float*)malloc(sizeof(float) * n * n * 4 * dof);
    auto row = (int*)malloc(sizeof(int) * (n * n * dof + 1));
    auto column = (int*)malloc(sizeof(int) * n * n * 4 * dof);
    int index = 0;
    for (int i = 0; i < n * n * dof; i++) {
        row[i] = index;
        if (i >= n * dof + 1) {
            column[index] = i - n * dof - 1;
            data[index] = -1;
            index += 1;
        }
        if (i >= dof) {
            column[index] = i - dof;
            data[index] = -1;
            index += 1;
        }
        if (i + dof * n + 1 < n * n * dof) {
            column[index] = i + dof * n + 1;
            data[index] = 1;
            index += 1;
        }
        if (i + 2 < 2 * n * n) {
            column[index] = i + 2;
            data[index] = 1;
            index += 1;
        }
    }
    row[n * n * dof] = index;
    matrix->row = row;
    matrix->column = column;
    matrix->data = data;
    matrix->n = n * n * dof;
}
void make_csr(float** data, int i, int j, CSR* matrix) {
    int count = 0;
    for (int a = 0; a < i; a++) {
        for (int b = 0; b < j; b++) {
            count += data[a][b] != 0 ? 1 : 0;
        }
    }
    cudaMallocManaged(&matrix->data, sizeof(float) * count);
    cudaMallocManaged(&matrix->column, sizeof(float) * count);
    cudaMallocManaged(&matrix->row, sizeof(float) * (i + 1));
    int index = 0;
    for (int a = 0; a < i; a++) {
        matrix->row[a] = index;
        for (int b = 0; b < j; b++) {
            if (data[a][b] != 0) {
                matrix->data[index] = data[a][b];
                matrix->column[index] = b;
                index += 1;
            }
        }
    }
    matrix->row[i] = index;
}