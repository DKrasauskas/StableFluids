#pragma once
#include "Mesh.h"
#include "advect.h"
#include "linalg.h"

__managed__ Mesh read, write;

void _init_() {
    float* coords = get_coordinates();
    cudaMallocManaged(&read.coord, DOMAIN_SIZE * sizeof(float));
    cudaMallocManaged(&write.coord, DOMAIN_SIZE * sizeof(float));
    cudaMallocManaged(&read.u_x, DOMAIN_SIZE * DOMAIN_SIZE * sizeof(float));
    cudaMallocManaged(&write.u_x, DOMAIN_SIZE * DOMAIN_SIZE * sizeof(float));
    cudaMallocManaged(&read.u_y, DOMAIN_SIZE * DOMAIN_SIZE * sizeof(float));
    cudaMallocManaged(&write.u_y, DOMAIN_SIZE * DOMAIN_SIZE * sizeof(float));
    cudaMallocManaged(&read.grad_x, (DOMAIN_SIZE) * (DOMAIN_SIZE) * sizeof(float));
    cudaMallocManaged(&write.grad_x, (DOMAIN_SIZE) * (DOMAIN_SIZE) * sizeof(float));
    cudaMallocManaged(&read.grad_y, (DOMAIN_SIZE) * (DOMAIN_SIZE) * sizeof(float));
    cudaMallocManaged(&write.grad_y, (DOMAIN_SIZE) * (DOMAIN_SIZE) * sizeof(float));
    cudaMallocManaged(&write.div, (DOMAIN_SIZE - 2) * (DOMAIN_SIZE - 2) * sizeof(float));
    cudaMallocManaged(&read.div, (DOMAIN_SIZE - 2) * (DOMAIN_SIZE - 2) * sizeof(float));
    cudaMallocManaged(&write.curl, (DOMAIN_SIZE - 2) * (DOMAIN_SIZE - 2) * sizeof(float));
    cudaMallocManaged(&read.curl, (DOMAIN_SIZE - 2) * (DOMAIN_SIZE - 2) * sizeof(float));
    cudaMallocManaged(&write.p, (DOMAIN_SIZE) * (DOMAIN_SIZE) * sizeof(float));
    cudaMallocManaged(&read.p, (DOMAIN_SIZE - 2) * (DOMAIN_SIZE - 2) * sizeof(float));
    memcpy(read.coord, coords, sizeof(float) * DOMAIN_SIZE);
    memcpy(write.coord, coords, sizeof(float) * DOMAIN_SIZE);
    read.n = DOMAIN_SIZE;
    write.n = DOMAIN_SIZE;
    read.spacing = coords[DOMAIN_SIZE];
    write.spacing = read.spacing;
    for (int i = 0; i < read.n; i++) {
        for (int j = 0; j < read.n; j++) {
            read.u_x[j + i * read.n] = 0;
            read.u_y[j + i * read.n] = 0;// (i >= 1 && i < read.n - 1) ? ((j >= 1 && j < read.n - 1) ? 1 : 0) : 0;
            write.u_x[j + i * read.n] = 0;
            write.u_y[j + i * read.n] = 0;
        }
    }
    free(coords);
}
void _terminate_() {
    cudaFree(&read.u_x);
    cudaFree(&read.u_y);
    cudaFree(&write.u_x);
    cudaFree(&write.u_y);
    cudaFree(&write.div);
    cudaFree(&read.div);
    cudaFree(&write.p);
    cudaFree(&read.p);
    cudaFree(&write.curl);
    cudaFree(&read.curl);
    cudaFree(&read.coord);
    cudaFree(&write.coord);
    cudaFree(&write.grad_x);
    cudaFree(&write.grad_y);
    cudaFree(&read.grad_x);
    cudaFree(&read.grad_y);
}

uint block_dim, thread_dim;
#define UNCHARTED ;

void thread_setup(int n) {
    block_dim = n;
    thread_dim = 1;// block_dim == 32 ? n / 32 : 1;
    std::cout << "current setup yields -> " << block_dim * block_dim * thread_dim * thread_dim;
}

void solve(Mesh* read, Mesh* write, float time_step) {

    thread_setup(read->n);
    dim3 fkernel_b(block_dim - block_dim*0.2, 2);
    dim3 fkernel_t(thread_dim, thread_dim);
    dim3 main_kernel_b(block_dim, block_dim);
    dim3 main_kernel_t(thread_dim, thread_dim);
    dim3 main_kernel_bb(block_dim - 2, block_dim - 2);
    LinearOperator_laplace(&mat, (read->n - 2), read->spacing);
    cudaMallocManaged(&linsys.q, sizeof(float) * (read->n - 2) * (read->n - 2));
    cudaMallocManaged(&linsys.r, sizeof(float) * (read->n - 2) * (read->n - 2));
    cudaMallocManaged(&linsys.d, sizeof(float) * (read->n - 2) * (read->n - 2));
    linsys.n_dim = (read->n - 2) * (read->n - 2);

    // main calculations
    for (int j = 0; j < 2; j++) {
        force << <fkernel_t, fkernel_b >> > (read, 10, 0.1f);
        cudaDeviceSynchronize();
        advect << <main_kernel_t, main_kernel_b >> > (read, write, read->spacing);
        cudaDeviceSynchronize();
      /*  for (int i = 0; i < (read->n) * (read->n ); i++) {
            cout << write->u_y[i];
            if ((i + 1) % (read->n) == 0) cout << "\n";
        }*/
        //cout << "\n\n";
         Mesh* ptr = read;
         read = write;
         write = ptr;
         divergence<<<1, main_kernel_bb>>>(read, read->div, 0.0);
         cudaDeviceSynchronize();
         linsys.x = read->p;
         linsys.d = read->div;
         CG(&linsys, &mat, read->div);
         grad << < main_kernel_t, main_kernel_b >>> (read, read->spacing);
         cudaDeviceSynchronize();
         add(&read->u_x, &read->u_x, &read->grad_x, 16 * 16, .1f);
         add(&read->u_y, &read->u_y, &read->grad_y, 16 * 16, -.1f);
         float max, min;
         max = 0;
         min = 0xFFFF;
         for (int i = 0; i < (read->n - 2) * (read->n - 2); i++) {
            // cout << fixed << setprecision(2) << read->p[i];
             if (read->p[i] < min) min = read->p[i];
             if (read->p[i] > max) max = read->p[i];    
            // if ((i + 1) % (read->n) == 0) cout << "\n";
         }
         std::cout << "pmax is " << max << " pmin is " << min << "\n";
        cout << "\n\n";

    }
    UNCHARTED
    /*
    linsys.d = read->div;
	CG(&linsys, &mat);
	grad <<<1, kernel >>> (read);
	cudaDeviceSynchronize();
	add (&read->u_x, &read->u_x, &read->grad_x, read->n, -1.0);
	cudaDeviceSynchronize();
	add (&read->u_y, &read->u_y, &read->grad_y, read->n, - 1.0);
	cudaDeviceSynchronize();
    */

}