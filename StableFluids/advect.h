#pragma once
#define uint unsigned int
#include "Mesh.h"

__global__ void advect(Mesh* read, Mesh* write, float dt) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x + 1;
    uint idy = threadIdx.y + blockIdx.y * blockDim.y + 1;
    float x = read->coord[idx] - read->u_x[idx + idy * read->n] * dt;
    float y = read->coord[idy] - read->u_y[idx + idy * read->n] * dt;
    /*
    |lx, ly ------ rx, ly|     
    |                    |
    |                    |
    |lx, ry ------ rx, ry|
    */
    uint lx = floor(x / read->spacing);
    uint ly = floor(y / read->spacing);
    uint rx = ceil(x / read->spacing);
    uint ry = ceil(y / read->spacing);
    // clip to the domain
    lx = lx >= 0 ? (lx < read->n ? lx : read->n - 1) : 0;
    rx = rx >= 0 ? (rx < read->n ? rx : read->n - 1) : 0;
    ly = ly >= 0 ? (ly < read->n ? ly : read->n - 1) : 0;
    ry = ry >= 0 ? (ry < read->n ? ry : read->n - 1) : 0;
    //interpolate first along x
    float rxlx = (rx - lx);
    float ryly = (ry - ly);
    float f1 = rxlx == 0 ? 1 : (x / read->spacing - lx) / (rx - lx);
    float f2 = rxlx == 0 ? 1 : (rx - x / read->spacing) / (rx - lx);
    float f3 = ryly == 0 ? 1 : (y / read->spacing - ly) / (ry - ly);
    float f4 = ryly == 0 ? 1 : (ry - y / read->spacing) / (ry - ly);
    write->u_x[idx + idy * read->n] = f4* (f2 * read->u_x[lx + ly * read->n] + f1 * read->u_x[rx + ly * read->n]) + f3 * (f2 * read->u_x[lx + ry * read->n] + f1 * read->u_x[rx + ry * read->n]);
    write->u_y[idx + idy * read->n] =  f4* (f2 * read->u_y[lx + ly * read->n] + f1 * read->u_y[rx + ly * read->n]) + f3 * (f2 * read->u_y[lx + ry * read->n] + f1 * read->u_y[rx + ry * read->n]);
}
__global__ void force(Mesh* read, float f, float dt) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x + 1;
    uint idy = threadIdx.y + blockIdx.y * blockDim.y + 1;
    read->u_y[idx + idy * (read->n)] = read->u_y[idx + (read->n) * idy] + f * dt;
   // read->u_x[idx + idy * (read->n)] = read->u_x[idx + (read->n) * idy] + f * dt;
}

__global__ void divergence(Mesh* read, float* write, float spacing) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x + 1; // starts at n_dim + 1 (inner boundary point)
    uint idy = threadIdx.y + blockIdx.y * blockDim.y + 1;
    float par_x = read->u_x[idx + 1 + idy * read->n] - read->u_x[idx - 1 + idy * read->n];
    float par_y = read->u_y[idx  + (idy + 1) * read->n] - read->u_y[idx + (idy - 1) * read->n];
    write[idx - 1 + (idy - 1) * (read->n - 2)] = 1.0f * ( par_y + par_x ) / (2 * spacing);
}
__global__ void grad(Mesh* read, float spacing) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x + 1; // starts at n_dim + 1 (inner boundary point)
    uint idy = threadIdx.y + blockIdx.y * blockDim.y + 1;
    float par_x = idx > 1 ? (idx < read->n - 2 ? read->p[idx + (read->n - 2) * (idy - 1)] - read->p[idx - 2 + (read->n - 2) * (idy - 1)] : 0 - read->p[idx - 2 + (read->n - 2) * (idy - 1)]) : read->p[idx + (read->n - 2) * (idy - 1)];
    float par_y = (idy > 1) ? (idy < read->n - 2 ? read->p[idx - 1 + (read->n - 2) * (idy)] - read->p[idx - 1 + (read->n - 2) * (idy - 2)] : 0 - read->p[idx - 1 + (read->n - 2) * (idy - 2)]) : read->p[idx - 1 + (read->n - 2) * (idy)];
    read->grad_x[idx + idy * read->n] = par_x/ (2 * spacing);
    read->grad_y[idx + idy * read->n] = par_y/ (2 * spacing);
}
__global__ void curl(Mesh* read, float* write, float spacing) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x + 1; // starts at n_dim + 1 (inner boundary point)
    uint idy = threadIdx.y + blockIdx.y * blockDim.y + 1;
    float par_x = idx > 1 ? (idx < read->n - 2 ? read->p[idx + (read->n - 2) * (idy - 1)] - read->p[idx - 2 + (read->n - 2) * (idy - 1)] : 0 - read->p[idx - 2 + (read->n - 2) * (idy - 1)]) : read->p[idx + (read->n - 2) * (idy - 1)];
    float par_y = (idy > 1) ? (idy < read->n - 2 ? read->p[idx - 1 + (read->n - 2) * (idy)] - read->p[idx - 1 + (read->n - 2) * (idy - 2)] : 0 - read->p[idx - 1 + (read->n - 2) * (idy - 2)]) : read->p[idx - 1 + (read->n - 2) * (idy)];
    write[idx - 1 + (idy - 1) * (read->n - 2)] = ( par_x - par_y) / (2 * spacing);
}