#pragma once
/* Conjugate Gradient */

/**
  r => b - Ax
  d = r
  del_new = <r, r>
  del_0 = del_new
  do:
  q = Ad
  alpha = del_new / <d, q>
  r = r - alpha * q
  del_old = del_new
  del_new = <r, r>
  beta = del_new / del_old
  d = r + beta * d
  => next iteration
 **/

 /*implementation*/
 /**
  compute per iteration:
  q = Ad
  <d, q>
  =>
  r - alpha * q
  <r, r>
  d = r + beta * d
  totaling:
  1 matrix-vector product
  2 dot products
  3 vector addition/subtraction operations
  **/

#include <iomanip>
struct ConjugateGrad {
    float* q, * d, * r, * x;
    float alpha, beta, del_new, del_old, d_dot_q;
    int n_dim;
};

__managed__ CSR mat;
__managed__ ConjugateGrad linsys;

void CG(ConjugateGrad* lin_sys, CSR* matrix, float* b){
    matrix_vector(&lin_sys->q, matrix, &lin_sys->x, lin_sys->n_dim); // compute Ax
    add(&lin_sys->d, &b, &lin_sys->q, lin_sys->n_dim, -1); // r = b - Ax
    add(&lin_sys->r, &lin_sys->d, &lin_sys->d, lin_sys->n_dim, 0);
    lin_sys->del_new = 0;
    dot(&lin_sys->del_new, &lin_sys->d, &lin_sys->d, lin_sys->n_dim);
    lin_sys->del_old = lin_sys->del_new;
    //if (lin_sys->del_new < 0.001f) return;
  /*  for (int i = 0; i < lin_sys->n_dim; i++) {
        cout << lin_sys->d[i] << "\n";
    }*/
    float del0 = lin_sys->del_new;
    for (int i = 0; i < lin_sys->n_dim * lin_sys->n_dim; i++) {
        if (lin_sys->del_new <= 0.001f * del0) break;
        matrix_vector(&lin_sys->q, matrix, &lin_sys->d, lin_sys->n_dim);
        lin_sys->d_dot_q = 0;
        dot(&lin_sys->d_dot_q, &lin_sys->d, &lin_sys->q, lin_sys->n_dim);
        lin_sys->alpha = lin_sys->del_new / lin_sys->d_dot_q;
        add(&lin_sys->x, &lin_sys->x, &lin_sys->d, lin_sys->n_dim, lin_sys->alpha);
        add(&lin_sys->r, &lin_sys->r, &lin_sys->q, lin_sys->n_dim, -1 * lin_sys->alpha);
        lin_sys->del_old = lin_sys->del_new;
        lin_sys->del_new = 0;
        dot(&lin_sys->del_new, &lin_sys->r, &lin_sys->r, lin_sys->n_dim);
        lin_sys->beta = lin_sys->del_new / lin_sys->del_old;
        add(&lin_sys->d, &lin_sys->r, &lin_sys->d, lin_sys->n_dim, lin_sys->beta);    
    }
    //matrix_vector(&lin_sys->q, matrix, &lin_sys->x, lin_sys->n_dim);
   // for (int i = 0; i < lin_sys->n_dim; i++) {
   //     std::cout << "|" << std::fixed << std::setprecision(4) << linsys.x[i] << "| |" << b[i] << "|\n";
   // }

}