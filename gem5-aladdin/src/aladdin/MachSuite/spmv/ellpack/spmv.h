/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include <stdlib.h>
#include <stdio.h>
#include "support.h"

// These constants valid for the IEEE 494 bus interconnect matrix
#define NNZ 1666
#define N 494
#define L 10

#define TYPE double

void ellpack(TYPE* host_nzval,
             int32_t* host_cols,
             TYPE* host_vec,
             TYPE* host_out,
             TYPE* nzval,
             int32_t* cols,
             TYPE* vec,
             TYPE* out);
////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  TYPE nzval[N*L];
  int32_t cols[N*L];
  TYPE vec[N];
  TYPE out[N];
};
