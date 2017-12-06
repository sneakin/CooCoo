#include <stdio.h>
#include <math.h>

#include "public.h"
#include "buffer.h"
#include "matrix.h"

#ifdef IN_PUBLIC
typedef struct Matrix2D_s
{
  size_t width, height;
  Buffer buffer;
  double **rows;
} *Matrix2D;
#endif

PUBLIC Matrix2D matrix_new(Buffer b, size_t w, size_t h)
{
  if(w * h >= b->length) {
    return NULL;
  }
  
  Matrix2D m = (Matrix2D)malloc(sizeof(Matrix2D_s));
  m->width = w;
  m->height = h;
  m->buffer = b;
  m->rows = (double **)malloc(sizeof(double) * h);
  
  for(size_t i = 0; i < h; i++) {
    m->rows[i] = b->data + (i * w);
  }

  return m;
}

PUBLIC void matrix_free(Matrix2D m)
{
  for(size_t i = 0; i < m->height; i++) {
    free(m->rows[i]);
    m->rows[i] = NULL;
  }

  free(m);
}

PUBLIC Buffer matrix_buffer(Matrix2D m)
{
  return m->buffer;
}
