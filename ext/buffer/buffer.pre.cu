#include <stdio.h>
#include <math.h>

#include "public.h"
#include "buffer.h"

#ifdef IN_PUBLIC
typedef struct Buffer_s
{
  double *data;
  size_t length;
} *Buffer;
#endif

__device__ int grid(int ndims)
{
  int ret = blockIdx.x * blockDim.x + threadIdx.x;
  if(ndims == 2) {
    ret += threadIdx.y + blockIdx.y * blockDim.y * blockDim.x;
  } else if(ndims == 3) {
    ret += threadIdx.z + blockIdx.z * blockDim.z * blockDim.y * blockDim.x;
  }

  return ret;
}

static int _initialized = -1;
static int _block_size = 1024;
static dim3 _block_dim(1024, 1024, 64);
static int _max_grid_size = 2147483647;
static dim3 _max_grid_dim(2147483647, 65535, 65535);

//static int _threads_per_block = 1;

PUBLIC int buffer_block_size()
{
  return _block_size;
}

PUBLIC dim3 *buffer_block_dim()
{
  return &_block_dim;
}

PUBLIC void buffer_set_block_size(int bs)
{
  _block_size = bs;
}

PUBLIC int buffer_max_grid_size()
{
  return _max_grid_size;
}

PUBLIC dim3 *buffer_max_grid_dim()
{
  return &_max_grid_dim;
}

PUBLIC void buffer_set_max_grid_size(int gs)
{
  _max_grid_size = gs;
}

static size_t _total_bytes_allocated = 0;

PUBLIC size_t buffer_total_bytes_allocated()
{
  return _total_bytes_allocated;
}

static long long _num_allocated = 0;

PUBLIC long long buffer_num_allocated()
{
  return _num_allocated;
}


typedef void (*kernel_func_t)(int len, double *out, const double *a, const double *b, int grid_offset, void *);

Buffer launch_kerneln(kernel_func_t kernel, int length, const Buffer a, const Buffer b, void *data)
{
  Buffer out;
  int i;

  if(a != NULL) {
    int grid_size = (length + _block_size - 1) / _block_size;
    
    out = buffer_new(length, 0.0);
    if(out == NULL) {
      return NULL;
    }
    
    if(grid_size >= _max_grid_size) {
      for(i = 0; i < (grid_size / _max_grid_size); i++) {
        kernel<<< _max_grid_size, _block_size >>>(length, out->data, a->data, b? b->data : NULL, i * _max_grid_size, data);
      }
    } else {
      kernel<<< grid_size, _block_size >>>(length, out->data, a->data, b? b->data : NULL, 0, data);
    }
    return out;
  } else {
    return NULL;
  }
}

Buffer launch_kernel(kernel_func_t kernel, const Buffer a, const Buffer b, void *data)
{
  if(a != NULL && (b == NULL || a->length == b->length)) {
    return launch_kerneln(kernel, a->length, a, b, data);
  } else {
    return NULL;
  }
}

typedef void (*kerneld_func_t)(int, double *, const double *, double, int);

Buffer launchd_kernel(kerneld_func_t kernel, const Buffer a, double b, size_t offset)
{
  Buffer out;
  int i;

  size_t length = a->length - offset;
  int grid_size = (length + _block_size - 1) / _block_size;

  if(a == NULL) return NULL;

  out = buffer_new(length, 0.0);
  if(out == NULL) return NULL;

  if(grid_size >= _max_grid_size) {
    for(i = 0; i < (grid_size / _max_grid_size); i++) {
      kernel<<< _max_grid_size, _block_size >>>(out->length, out->data, a->data + offset, b, i * _max_grid_size);
    }
  } else {
    kernel<<< grid_size, _block_size >>>(out->length, out->data, a->data + offset, b, 0);
  }
  return out;
}

typedef void (*kernel_2d_func_t)(double *a, size_t aw, size_t ah, const double *b, size_t bw, size_t bh, int x, int y, size_t w, size_t h, size_t grid_offset);

cudaError_t launch_2d_kernel(kernel_2d_func_t kernel, Buffer a, size_t aw, const Buffer b, size_t bw, int x, int y, size_t w, size_t h, void *data)
{
  if((a != NULL && (a->length % aw) == 0)
     && (b != NULL && (b->length % bw) == 0)
     && x < aw && (x + (int)w) >= 0
     && y < (a->length / aw) && (y + (int)h) >= 0) {
    unsigned int i, j;

    //dim3 bsize(_block_dim.x, _block_dim.y);
    dim3 bsize(min((unsigned long)_block_dim.y, w),
               min((unsigned long)_block_dim.y, h));
    dim3 grid_size((w + bsize.x - 1) / bsize.x,
                   (h + bsize.y - 1) / bsize.y);
    
    if(grid_size.x >= _max_grid_dim.x
       || grid_size.y >= _max_grid_dim.y) {
      dim3 max_grid(_max_grid_dim.x, _max_grid_dim.y);
      
      for(j = 0; j < (1 + grid_size.y / _max_grid_dim.y); j++) {
        for(i = 0; i < (1 + grid_size.x / _max_grid_dim.x); i++) {
          kernel<<< max_grid, bsize >>>(a->data, aw, a->length / aw,
                                        b? b->data : NULL, b? bw : 0, b? b->length / bw : 0,
                                        x, y,
                                        w, h,
                                        j * _max_grid_dim.y * aw + i * _max_grid_dim.x);
        }
      }
    } else {
      kernel<<< grid_size, bsize >>>(a->data, aw, a->length / aw,
                                     b? b->data : NULL, b? bw : 0, b? b->length / bw : 0,
                                     x, y,
                                     w, h,
                                     0);
    }
    
    return cudaGetLastError();
  } else {
    return cudaErrorInvalidValue;
  }
}


typedef void (*modkerneld_func_t)(int, double *, double, int);

void launchd_modkernel(modkerneld_func_t kernel, const Buffer a, double b, size_t offset)
{
  int i;

  size_t length = a->length - offset;
  int grid_size = (length + _block_size - 1) / _block_size;

  if(a == NULL) return;

  if(grid_size >= _max_grid_size) {
    for(i = 0; i < (grid_size / _max_grid_size); i++) {
      kernel<<< _max_grid_size, _block_size >>>(length, a->data + offset, b, i * _max_grid_size);
    }
  } else {
    kernel<<< grid_size, _block_size >>>(length, a->data + offset, b, 0);
  }
}


PUBLIC cudaError_t buffer_init(int device)
{
  if(_initialized == -1) {
    cudaDeviceProp props;
    cudaError_t err;

    err = cudaSetDevice(device);
    if(err != 0) {
      return err;
    }

    err = cudaGetDeviceProperties(&props, device);
    if(err == 0) {
      _block_size = props.maxThreadsPerBlock;
      _block_dim = dim3(props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
      _max_grid_size = props.maxGridSize[0];
      _max_grid_dim = dim3(props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
      _initialized = device;

      return cudaSuccess;
    } else {
      return err;
    }
  } else {
    return cudaSuccess;
  }
}

__global__ void buffer_setd_inner(int len, double *a, double b, int grid_offset)
{
  int i = grid_offset + grid(1);
  if(i < len) {
    a[i] = b;
  }
}

PUBLIC cudaError_t buffer_setd(Buffer b, double value, size_t offset, size_t length)
{
  if(value == 0.0) {
    if(offset < b->length) {
      if(offset + length >= b->length) {
        length = b->length - offset;
      }
      return cudaMemset(b->data, 0, length * sizeof(double));
    } else {
      return cudaSuccess;
    }
  } else {
    launchd_modkernel(buffer_setd_inner, b, value, offset);
    return cudaSuccess;
  }
}

PUBLIC Buffer buffer_new(size_t length, double initial_value)
{
  Buffer ptr = (Buffer)malloc(sizeof(Buffer_s));;
  if(buffer_init(0) != 0) {
    return NULL;
  }

  if(ptr != NULL) {
    if(cudaMalloc(&ptr->data, length * sizeof(double)) != cudaSuccess) {
      ptr->data = NULL;
      buffer_free(ptr);
      return NULL;
    }
    
    ptr->length = length;
    if(buffer_setd(ptr, initial_value, 0, length) != cudaSuccess) {
      buffer_free(ptr);
      return NULL;
    }

    _total_bytes_allocated += length * sizeof(double);
    _num_allocated++;
  }
      
  return ptr;
}

PUBLIC cudaError_t buffer_free(Buffer buffer)
{
  if(buffer != NULL) {
    if(buffer->data != NULL) {
      cudaError_t err = cudaFree(buffer->data);
      if(err != cudaSuccess) {
        return err;
      }
      _total_bytes_allocated -= buffer->length * sizeof(double);
      _num_allocated--;
    }
    free(buffer);
  }

  return cudaSuccess;
}

PUBLIC cudaError_t buffer_set(Buffer buffer, Buffer other)
{
  if(buffer == NULL || other == NULL) return cudaErrorUnknown;
  
  size_t length = other->length;
  if(length > buffer->length) {
    length = buffer->length;
  }
  return cudaMemcpy(buffer->data, other->data, length * sizeof(double), cudaMemcpyDeviceToDevice);
}

PUBLIC cudaError_t buffer_setn(Buffer buffer, size_t offset, Buffer other, size_t length)
{
  if(length > other->length) {
    length = other->length;
  }
  if((offset + length) > buffer->length) {
    length = buffer->length - offset;
  }
  return cudaMemcpy(buffer->data + offset, other->data, length * sizeof(double), cudaMemcpyDeviceToDevice);
}

PUBLIC cudaError_t buffer_setvn(Buffer buffer, size_t offset, void *data, size_t length)
{
  if((offset + length) > buffer->length) {
    length = buffer->length - offset;
  }
  return cudaMemcpy(buffer->data + offset, data, length * sizeof(double), cudaMemcpyHostToDevice);
}

PUBLIC cudaError_t buffer_setv(Buffer buffer, void *data, size_t length)
{
  return buffer_setvn(buffer, 0, data, length);
}

PUBLIC cudaError_t buffer_set_element(Buffer buffer, size_t n, double v)
{
  if(buffer != NULL && n < buffer->length) {
    return cudaMemcpy(buffer->data + n, &v, sizeof(double), cudaMemcpyHostToDevice);
  } else {
    return cudaErrorUnknown;
  }
}

PUBLIC double buffer_get_element(Buffer buffer, size_t n)
{
  if(buffer != NULL && n < buffer->length) {
    double out;
    cudaMemcpy(&out, buffer->data + n, sizeof(double), cudaMemcpyDeviceToHost);
    return out;
  } else {
    return NAN;
  }
}

PUBLIC cudaError_t buffer_get(Buffer buffer, void *out, size_t max_length)
{
  if(buffer == NULL) return cudaErrorUnknown;

  if(max_length > buffer->length) {
    max_length = buffer->length;
  }
  
  cudaError_t err = cudaMemcpy(out, buffer->data, max_length * sizeof(double), cudaMemcpyDeviceToHost);

  return err;
}

PUBLIC Buffer buffer_slice(Buffer buffer, size_t n, size_t max_length)
{
  if(buffer != NULL && n < buffer->length) {
    Buffer out = buffer_new(max_length, 0.0);
    if(out == NULL) return NULL;

    if((n + max_length) >= buffer->length) {
      max_length = buffer->length - n;
    }

    cudaError_t err = cudaMemcpy(out->data, buffer->data + n, max_length * sizeof(double), cudaMemcpyDeviceToDevice);

    if(err == cudaSuccess) {
      return out;
    } else {
      buffer_free(out);
      return NULL;
    }
  } else {
    return NULL;
  }
}

PUBLIC cudaError_t buffer_host_slice(Buffer buffer, void *out, size_t n, size_t max_length)
{
  if(buffer != NULL && n < buffer->length) {
    if((n + max_length) >= buffer->length) {
      max_length = buffer->length - n;
    }
    
    cudaError_t err = cudaMemcpy(out, buffer->data + n, max_length * sizeof(double), cudaMemcpyDeviceToHost);
    return err;
  } else {
    return cudaErrorUnknown;
  }
}

PUBLIC size_t buffer_length(const Buffer b)
{
  if(b != NULL) {
    return b->length;
  } else {
    return 0;
  }
}

PUBLIC Buffer buffer_slice_2d(const Buffer in, int width, int height, int x, int y, int out_width, int out_height, double empty)
{
  Buffer out = buffer_new(out_width * out_height, empty);
  if(out == NULL) return NULL;

  if(y >= height || x >= width) {
    return out;
  }
  
  for(int i = 0; i < out_height; i++) {
    int oi = i * out_width;
    int w = out_width;
    int iy = y + i;
    int ii = (iy) * width + x;

    if(x < 0) {
      ii = ii - x;
      w = w + x;
      oi = oi - x;
    } else if(x + w > width) {
      w = width - x;
    }
    
    if(iy < 0) {
      continue;
    } else if(iy >= height || ii >= in->length) {
      break;
    }

    cudaError_t err = cudaMemcpy(out->data + oi, in->data + ii, w * sizeof(double), cudaMemcpyDeviceToDevice);
    if(err != cudaSuccess) {
      buffer_free(out);
      return NULL;
    }
  }

  return out;
}

cudaError_t buffer_set2d_inner(Buffer dest, size_t dest_width, const void *src, size_t src_width, size_t src_height, size_t x, size_t y, cudaMemcpyKind kind)
{
  if(dest == NULL
     || dest->data == NULL
     || dest_width == 0
     || src == NULL
     || src_width == 0
     || src_height == 0) {
    return cudaErrorInvalidValue;
  }
  
  size_t copy_width = src_width;
  if(x + src_width > dest_width) {
    copy_width = dest_width - x;
  }
  size_t dest_height = dest->length / dest_width;
  size_t copy_height = src_height;
  if(y + src_height > dest_height) {
    copy_height = dest_height - y;
  }
  size_t idx = y * dest_width + x;
  if(idx >= dest->length
     || (idx + copy_width * copy_height) > dest->length) {
    return cudaSuccess;
  }
  cudaError_t err = cudaMemcpy2D(dest->data + idx,
                                 dest_width * sizeof(double),
                                 src, src_width * sizeof(double),
                                 copy_width * sizeof(double),
                                 copy_height,
                                 kind);
  return err;
}

PUBLIC cudaError_t buffer_set2d(Buffer dest, size_t dest_width, const Buffer src, size_t src_width, size_t x, size_t y)
{
  if(dest->length == 0
     || src->length == 0
     || src_width == 0
     || src->length % src_width > 0) {
    return cudaErrorInvalidValue;
  }
  return buffer_set2d_inner(dest, dest_width, src->data, src_width, src->length / src_width, x, y, cudaMemcpyDeviceToDevice);
}

PUBLIC cudaError_t buffer_set2dv(Buffer dest, size_t dest_width, const void *src, size_t src_width, size_t src_height, size_t x, size_t y)
{
  return buffer_set2d_inner(dest, dest_width, src, src_width, src_height, x, y, cudaMemcpyHostToDevice);
}

typedef double (*ReduceOp)(double a, double b);
typedef void (*ReduceKernel)(const double *, double *, size_t, size_t);

#define REDUCE_KERNEL(NAME, OP, INITIAL)                                \
  double NAME ## _initial_value = INITIAL;                              \
  double NAME ## _op(double a, double b)                                \
  {                                                                     \
    return OP;                                                          \
  }                                                                     \
                                                                        \
  __global__ void NAME(const double *data, double *partial_sums, size_t length, size_t offset) \
  {                                                                     \
    extern __shared__ double sdata[];                                   \
    int i = offset + grid(1);                                           \
    int n;                                                              \
                                                                        \
    sdata[threadIdx.x] = (i < length)? data[i] : INITIAL;               \
    __syncthreads();                                                    \
                                                                        \
    for(n = blockDim.x / 2; n > 0; n = n / 2) {                         \
      if(threadIdx.x < n) {                                             \
        double a = sdata[threadIdx.x];                                  \
        double b = sdata[threadIdx.x + n];                              \
        sdata[threadIdx.x] = OP;                                        \
      }                                                                 \
      __syncthreads();                                                  \
    }                                                                   \
                                                                        \
    if(threadIdx.x == 0) {                                              \
      partial_sums[blockIdx.x] = sdata[0];                              \
    }                                                                   \
  }

double launch_reduce_inner(ReduceOp op, ReduceKernel reduce_kernel, const Buffer b, double initial)
{
  int i;
  int grid_size = (b->length + _block_size - 1) / _block_size;
  Buffer partial_buffer;
  double *partial_sums;
  double out = initial;
  int usable_grid = grid_size;

  if(b == NULL || b->length == 0) {
    return NAN;
  }
  
  if(grid_size >= _max_grid_size) {
    usable_grid = _max_grid_size;
  }
  
  partial_buffer = buffer_new(_block_size, initial);
  if(partial_buffer == NULL) return NAN;

  for(i = 0; i < grid_size / usable_grid; i++) {
    reduce_kernel<<< usable_grid, _block_size, _block_size * sizeof(double) >>>(b->data, partial_buffer->data, b->length, i * _block_size);
  }

  partial_sums = (double *)malloc(sizeof(double) * _block_size);
  if(partial_sums == NULL) {
    buffer_free(partial_buffer);
    return NAN;
  }
  
  buffer_get(partial_buffer, partial_sums, _block_size);

  out = partial_sums[0];
  for(i = 1; i < _block_size; i++) {
    out = op(out, partial_sums[i]);
  }

  buffer_free(partial_buffer);
  free(partial_sums);
  
  return out;
}

#define launch_reduce(kernel, buffer) launch_reduce_inner(kernel ## _op, kernel, buffer, kernel ## _initial_value)


REDUCE_KERNEL(buffer_sum_kernel, a + b, 0.0);

PUBLIC double buffer_sum(const Buffer b)
{
  return launch_reduce(buffer_sum_kernel, b);
}

REDUCE_KERNEL(buffer_min_kernel, fmin(a, b), NAN);

PUBLIC double buffer_min(const Buffer b)
{
  return launch_reduce(buffer_min_kernel, b);
}

REDUCE_KERNEL(buffer_max_kernel, fmax(a, b), NAN);
PUBLIC double buffer_max(const Buffer b)
{
  return launch_reduce(buffer_max_kernel, b);
}

#define BINARY_OP(name, operation)                                      \
  __device__ inline double buffer_ ## name ## _op(double a, double b)          \
  {                                                                     \
    return(operation);                                                  \
  }                                                                     \
                                                                        \
  __global__ void buffer_ ## name ##_inner(int len, double *out, const double *a, const double *b, int grid_offset, void *) \
  {                                                                     \
    int i = grid_offset + grid(1);                                      \
    if(i < len) {                                                       \
      out[i] = buffer_ ## name ## _op(a[i], b[i]);                      \
    }                                                                   \
  }                                                                     \
                                                                        \
  PUBLIC Buffer buffer_ ## name(const Buffer a, const Buffer b)         \
  {                                                                     \
    return launch_kernel(buffer_ ## name ## _inner, a, b, NULL);        \
  }                                                                     \
                                                                        \
  __global__ void buffer_ ## name ## _2d_inner(double *a, size_t aw, size_t ah, const double *b, size_t bw, size_t bh, int x, int y, size_t w, size_t h, size_t grid_offset) \
  {                                                                     \
    int bx = blockIdx.x * blockDim.x + threadIdx.x;                     \
    int by = blockIdx.y * blockDim.y + threadIdx.y;                     \
    int ax = bx + x;                                                    \
    int ay = by + y;                                                    \
    if(bx >= 0 && bx < w                                                \
       && ax >= 0 && ax < aw                                            \
       && by >= 0 && by < h                                             \
       && ay >= 0 && ax < ah) {                                         \
      int ai = grid_offset + ay * aw + ax;                              \
      int bi = grid_offset + by * bw + bx;                              \
      a[ai] = buffer_ ## name ## _op(a[ai], b[bi]);                     \
    }                                                                   \
  }                                                                     \
                                                                        \
  PUBLIC cudaError_t buffer_ ## name ## _2d(Buffer a, size_t aw, const Buffer b, size_t bw, int x, int y, size_t w, size_t h) \
  {                                                                     \
    return launch_2d_kernel(buffer_ ## name ## _2d_inner, a, aw, b, bw, x, y, w, h, NULL); \
  }                                                                     \
                                                                        \
  __global__ void buffer_ ## name ## d_inner(int len, double *out, const double *a, const double b, int grid_offset) \
  {                                                                     \
    int i = grid_offset + grid(1);                                      \
    if(i < len) {                                                       \
      out[i] = buffer_ ## name ## _op(a[i], b);                         \
    }                                                                   \
  }                                                                     \
                                                                        \
  PUBLIC Buffer buffer_ ## name ## d(const Buffer a, double b)            \
  {                                                                     \
    return launchd_kernel(buffer_ ## name ## d_inner, a, b, 0);         \
  }

BINARY_OP(add, a + b);
BINARY_OP(sub, a - b);
BINARY_OP(mul, a * b);
BINARY_OP(pow, pow(a, b));
BINARY_OP(div, a / b);
BINARY_OP(collect_eq, a == b);
BINARY_OP(collect_neq, a != b);
BINARY_OP(collect_lt, a < b);
BINARY_OP(collect_lte, a <= b);
BINARY_OP(collect_gt, a > b);
BINARY_OP(collect_gte, a >= b);

#undef BINARY_OP

PUBLIC int buffer_eq(const Buffer a, const Buffer b)
{
  // compare
  Buffer results = buffer_collect_eq(a, b);
  if(results != NULL) {
    // reduce
    double sum = buffer_sum(results);
    
    // clean up
    buffer_free(results);
  
    return sum == a->length;
  } else {
    return 0;
  }
}

#define FUNCTION_OP(name, operation) \
  __global__ void buffer_ ## name ## _inner(int len, double *out, const double *a, const double b, int grid_offset) \
  {                                                                     \
    int i = grid_offset + grid(1);                                      \
    if(i < len) {                                                       \
      operation;                                                        \
    }                                                                   \
  }                                                                     \
                                                                        \
  PUBLIC Buffer buffer_##name(const Buffer a)                   \
  {                                                                     \
    return launchd_kernel(buffer_ ## name ## _inner, a, 0.0, 0);         \
  }


FUNCTION_OP(abs, { out[i] = abs(a[i]); });
FUNCTION_OP(exp, { out[i] = exp(a[i]); });
FUNCTION_OP(log, { out[i] = log(a[i]); });
FUNCTION_OP(log10, { out[i] = log10(a[i]); });
FUNCTION_OP(log2, { out[i] = log2(a[i]); });
FUNCTION_OP(sqrt, { out[i] = sqrt(a[i]); });
FUNCTION_OP(floor, { out[i] = floor(a[i]); });
FUNCTION_OP(ceil, { out[i] = ceil(a[i]); });
FUNCTION_OP(round, { out[i] = round(a[i]); });
FUNCTION_OP(sin, { out[i] = sin(a[i]); });
FUNCTION_OP(cos, { out[i] = cos(a[i]); });
FUNCTION_OP(tan, { out[i] = tan(a[i]); });
FUNCTION_OP(asin, { out[i] = asin(a[i]); });
FUNCTION_OP(acos, { out[i] = acos(a[i]); });
FUNCTION_OP(atan, { out[i] = atan(a[i]); });
FUNCTION_OP(sinh, { out[i] = sinh(a[i]); });
FUNCTION_OP(cosh, { out[i] = cosh(a[i]); });
FUNCTION_OP(tanh, { out[i] = tanh(a[i]); });
FUNCTION_OP(asinh, { out[i] = asinh(a[i]); });
FUNCTION_OP(acosh, { out[i] = acosh(a[i]); });
FUNCTION_OP(atanh, { out[i] = atanh(a[i]); });
FUNCTION_OP(collect_nan, { out[i] = isnan(a[i]); });
FUNCTION_OP(collect_inf, { out[i] = isinf(a[i]); });

#undef FUNCTION_OP


__global__ void buffer_dot_inner(double *out, const double *a, const double *b, size_t aw, size_t ah, size_t bw, size_t bh)
{
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  double sum = 0.0;

  if (row < ah && col < bw) {
    for (size_t i = 0; i < bh; i++) {
      sum += a[row * aw + i] * b[i * bw + col];
    }
  }
  out[row * bw + col] = sum;
}

PUBLIC Buffer buffer_dot(const Buffer a, size_t aw, size_t ah, const Buffer b, size_t bw, size_t bh)
{
  if(aw * ah != a->length && bw * bh != b-> length && aw != bh) {
    return NULL;
  } else {
    Buffer out = buffer_new(ah * bw, 0.0);
    if(out == NULL) return NULL;
    
    dim3 dim(bw, ah);
    
    buffer_dot_inner<<< dim, 1 >>>(out->data, a->data, b->data, aw, ah, bw, bh);
    
    return out;
  }
}

__global__ void buffer_identity_inner(double *data, size_t w, size_t h)
{
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < h && col < w) {
    data[row * w + col] = (double)(row == col);
  }
}

PUBLIC Buffer buffer_identity(size_t w, size_t h)
{
  Buffer i = buffer_new(w * h, 0.0);
  dim3 dim(w, h);
  buffer_identity_inner<<< dim, 1 >>>(i->data, w, h);
  return i;
}

__global__ void buffer_diagflat_inner(double *data, const double *a, size_t len)
{
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row >= len || col >= len) {
    return;
  }
  
  if(row == col) {
    data[row * len + col] = a[row];
  } else {
    data[row * len + col] = 0.0;
  }
}

PUBLIC Buffer buffer_diagflat(const Buffer a)
{
  Buffer i = buffer_new(a->length * a->length, 0.0);
  dim3 dim(a->length, a->length);
  buffer_diagflat_inner<<< dim, 1 >>>(i->data, a->data, a->length);
  return i;
}
