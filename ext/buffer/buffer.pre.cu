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
static int _block_size = 256;
static int _max_grid_size = 1024;

//static int _threads_per_block = 1;

PUBLIC int buffer_block_size()
{
  return _block_size;
}

PUBLIC void buffer_set_block_size(int bs)
{
  _block_size = bs;
}

PUBLIC int buffer_max_grid_size()
{
  return _max_grid_size;
}

PUBLIC void buffer_set_max_grid_size(int gs)
{
  _max_grid_size = gs;
}

static size_t _total_bytes_allocated;

PUBLIC size_t buffer_total_bytes_allocated()
{
  return _total_bytes_allocated;
}

typedef void (*kernel_func_t)(int, double *, const double *, const double *, int, void *);

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
      _max_grid_size = props.maxGridSize[0];
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

__global__ void buffer_sum_minor(const double *a, double *partial_sums, size_t length, size_t offset)
{
  extern __shared__ double sdata[];
  int i = offset + grid(1);
  int n;
  
  sdata[threadIdx.x] = (i < length)? a[i] : 0.0;
  __syncthreads();

  for(n = blockDim.x / 2; n > 0; n = n / 2) {
    if(threadIdx.x < n) {
      sdata[threadIdx.x] += sdata[threadIdx.x + n];
    }
    __syncthreads();
  }

  if(threadIdx.x == 0) {
    partial_sums[blockIdx.x] = sdata[0];
  }
}

PUBLIC double buffer_sum(const Buffer b)
{
  int i;
  int grid_size = (b->length + _block_size - 1) / _block_size;
  Buffer partial_buffer;
  double *partial_sums;
  double out = 0.0;
  int usable_grid = grid_size;

  if(b == NULL || b->length == 0) {
    return 0.0;
  }
  
  if(grid_size >= _max_grid_size) {
    usable_grid = _max_grid_size;
  }
  
  partial_buffer = buffer_new(_block_size, 0.0);
  if(partial_buffer == NULL) return 0.0;

  for(i = 0; i < grid_size / usable_grid; i++) {
    buffer_sum_minor<<< usable_grid, _block_size, _block_size * sizeof(double) >>>(b->data, partial_buffer->data, b->length, i * _block_size);
  }

  partial_sums = (double *)malloc(sizeof(double) * _block_size);
  buffer_get(partial_buffer, partial_sums, _block_size);

  for(i = 0; i < _block_size; i++) {
    out += partial_sums[i];
  }

  buffer_free(partial_buffer);
  free(partial_sums);
  
  return out;
}

#define BINARY_OP(name, operation) Buffer buffer_##name(const Buffer, const Buffer); \
    __global__ void buffer_ ## name ## _inner(int len, double *out, const double *a, const double *b, int grid_offset, void *) \
  {                                                                     \
    int i = grid_offset + grid(1);                                      \
    if(i < len) {                                                       \
      operation;                                                        \
    }                                                                   \
  }                                                                     \
                                                                        \
  Buffer buffer_##name(const Buffer a, const Buffer b)                 \
  {                                                                     \
    return launch_kernel(buffer_ ## name ## _inner, a, b, NULL);        \
  }

PUBLIC BINARY_OP(add, { out[i] = a[i] + b[i]; });
PUBLIC BINARY_OP(sub, { out[i] = a[i] - b[i]; });
PUBLIC BINARY_OP(mul, { out[i] = a[i] * b[i]; });
PUBLIC BINARY_OP(div, { out[i] = a[i] / b[i]; });
PUBLIC BINARY_OP(any_eq, { out[i] = a[i] == b[i]; });
PUBLIC BINARY_OP(any_neq, { out[i] = a[i] != b[i]; });
PUBLIC BINARY_OP(any_lt, { out[i] = a[i] < b[i]; });
PUBLIC BINARY_OP(any_lte, { out[i] = a[i] <= b[i]; });
PUBLIC BINARY_OP(any_gt, { out[i] = a[i] > b[i]; });
PUBLIC BINARY_OP(any_gte, { out[i] = a[i] >= b[i]; });

#define SCALAR_OP(name, operation) Buffer buffer_##name ## d(const Buffer, double); \
  __global__ void buffer_ ## name ## d_inner(int len, double *out, const double *a, const double b, int grid_offset) \
  {                                                                     \
    int i = grid_offset + grid(1);                                      \
    if(i < len) {                                                       \
      operation;                                                        \
    }                                                                   \
  }                                                                     \
                                                                        \
  Buffer buffer_##name ## d(const Buffer a, double b)                   \
  {                                                                     \
    return launchd_kernel(buffer_ ## name ## d_inner, a, b, 0);         \
  }

PUBLIC SCALAR_OP(add, { out[i] = a[i] + b; });
PUBLIC SCALAR_OP(sub, { out[i] = a[i] - b; });
PUBLIC SCALAR_OP(mul, { out[i] = a[i] * b; });
PUBLIC SCALAR_OP(div, { out[i] = a[i] / b; });
PUBLIC SCALAR_OP(any_eq, { out[i] = a[i] == b; });
PUBLIC SCALAR_OP(any_neq, { out[i] = a[i] != b; });
PUBLIC SCALAR_OP(any_lt, { out[i] = a[i] < b; });
PUBLIC SCALAR_OP(any_lte, { out[i] = a[i] <= b; });
PUBLIC SCALAR_OP(any_gt, { out[i] = a[i] > b; });
PUBLIC SCALAR_OP(any_gte, { out[i] = a[i] >= b; });

PUBLIC int buffer_eq(const Buffer a, const Buffer b)
{
  // compare
  Buffer results = buffer_any_eq(a, b);
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

#define FUNCTION_OP(name, operation) Buffer buffer_##name(const Buffer); \
  __global__ void buffer_ ## name ## _inner(int len, double *out, const double *a, const double b, int grid_offset) \
  {                                                                     \
    int i = grid_offset + grid(1);                                      \
    if(i < len) {                                                       \
      operation;                                                        \
    }                                                                   \
  }                                                                     \
                                                                        \
  Buffer buffer_##name(const Buffer a)                   \
  {                                                                     \
    return launchd_kernel(buffer_ ## name ## _inner, a, 0.0, 0);         \
  }


PUBLIC FUNCTION_OP(abs, { out[i] = abs(a[i]); });
PUBLIC FUNCTION_OP(exp, { out[i] = exp(a[i]); });
PUBLIC FUNCTION_OP(floor, { out[i] = floor(a[i]); });
PUBLIC FUNCTION_OP(ceil, { out[i] = ceil(a[i]); });
PUBLIC FUNCTION_OP(round, { out[i] = round(a[i]); });
PUBLIC FUNCTION_OP(sin, { out[i] = sin(a[i]); });
PUBLIC FUNCTION_OP(cos, { out[i] = cos(a[i]); });
PUBLIC FUNCTION_OP(tan, { out[i] = tan(a[i]); });
PUBLIC FUNCTION_OP(asin, { out[i] = asin(a[i]); });
PUBLIC FUNCTION_OP(acos, { out[i] = acos(a[i]); });
PUBLIC FUNCTION_OP(atan, { out[i] = atan(a[i]); });
PUBLIC FUNCTION_OP(sinh, { out[i] = sinh(a[i]); });
PUBLIC FUNCTION_OP(cosh, { out[i] = cosh(a[i]); });
PUBLIC FUNCTION_OP(tanh, { out[i] = tanh(a[i]); });
PUBLIC FUNCTION_OP(asinh, { out[i] = asinh(a[i]); });
PUBLIC FUNCTION_OP(acosh, { out[i] = acosh(a[i]); });
PUBLIC FUNCTION_OP(atanh, { out[i] = atanh(a[i]); });

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
