#include <stdio.h>
#include <math.h>

#include "public.h"
#include "buffer.h"

#ifdef IN_PUBLIC
typedef float BufferValue;
typedef struct Buffer_s
{
  size_t length;
  BufferValue *data;
} *Buffer;
#endif

__device__ int grid(int ndims)
{
  int ret = blockIdx.x * blockDim.x + threadIdx.x;
  if(ndims == 2) {
    ret += (threadIdx.y + blockIdx.y * blockDim.y) * blockDim.x;
  } else if(ndims == 3) {
    ret += (threadIdx.z + blockIdx.z * blockDim.z) * blockDim.y * blockDim.x;
  }

  return ret;
}

static int _initialized = -1;
__managed__ static size_t _block_size = 1024;
static dim3 _block_dim(1024, 1024, 64);
__managed__ static size_t _max_grid_size = 2147483647;
static dim3 _max_grid_dim(2147483647, 65535, 65535);

//static int _threads_per_block = 1;

PUBLIC size_t buffer_block_size()
{
  return _block_size;
}

PUBLIC dim3 *buffer_block_dim()
{
  return &_block_dim;
}

PUBLIC void buffer_set_block_size(size_t bs)
{
  _block_size = bs;
}

PUBLIC size_t buffer_max_grid_size()
{
  return _max_grid_size;
}

PUBLIC dim3 *buffer_max_grid_dim()
{
  return &_max_grid_dim;
}

PUBLIC void buffer_set_max_grid_size(size_t gs)
{
  _max_grid_size = gs;
}

static size_t _total_memory = 0;
static size_t _total_bytes_free = 0;
static size_t _total_bytes_allocated = 0;

PUBLIC size_t buffer_total_memory()
{
  return _total_memory;
}

PUBLIC size_t buffer_total_bytes_free()
{
  return _total_bytes_free;
}

PUBLIC size_t buffer_total_bytes_allocated()
{
  return _total_bytes_allocated;
}

static long long _num_allocated = 0;

PUBLIC long long buffer_num_allocated()
{
  return _num_allocated;
}

PUBLIC void buffer_mem_stats_add(size_t amt)
{
  _total_bytes_allocated += amt;
  _total_bytes_free -= amt;
  _num_allocated++;
}

PUBLIC void buffer_mem_stats_freed(size_t amt)
{
  _total_bytes_allocated -= amt;
  _total_bytes_free += amt;
  _num_allocated--;
}

PUBLIC cudaError_t cuda_alloc(void **out, size_t bytes)
{
  cudaError_t err = cudaMalloc(out, bytes);
  if(err == cudaSuccess) {
    buffer_mem_stats_add(bytes);
  }
  return err;
}

PUBLIC cudaError_t cuda_free(void *ptr, size_t bytes)
{
  cudaError_t err = cudaFree(ptr);
  buffer_mem_stats_freed(bytes);
  return err;
}

typedef void (*kernel_func_t)(int len, BufferValue *out, const BufferValue *a, const BufferValue *b, int grid_offset, void *);

Buffer launch_kerneln(kernel_func_t kernel, int length, const Buffer a, const Buffer b, void *data)
{
  Buffer out;
  size_t i;

  if(a != NULL) {
    size_t grid_size = (length + _block_size - 1) / _block_size;
    
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

__device__ void dev_launch_kerneln(kernel_func_t kernel, Buffer out, int length, const Buffer a, const Buffer b, void *data)
{
  size_t i;

  if(a != NULL) {
    size_t grid_size = (length + _block_size - 1) / _block_size;
    
    if(grid_size >= _max_grid_size) {
      for(i = 0; i < (grid_size / _max_grid_size); i++) {
        kernel<<< _max_grid_size, _block_size >>>(length, out->data, a->data, b? b->data : NULL, i * _max_grid_size, data);
      }
    } else {
      kernel<<< grid_size, _block_size >>>(length, out->data, a->data, b? b->data : NULL, 0, data);
    }
    return;
  } else {
    return;
  }
}

__device__ void dev_launch_kernel(kernel_func_t kernel, Buffer out, const Buffer a, const Buffer b, void *data)
{
  if(a != NULL && (b == NULL || a->length == b->length)) {
    dev_launch_kerneln(kernel, out, a->length, a, b, data);
  } else {
    return;
  }
}

typedef void (*kerneld_func_t)(int, BufferValue *, const BufferValue *, BufferValue, int);

Buffer launchd_kernel(kerneld_func_t kernel, const Buffer a, BufferValue b, size_t offset)
{
  Buffer out;
  size_t i;

  size_t length = a->length - offset;
  size_t grid_size = (length + _block_size - 1) / _block_size;

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

typedef void (*kernel_2d_func_t)(BufferValue *a, size_t aw, size_t ah, const BufferValue *b, size_t bw, size_t bh, int x, int y, size_t w, size_t h, size_t grid_offset);

cudaError_t launch_2d_kernel(kernel_2d_func_t kernel, Buffer a, size_t aw, const Buffer b, size_t bw, int x, int y, size_t w, size_t h, void *data)
{
  if((a != NULL && (a->length % aw) == 0)
     && (b != NULL && (b->length % bw) == 0)
     && x < aw && (x + (int)w) >= 0
     && y < (a->length / aw) && (y + (int)h) >= 0) {
    unsigned int i, j;

    //dim3 bsize(_block_dim.x, _block_dim.y);
    dim3 bsize(min((unsigned long)_block_dim.y, (unsigned long)w),
               min((unsigned long)_block_dim.y, (unsigned long)h)); // TODO may have a 4G limit
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


typedef void (*modkerneld_func_t)(int, BufferValue *, BufferValue, int);

void launchd_modkernel(modkerneld_func_t kernel, const Buffer a, BufferValue b, size_t offset)
{
  size_t i;
  size_t length = a->length - offset;
  size_t grid_size = (length + _block_size - 1) / _block_size;

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
      err = cudaMemGetInfo(&_total_bytes_free, &_total_memory);
      if(err != cudaSuccess) {
        return err;
      }

      _initialized = device;

      return cudaSuccess;
    } else {
      return err;
    }
  } else {
    return cudaSuccess;
  }
}

__global__ void buffer_setd_inner(int len, BufferValue *a, BufferValue b, int grid_offset)
{
  int i = grid_offset + grid(1);
  if(i < len) {
    a[i] = b;
  }
}

PUBLIC cudaError_t buffer_setd(Buffer b, BufferValue value, size_t offset, size_t length)
{
  if(value == 0.0) {
    if(offset < b->length) {
      if(offset + length >= b->length) {
        length = b->length - offset;
      }
      return cudaMemset(b->data, 0, length * sizeof(BufferValue));
    } else {
      return cudaSuccess;
    }
  } else {
    launchd_modkernel(buffer_setd_inner, b, value, offset);
    return cudaSuccess;
  }
}

PUBLIC Buffer buffer_new(size_t length, BufferValue initial_value)
{
  size_t bytes = length * sizeof(BufferValue);
    
  if(buffer_init(0) != 0 || bytes >= _total_memory) {
    return NULL;
  }
  
  Buffer ptr = (Buffer)malloc(sizeof(Buffer_s));;
  if(ptr != NULL) {
    if(cuda_alloc((void **)&ptr->data, bytes) != cudaSuccess) {
      ptr->data = NULL;
      buffer_free(ptr);
      return NULL;
    }
    
    ptr->length = length;
    if(buffer_setd(ptr, initial_value, 0, length) != cudaSuccess) {
      buffer_free(ptr);
      return NULL;
    }
  }
      
  return ptr;
}

PUBLIC cudaError_t buffer_free(Buffer buffer)
{
  if(buffer != NULL) {
    if(buffer->data != NULL) {
      cudaError_t err = cuda_free(buffer->data, buffer->length * sizeof(BufferValue));
      if(err != cudaSuccess) {
        return err;
      }
    }
    buffer->data = NULL;
    buffer->length = 0;
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
  return cudaMemcpy(buffer->data, other->data, length * sizeof(BufferValue), cudaMemcpyDeviceToDevice);
}

PUBLIC cudaError_t buffer_setn(Buffer buffer, size_t offset, Buffer other, size_t length)
{
  if(length > other->length) {
    length = other->length;
  }
  if((offset + length) > buffer->length) {
    length = buffer->length - offset;
  }
  return cudaMemcpy(buffer->data + offset, other->data, length * sizeof(BufferValue), cudaMemcpyDeviceToDevice);
}

PUBLIC cudaError_t buffer_setvn(Buffer buffer, size_t offset, void *data, size_t length)
{
  if((offset + length) > buffer->length) {
    length = buffer->length - offset;
  }
  return cudaMemcpy(buffer->data + offset, data, length * sizeof(BufferValue), cudaMemcpyHostToDevice);
}

PUBLIC cudaError_t buffer_setv(Buffer buffer, void *data, size_t length)
{
  return buffer_setvn(buffer, 0, data, length);
}

PUBLIC cudaError_t buffer_set_element(Buffer buffer, size_t n, BufferValue v)
{
  if(buffer != NULL && n < buffer->length) {
    return cudaMemcpy(buffer->data + n, &v, sizeof(BufferValue), cudaMemcpyHostToDevice);
  } else {
    return cudaErrorUnknown;
  }
}

PUBLIC BufferValue buffer_get_element(Buffer buffer, size_t n)
{
  if(buffer != NULL && n < buffer->length) {
    BufferValue out;
    cudaMemcpy(&out, buffer->data + n, sizeof(BufferValue), cudaMemcpyDeviceToHost);
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
  
  cudaError_t err = cudaMemcpy(out, buffer->data, max_length * sizeof(BufferValue), cudaMemcpyDeviceToHost);

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

    cudaError_t err = cudaMemcpy(out->data, buffer->data + n, max_length * sizeof(BufferValue), cudaMemcpyDeviceToDevice);

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
    
    cudaError_t err = cudaMemcpy(out, buffer->data + n, max_length * sizeof(BufferValue), cudaMemcpyDeviceToHost);
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

PUBLIC Buffer buffer_slice_2d(const Buffer in, int width, int height, int x, int y, int out_width, int out_height, BufferValue empty)
{
  Buffer out = buffer_new(out_width * out_height, empty);
  if(out == NULL) return NULL;

  int w = out_width;
  int h = out_height;
  int ox = 0, oy = 0;

  //fprintf(stderr, "slice2d: %ix%i %i,%i %i,%i %i,%i\n", width, height, x, y, ox, oy, w, h);
  
  if(y+h < 0 || y >= height || x+w < 0 || x >= width) {
    return out;
  }

  if(y < 0) {
    h+=y;
    oy=-y;
    y=0;
  } else if(y+h >= height) {
    h=height-y;
  }
    
  if(x < 0) {
    w+=x;
    ox=-x;
    x=0;
  } else if(x+w >= width) {
    w=width-x;
  }
    
  int oi = oy * out_width + ox;
  int ii = y * width + x;

  if(ii >= in->length || oi >= out->length) {
    return out;
  }
  //fprintf(stderr, "  slice2d: %i %i %i,%i %i,%i %i,%i\n", oi, ii, x, y, ox, oy, w, h);
  cudaError_t err = cudaMemcpy2D(out->data + oi, out_width * sizeof(BufferValue),
                                 in->data + ii, width * sizeof(BufferValue),
                                 w * sizeof(BufferValue), h,
                                 cudaMemcpyDeviceToDevice);
  if(err != cudaSuccess) {
    buffer_free(out);
    return NULL;
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
                                 dest_width * sizeof(BufferValue),
                                 src, src_width * sizeof(BufferValue),
                                 copy_width * sizeof(BufferValue),
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

typedef BufferValue (*ReduceOp)(BufferValue a, BufferValue b);
typedef void (*ReduceKernel)(const BufferValue *, BufferValue *, size_t, size_t);

// todo may only work w/ sizes that are powers of two

#define REDUCE_KERNEL(NAME, OP, INITIAL)                                \
  BufferValue NAME ## _initial_value = INITIAL;                              \
  BufferValue NAME ## _op(BufferValue a, BufferValue b)                                \
  {                                                                     \
    return OP;                                                          \
  }                                                                     \
  __device__ BufferValue NAME ## _op_device(BufferValue a, BufferValue b)              \
  {                                                                     \
    return OP;                                                          \
  }                                                                     \
                                                                        \
  __global__ void NAME(const BufferValue *data, BufferValue *partial_sums, size_t length, size_t offset) \
  {                                                                     \
    extern __shared__ BufferValue sdata[];                                   \
    int i = offset + grid(1);                                           \
    int n;                                                              \
                                                                        \
    sdata[threadIdx.x] = (i < length)? data[i] : INITIAL;               \
    __syncthreads();                                                    \
                                                                        \
    for(n = blockDim.x / 2; n > 0; n = n / 2) {                         \
      if(threadIdx.x < n) {                                             \
        sdata[threadIdx.x] = NAME ## _op_device(sdata[threadIdx.x], sdata[threadIdx.x + n]); \
      }                                                                 \
      __syncthreads();                                                  \
    }                                                                   \
                                                                        \
    if(threadIdx.x == 0) {                                              \
      partial_sums[blockIdx.x] = NAME ## _op_device(partial_sums[blockIdx.x], sdata[0]); \
    }                                                                   \
  }

// todo reduce2d for pooling

BufferValue launch_reduce_inner(ReduceOp op, ReduceKernel reduce_kernel, const Buffer b, BufferValue initial)
{
  size_t i;
  size_t grid_size = (b->length + _block_size - 1) / _block_size;
  Buffer partial_buffer;
  BufferValue *partial_sums;
  BufferValue out = initial;
  size_t usable_grid = grid_size;

  if(b == NULL || b->length == 0) {
    return NAN;
  }
  
  if(grid_size >= _max_grid_size) {
    usable_grid = _max_grid_size;
  }

  int num_partials = usable_grid;
  partial_buffer = buffer_new(num_partials, initial);
  if(partial_buffer == NULL) return NAN;
  partial_sums = (BufferValue *)malloc(sizeof(BufferValue) * num_partials);
  if(partial_sums == NULL) {
    buffer_free(partial_buffer);
    return NAN;
  }

  for(i = 0; i < grid_size / usable_grid; i++) {
    reduce_kernel<<< usable_grid, _block_size, _block_size * sizeof(BufferValue) >>>(b->data, partial_buffer->data, b->length, i * usable_grid);

    buffer_get(partial_buffer, partial_sums, num_partials);

    for(i = 0; i < num_partials; i++) {
      out = op(out, partial_sums[i]);
    }
  }

  buffer_free(partial_buffer);
  free(partial_sums);
  
  return out;
}

#define launch_reduce(kernel, buffer) launch_reduce_inner(kernel ## _op, kernel, buffer, kernel ## _initial_value)


REDUCE_KERNEL(buffer_sum_kernel, a + b, 0.0);
PUBLIC BufferValue buffer_sum(const Buffer b)
{
  return launch_reduce(buffer_sum_kernel, b);
}

REDUCE_KERNEL(buffer_product_kernel, a * b, 1.0);
PUBLIC BufferValue buffer_product(const Buffer b)
{
  return launch_reduce(buffer_product_kernel, b);
}

REDUCE_KERNEL(buffer_min_kernel, fmin(a, b), NAN);
PUBLIC BufferValue buffer_min(const Buffer b)
{
  return launch_reduce(buffer_min_kernel, b);
}

REDUCE_KERNEL(buffer_max_kernel, fmax(a, b), NAN);
PUBLIC BufferValue buffer_max(const Buffer b)
{
  return launch_reduce(buffer_max_kernel, b);
}

// todo overrun potential adding grid_size in the cuda kernel

#define BINARY_OP(name, operation)                                      \
  __device__ inline BufferValue buffer_ ## name ## _op(BufferValue a, BufferValue b)          \
  {                                                                     \
    return(operation);                                                  \
  }                                                                     \
                                                                        \
  __global__ void buffer_ ## name ##_inner(int len, BufferValue *out, const BufferValue *a, const BufferValue *b, int grid_offset, void *) \
  {                                                                     \
    int i = grid_offset + grid(1);                                      \
    if(i < len) {                                                       \
      out[i] = buffer_ ## name ## _op(a[i], b[i]);                      \
    }                                                                   \
  }                                                                     \
                                                                        \
  __device__ void dev_buffer_ ## name(Buffer out, const Buffer a, const Buffer b) \
  {                                                                     \
    dev_launch_kernel(buffer_ ## name ## _inner, out, a, b, NULL);       \
  }                                                                     \
  PUBLIC Buffer buffer_ ## name(const Buffer a, const Buffer b)         \
  {                                                                     \
    return launch_kernel(buffer_ ## name ## _inner, a, b, NULL);        \
  }                                                                     \
                                                                        \
  __global__ void buffer_ ## name ## _2d_inner(BufferValue *a, size_t aw, size_t ah, const BufferValue *b, size_t bw, size_t bh, int x, int y, size_t w, size_t h, size_t grid_offset) \
  {                                                                     \
    int bx = blockIdx.x * blockDim.x + threadIdx.x;                     \
    int by = blockIdx.y * blockDim.y + threadIdx.y;                     \
    int ax = bx + x;                                                    \
    int ay = by + y;                                                    \
    if(bx >= 0 && bx < w                                                \
       && ax >= 0 && ax < aw                                            \
       && by >= 0 && by < h                                             \
       && ay >= 0 && ay < ah) {                                         \
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
  __global__ void buffer_ ## name ## d_inner(int len, BufferValue *out, const BufferValue *a, const BufferValue b, int grid_offset) \
  {                                                                     \
    int i = grid_offset + grid(1);                                      \
    if(i < len) {                                                       \
      out[i] = buffer_ ## name ## _op(a[i], b);                         \
    }                                                                   \
  }                                                                     \
                                                                        \
  PUBLIC Buffer buffer_ ## name ## d(const Buffer a, BufferValue b)            \
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
BINARY_OP(bsl, (BufferValue)((unsigned long)a << (unsigned long)b));
BINARY_OP(bsr, (BufferValue)((unsigned long)a >> (unsigned long)b));
BINARY_OP(and, (BufferValue)((unsigned long)a & (unsigned long)b));
BINARY_OP(or, (BufferValue)((unsigned long)a | (unsigned long)b));
BINARY_OP(xor, (BufferValue)((unsigned long)a ^ (unsigned long)b));

#undef BINARY_OP

PUBLIC int buffer_eq(const Buffer a, const Buffer b)
{
  // compare
  Buffer results = buffer_collect_eq(a, b);
  if(results != NULL) {
    // reduce
    BufferValue sum = buffer_sum(results);
    
    // clean up
    buffer_free(results);
  
    return sum == a->length;
  } else {
    return 0;
  }
}

#define FUNCTION_OP(name, operation) \
  __global__ void buffer_ ## name ## _inner(int len, BufferValue *out, const BufferValue *a, const BufferValue b, int grid_offset) \
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

// todo rename dot?

__global__ void buffer_dot_inner(BufferValue *out, size_t out_pitch, const BufferValue *a, const BufferValue *b, size_t aw, size_t ap, size_t ah, size_t bw, size_t bp, size_t bh)
{
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  BufferValue sum = 0.0;

  if (row < ah && col < bw) {
    for (size_t i = 0; i < min(bh, aw); i++) {
        sum += a[row * ap + i] * b[i * bp + col];
    }
  }
  out[row * out_pitch + col] = sum;
}

PUBLIC Buffer buffer_dot(const Buffer a, size_t aw, size_t ah, const Buffer b, size_t bw, size_t bh)
{
  if(aw * ah != a->length && bw * bh != b-> length && aw != bh) {
    return NULL;
  } else {
    Buffer out = buffer_new(ah * bw, 0.0);
    if(out == NULL) return NULL;

    dim3 dim(bw, ah);
    buffer_dot_inner<<< dim, 1 >>>(out->data, bw, a->data, b->data, aw, aw, ah, bw, bw, bh);
    
    return out;
  }
}

__global__ void buffer_identity_inner(BufferValue *a, size_t size, size_t grid_offset)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x + grid_offset;
  size_t i = col * (size_t)size + col;

  if(i < size*size) {
    a[i] = 1.0;
  }
}

PUBLIC Buffer buffer_identity(size_t size)
{
  Buffer out = buffer_new(size * size, 0.0);
  if(out == NULL) {
    return NULL;
  }

  size_t grid_size = (size + _block_size - 1) / _block_size;
  size_t usable_grid = grid_size;
  if(grid_size >= _max_grid_size) {
    usable_grid = _max_grid_size;
  }

  for(size_t i = 0; i < grid_size / usable_grid; i++) {
    buffer_identity_inner<<< usable_grid, _block_size >>>(out->data, size, i * usable_grid);
  }
  
  return out;
}

__global__ void buffer_diagflat_inner(BufferValue *data, const BufferValue *a, size_t len)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if(col >= len) {
    return;
  }
  
  data[col * len + col] = a[col];
}

PUBLIC Buffer buffer_diagflat(const Buffer a)
{
  Buffer i = buffer_new(a->length * a->length, 0.0);
  dim3 dim(a->length);
  buffer_diagflat_inner<<< dim, 1 >>>(i->data, a->data, a->length);
  return i;
}

__global__ void buffer_transpose_inner(BufferValue *out, size_t outw, size_t outh, const BufferValue *in, size_t inw, size_t inh, int x, int y, size_t w, size_t h, size_t grid_offset)
//__global__ void buffer_transpose_inner(const BufferValue *in, BufferValue *out, size_t size, size_t width, size_t height)
{
  int bx = blockIdx.x * blockDim.x + threadIdx.x;
  int by = blockIdx.y * blockDim.y + threadIdx.y;
  int col = bx + x;
  int row = by + y;
  size_t ii = grid_offset + row * inw + col;
  size_t oi = grid_offset + col * outw + row;

  if(col >= outh || row >= outw ||
     col >= inw || row >= inh) {
    return;
  }
  
  out[oi] = in[ii];
}

PUBLIC Buffer buffer_transpose(const Buffer in, size_t width, size_t height)
{
  Buffer out = buffer_new(width * height, 0.0);
  if(out == NULL) {
    return NULL;
  }

  launch_2d_kernel(buffer_transpose_inner, out, height, in, width, 0, 0, width, height, NULL);
  // buffer_transpose_inner<<< width, height >>>(in->data, out->data, width*height, width, height);
  
  return out;
}

// convolve dot

__global__ void buffer_conv2d_dot_inner(BufferValue *out, size_t op,
                                      const BufferValue *a,
                                      const BufferValue *b,
                                      size_t aw, size_t ah,
                                      size_t bw, size_t bh,
                                      int sx, int sy,
                                      size_t cw, size_t ch,
                                      dim3 conv_size)
{
  // This is called for each convolution box. The A input
  // is sliced into CWxCH buffers and a dot product of that
  // and B is stored in OUT.
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  // todo not a matrix multiplication
  if (row < conv_size.y && col < conv_size.x) {
    // Out[i*cw, j*ch] = A[i*sx,j*sy,cw,ch] * B
    size_t out_start = row * ch * op + col * bw;
    size_t a_start = row * sy * aw + col * sx;
    dim3 dim(bw, ch);
    buffer_dot_inner<<< dim, 1 >>>(out + out_start, op,
                                   a + a_start,
                                   b,
                                   min(cw, aw - sx * col), aw, min(ch, ah - sy * row),
                                   bw, bw, bh);
  }
}

PUBLIC Buffer buffer_conv2d_dot(const Buffer a, size_t aw, size_t ah, const Buffer b, size_t bw, size_t bh, int sx, int sy, size_t cw, size_t ch, BufferValue init)
{
  if(aw * ah != a->length && bw * bh != b-> length && cw != bh) {
    return NULL;
  } else {
    dim3 convolutions(aw / sx, ah / sy);
    Buffer out = buffer_new(convolutions.x * bw * convolutions.y * ch, init);
    if(out == NULL) return NULL;

    buffer_conv2d_dot_inner<<< convolutions, 1 >>>
      (out->data, convolutions.x * bw,
       a->data, b->data,
       aw, ah,
       bw, bh,
       sx, sy,
       cw, ch, convolutions);
    
    return out;
  }
}

__global__ void buffer_maxpool1d_kernel(const BufferValue *in, size_t in_size, size_t pool_size, BufferValue *out)
{
  extern __shared__ BufferValue data[];
  int in_x = blockIdx.x * blockDim.x + threadIdx.x;
  int n;
  
  data[threadIdx.x] = in_x < in_size ? in[in_x] : -INFINITY;
  __syncthreads();

  for(n = blockDim.x / 2; n > 0; n = n / 2) {
    if(threadIdx.x < n) {
      data[threadIdx.x] = data[threadIdx.x] > data[threadIdx.x + n] ? data[threadIdx.x] : data[threadIdx.x + n];
      __syncthreads();
    }
  }

  if(threadIdx.x == 0) {
    out[blockIdx.x] = out[blockIdx.x] > data[0] ? out[blockIdx.x] : data[0];
  }
}

PUBLIC Buffer buffer_maxpool1d(const Buffer buf, size_t req_pool_size)
{
  size_t pool_size = min(buf->length, req_pool_size);
  size_t n = (buf->length + pool_size - 1) / pool_size;
  Buffer out = buffer_new(n, -INFINITY);
  buffer_maxpool1d_kernel<<< n, pool_size, pool_size * sizeof(BufferValue) >>>(buf->data, buf->length, pool_size, out->data);
  return out;
}

__global__ void buffer_maxpool1d_idx_kernel(const BufferValue *in, size_t in_size, size_t pool_size, BufferValue *out, BufferValue *out_vals)
{
  extern __shared__ BufferValue data[];
  int *indexes = (int *)data;
  BufferValue *values = (BufferValue *)(data + blockDim.x);
  int in_x = blockIdx.x * blockDim.x + threadIdx.x;
  int n;

  if(in_x < in_size) {
    values[threadIdx.x] = in[in_x];
    indexes[threadIdx.x] = in_x;
  } else {
    values[threadIdx.x] = -INFINITY;
    indexes[threadIdx.x] = in_x;
  }
  __syncthreads();

  for(n = blockDim.x / 2; n > 0; n = n / 2) {
    if(threadIdx.x < n) {
      if(values[threadIdx.x] < values[threadIdx.x + n]) {
        values[threadIdx.x] = values[threadIdx.x + n];
        indexes[threadIdx.x] = indexes[threadIdx.x + n];
        __syncthreads();
      }
    }
  }

  if(threadIdx.x == 0) {
    if(out_vals[blockIdx.x] < values[0]) {
      out_vals[blockIdx.x] = values[0];
      out[blockIdx.x] = indexes[0];
    }
  }
}

// todo return both the indexes and values

PUBLIC Buffer buffer_maxpool1d_idx(const Buffer buf, size_t req_pool_size)
{
  size_t pool_size = min(buf->length, req_pool_size);
  size_t n = (buf->length + pool_size - 1) / pool_size;
  Buffer out = buffer_new(n, -1.0);
  Buffer values = buffer_new(n, -INFINITY);
  buffer_maxpool1d_idx_kernel<<< n, pool_size, pool_size * (sizeof(int) + sizeof(BufferValue)) >>>(buf->data, buf->length, pool_size, out->data, values->data);
  buffer_free(values);
  return out;
}

__global__ void buffer_maxpool2d_kernel(const BufferValue *in, dim3 in_dim, dim3 blocks, dim3 pool_size, BufferValue *out)
{
  extern __shared__ BufferValue data[];
  int in_x = blockIdx.x * blockDim.x + threadIdx.x;
  int in_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = in_y * in_dim.x + in_x;
  int nx, ny;

  data[threadIdx.y * blockDim.x + threadIdx.x] = idx < in_dim.x*in_dim.y ? in[idx] : -INFINITY;
  __syncthreads();

  /*
  printf("X %i\tY %i\t\t"
         "Bx %i\tBy %i\t"
         "W %i\tH %i\t"
         "idx %i\tn = %lf\n",
         threadIdx.x, threadIdx.y,
         blockIdx.x, blockIdx.y,
         blockDim.x, blockDim.y,
         idx, data[threadIdx.y * blockDim.x + threadIdx.x]);
  */
  
  for(ny = blockDim.y / 2; ny > 0; ny = ny / 2) {
    if(threadIdx.y < ny) {
      for(nx = blockDim.x / 2; nx > 0; nx = nx / 2) {
        if(threadIdx.x < nx) {
          int i = threadIdx.y * blockDim.x + threadIdx.x;
          int j = ny * blockDim.x + nx;
          data[i] = data[i] > data[i + j] ? data[i] : data[i + j];
          __syncthreads();
        }
      }
    }
  }
  
  if(threadIdx.y == 0 && threadIdx.x == 0) {
    int i = blockIdx.y * blocks.x + blockIdx.x;
    out[i] = out[i] > data[0] ? out[i] : data[0];
  }
}

PUBLIC Buffer buffer_maxpool2d(const Buffer buf, size_t buf_w, size_t buf_h, size_t req_pool_w, size_t req_pool_h)
{
  size_t pool_w = min(buf_w, req_pool_w);
  size_t pool_h = min(buf_h, req_pool_h);
  dim3 pool_dim = { pool_w, pool_h };
  dim3 buf_d = { buf_w, buf_h };
  size_t nx = (buf_w + pool_w - 1) / pool_w;
  size_t ny = (buf_h + pool_h - 1) / pool_h;
  dim3 n = { nx, ny };
  Buffer out = buffer_new(nx*ny, -INFINITY);
  buffer_maxpool2d_kernel<<< n, pool_dim, pool_w * pool_h * sizeof(BufferValue) >>>(buf->data, buf_d, n, pool_dim, out->data);
  return out;
}

__global__ void buffer_maxpool2d_idx_kernel(const BufferValue *in, dim3 in_dim, dim3 blocks, dim3 pool_size, BufferValue *out, BufferValue *out_vals)
{
  extern __shared__ BufferValue data[];
  int *indexes = (int *)data;
  BufferValue *values = (BufferValue *)((int *)data + blockDim.x*blockDim.y);

  int in_x = blockIdx.x * blockDim.x + threadIdx.x;
  int in_y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = in_y * in_dim.x + in_x;
  int nx, ny;

  if(idx < in_dim.x*in_dim.y) {
    values[threadIdx.y * blockDim.x + threadIdx.x] = in[idx];
    indexes[threadIdx.y * blockDim.x + threadIdx.x] = idx;
  } else {
    values[threadIdx.y * blockDim.x + threadIdx.x] = -INFINITY;
    indexes[threadIdx.y * blockDim.x + threadIdx.x] = idx;
  }
  __syncthreads();

  for(ny = blockDim.y / 2; ny > 0; ny = ny / 2) {
    if(threadIdx.y < ny) {
      for(nx = blockDim.x / 2; nx > 0; nx = nx / 2) {
        if(threadIdx.x < nx) {
          int i = threadIdx.y * blockDim.x + threadIdx.x;
          int j = ny * blockDim.x + nx;
          if(values[i] < values[i + j]) {
            values[i] = values[i + j];
            indexes[i] = indexes[i + j];
            __syncthreads();
          }
        }
      }
    }
  }
  
  if(threadIdx.y == 0 && threadIdx.x == 0) {
    int i = blockIdx.y * blocks.x + blockIdx.x;
    if(out[i] < values[0]) {
      out_vals[i] = values[0];
      out[i] = indexes[0];
    }
  }
}

PUBLIC Buffer buffer_maxpool2d_idx(const Buffer buf, size_t buf_w, size_t buf_h, size_t req_pool_w, size_t req_pool_h)
{
  size_t pool_w = min(buf_w, req_pool_w);
  size_t pool_h = min(buf_h, req_pool_h);
  dim3 pool_dim = { pool_w, pool_h };
  dim3 buf_d = { buf_w, buf_h };
  size_t nx = (buf_w + pool_w - 1) / pool_w;
  size_t ny = (buf_h + pool_h - 1) / pool_h;
  dim3 n = { nx, ny };
  Buffer out = buffer_new(nx*ny, -INFINITY);
  Buffer out_vals = buffer_new(nx*ny, -INFINITY);
  buffer_maxpool2d_idx_kernel<<< n, pool_dim, pool_w * pool_h * (sizeof(int) + sizeof(BufferValue)) >>>(buf->data, buf_d, n, pool_dim, out->data, out_vals->data);
  buffer_free(out_vals);
  return out;
}
