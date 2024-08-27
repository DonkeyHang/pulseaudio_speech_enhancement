#ifndef VIMDEMO_COMMON_AUDIO_RING_BUFFER_H_
#define VIMDEMO_COMMON_AUDIO_RING_BUFFER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>  // size_t

enum Wrap { SAME_WRAP, DIFF_WRAP };

typedef struct RingBuffer {
  size_t read_pos;
  size_t write_pos;
  size_t element_count;
  size_t element_size;
  enum Wrap rw_wrap;
  char* data;
} RingBuffer;

// Creates and initializes the buffer. Returns null on failure.
RingBuffer* CreateBuffer(size_t element_count, size_t element_size);
void InitBuffer(RingBuffer* handle);
void FreeBuffer(RingBuffer* handle);

// A ring buffer to hold arbitrary data. Provides no thread safety. Unless
// otherwise specified, functions return 0 on success and -1 on error.


// Reads data from the buffer. The |data_ptr| will point to the address where
// it is located. If all |element_count| data are feasible to read without
// buffer wrap around |data_ptr| will point to the location in the buffer.
// Otherwise, the data will be copied to |data| (memory allocation done by the
// user) and |data_ptr| points to the address of |data|. |data_ptr| is only
// guaranteed to be valid until the next call to WebRtc_WriteBuffer().
//
// To force a copying to |data|, pass a null |data_ptr|.
//
// Returns number of elements read.
size_t ReadBuffer(RingBuffer* handle,
                         void** data_ptr,
                         void* data,
                         size_t element_count);

// Writes |data| to buffer and returns the number of elements written.
size_t WriteBuffer(RingBuffer* handle, const void* data,
                          size_t element_count);

// Moves the buffer read position and returns the number of elements moved.
// Positive |element_count| moves the read position towards the write position,
// that is, flushing the buffer. Negative |element_count| moves the read
// position away from the the write position, that is, stuffing the buffer.
// Returns number of elements moved.
int MoveReadPtr(RingBuffer* handle, int element_count);

// Returns number of available elements to read.
size_t Available_read(const RingBuffer* handle);

// Returns number of available elements for write.
size_t Available_write(const RingBuffer* handle);

#ifdef __cplusplus
}
#endif

#endif  // VIMDEMO_COMMON_AUDIO_RING_BUFFER_H_ 
