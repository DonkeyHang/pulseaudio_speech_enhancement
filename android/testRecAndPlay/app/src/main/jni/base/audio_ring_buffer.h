//
//  audio_ring_buffer.h
//
#ifndef SUBMODULES_FFMPEG_LIBAVFILTER_AGC_COMMON_AUDIO_AUDIO_RING_BUFFER_H_
#define SUBMODULES_FFMPEG_LIBAVFILTER_AGC_COMMON_AUDIO_AUDIO_RING_BUFFER_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <vector>

struct RingBuffer;

// A ring buffer tailored for float deinterleaved audio. Any operation that
// cannot be performed as requested will cause a crash (e.g. insufficient data
// in the buffer to fulfill a read request.)
class AudioRingBuffer {
 public:
  // Specify the number of channels and maximum number of samples the buffer
  // will contain.
  AudioRingBuffer(size_t channels, size_t max_samples);
  ~AudioRingBuffer();

  // Copies |data| to the buffer and advances the write pointer. |channels| must
  // be the same as at creation time.
  void Write(float *data, size_t channels, size_t frames);
  void Write(const std::vector<float> &data, size_t channels,
             size_t samples);

  // Copies from the buffer to |data| and advances the read pointer. |channels|
  // must be the same as at creation time.
  void Read(float *data, size_t channels, size_t frames);
  void Read(std::vector<float> &data, size_t channels, size_t samples);

  size_t ReadFramesAvailable() const;
  size_t WriteFramesAvailable() const;

  // Moves the read position. The forward version advances the read pointer
  // towards the write pointer and the backward verison withdraws the read
  // pointer away from the write pointer (i.e. flushing and stuffing the buffer
  // respectively.)
  void MoveReadPositionForward(size_t samples);
  void MoveReadPositionBackward(size_t samples);

 private:
  // TODO(kwiberg): Use std::vector<std::unique_ptr<RingBuffer>> instead.
  std::vector<RingBuffer*> buffers_;
};

#endif  // SUBMODULES_FFMPEG_LIBAVFILTER_AGC_COMMON_AUDIO_AUDIO_RING_BUFFER_H_
