#include "audio_ring_buffer.h"
#include "ring_buffer.h"

// This is a simple multi-channel wrapper over the ring_buffer.h C interface.


AudioRingBuffer::AudioRingBuffer(size_t channels, size_t max_frames) {
  buffers_.reserve(channels);
  for (size_t i = 0; i < channels; ++i)
    buffers_.push_back(CreateBuffer(max_frames, sizeof(float)));
}

AudioRingBuffer::~AudioRingBuffer() {
  // for ( RingBuffer* buf : buffers_ )
  int32_t channels = buffers_.size();
  for (int32_t i = 0; i < channels; ++i)
    FreeBuffer(buffers_[i]);
  /*for (std::vector<RingBuffer*>::iterator buf = buffers_.begin();
      buf != buffers_.end(); ++buf)*/
  // FreeBuffer(buf);
}

void AudioRingBuffer::Write(float *data, size_t channels,
                            size_t frames) {
  if (buffers_.size() != channels) {
    return;
  }

  for (size_t i = 0; i < channels; ++i) {
    const size_t written = WriteBuffer(buffers_[i], &data[i], frames);
    if (written != frames) {
      return;
    }
  }
}

void AudioRingBuffer::Write(const std::vector<float> &data, size_t channels,
                            size_t frames) {
  if (buffers_.size() != channels) {
    return;
  }

  for (size_t i = 0; i < channels; ++i) {
    const size_t written = WriteBuffer(buffers_[i], &data[i], frames);
    if (written != frames) {
      return;
    }
  }
}

void AudioRingBuffer::Read(float *data, size_t channels, size_t frames) {
  if (buffers_.size() != channels) {
    return;
  }
  for (size_t i = 0; i < channels; ++i) {
    const size_t read =
        ReadBuffer(buffers_[i], NULL, &data[i], frames);
    if (read != frames) {
      return;
    }
  }
}

void AudioRingBuffer::Read(std::vector<float> &data, size_t channels,
                           size_t frames) {
  if (buffers_.size() != channels) {
    return;
  }
  for (size_t i = 0; i < channels; ++i) {
    const size_t read =
        ReadBuffer(buffers_[i], NULL, &data[i], frames);
    if (read != frames) {
      return;
    }
  }
}

size_t AudioRingBuffer::ReadFramesAvailable() const {
  // All buffers have the same amount available.
  return Available_read(buffers_[0]);
}

size_t AudioRingBuffer::WriteFramesAvailable() const {
  // All buffers have the same amount available.
  return Available_write(buffers_[0]);
}

void AudioRingBuffer::MoveReadPositionForward(size_t frames) {
  // for (std::vector<RingBuffer*>::iterator buf = buffers_.begin();
  //   buf != buffers_.end(); ++buf) {
  int32_t channels = buffers_.size();
  for (int32_t i = 0; i < channels; ++i) {
    const size_t moved =
      static_cast<size_t>(MoveReadPtr(buffers_[i],
                          static_cast<int32_t>(frames)));
    if (moved != frames) {
      return;
    }
  }
}

void AudioRingBuffer::MoveReadPositionBackward(size_t frames) {
  // for (std::vector<RingBuffer*>::iterator buf = buffers_.begin();
  //   buf != buffers_.end(); ++buf) {
  int32_t channels = buffers_.size();
  for (int32_t i = 0; i < channels; ++i) {
    const size_t moved = static_cast<size_t>(
        -MoveReadPtr(buffers_[i], -static_cast<int32_t>(frames)));
    if (moved != frames) {
      return;
    }
  }
}
