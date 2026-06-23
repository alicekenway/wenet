#include "frontend/ring_buffer.h"

#include <algorithm>

namespace wenet_sdk::internal {

FloatRingBuffer::FloatRingBuffer(size_t capacity) : buffer_(capacity) {}

size_t FloatRingBuffer::Write(const float* data, size_t n) {
  if (buffer_.empty() || data == nullptr) {
    return 0;
  }
  size_t written = 0;
  while (written < n && size_ < buffer_.size()) {
    buffer_[write_pos_] = data[written++];
    write_pos_ = (write_pos_ + 1) % buffer_.size();
    ++size_;
  }
  return written;
}

size_t FloatRingBuffer::Read(float* data, size_t n) {
  if (buffer_.empty() || data == nullptr) {
    return 0;
  }
  size_t read = 0;
  while (read < n && size_ > 0) {
    data[read++] = buffer_[read_pos_];
    read_pos_ = (read_pos_ + 1) % buffer_.size();
    --size_;
  }
  return read;
}

void FloatRingBuffer::Clear() {
  read_pos_ = 0;
  write_pos_ = 0;
  size_ = 0;
}

}  // namespace wenet_sdk::internal
