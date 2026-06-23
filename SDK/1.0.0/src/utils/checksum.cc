#include "utils/checksum.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "decoder/symbol_table.h"

namespace wenet_sdk::internal {

namespace {

constexpr std::array<uint32_t, 64> kSha256K = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

uint32_t RotateRight(uint32_t value, int bits) {
  return (value >> bits) | (value << (32 - bits));
}

uint32_t ReadBigEndian32(const unsigned char* data) {
  return (static_cast<uint32_t>(data[0]) << 24) |
         (static_cast<uint32_t>(data[1]) << 16) |
         (static_cast<uint32_t>(data[2]) << 8) |
         static_cast<uint32_t>(data[3]);
}

void ProcessSha256Block(const unsigned char* block,
                        std::array<uint32_t, 8>* state) {
  std::array<uint32_t, 64> w{};
  for (int i = 0; i < 16; ++i) {
    w[static_cast<size_t>(i)] = ReadBigEndian32(block + i * 4);
  }
  for (int i = 16; i < 64; ++i) {
    const uint32_t s0 = RotateRight(w[static_cast<size_t>(i - 15)], 7) ^
                        RotateRight(w[static_cast<size_t>(i - 15)], 18) ^
                        (w[static_cast<size_t>(i - 15)] >> 3);
    const uint32_t s1 = RotateRight(w[static_cast<size_t>(i - 2)], 17) ^
                        RotateRight(w[static_cast<size_t>(i - 2)], 19) ^
                        (w[static_cast<size_t>(i - 2)] >> 10);
    w[static_cast<size_t>(i)] = w[static_cast<size_t>(i - 16)] + s0 +
                                w[static_cast<size_t>(i - 7)] + s1;
  }

  uint32_t a = (*state)[0];
  uint32_t b = (*state)[1];
  uint32_t c = (*state)[2];
  uint32_t d = (*state)[3];
  uint32_t e = (*state)[4];
  uint32_t f = (*state)[5];
  uint32_t g = (*state)[6];
  uint32_t h = (*state)[7];

  for (int i = 0; i < 64; ++i) {
    const uint32_t s1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^
                        RotateRight(e, 25);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t temp1 =
        h + s1 + ch + kSha256K[static_cast<size_t>(i)] +
        w[static_cast<size_t>(i)];
    const uint32_t s0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^
                        RotateRight(a, 22);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temp2 = s0 + maj;

    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  (*state)[0] += a;
  (*state)[1] += b;
  (*state)[2] += c;
  (*state)[3] += d;
  (*state)[4] += e;
  (*state)[5] += f;
  (*state)[6] += g;
  (*state)[7] += h;
}

std::string HexState(const std::array<uint32_t, 8>& state) {
  std::ostringstream os;
  os << std::hex << std::setfill('0');
  for (uint32_t value : state) {
    os << std::setw(8) << value;
  }
  return os.str();
}

uint64_t Fnv1a64(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  uint64_t hash = 1469598103934665603ull;
  char c = 0;
  while (in.get(c)) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

std::string NormalizeChecksumPath(std::string path) {
  path = Trim(std::move(path));
  if (!path.empty() && path.front() == '*') {
    path.erase(path.begin());
  }
  return Trim(std::move(path));
}

}  // namespace

Status ValidateChecksumFileIfPresent(const std::filesystem::path& model_dir) {
  const auto checksum = model_dir / "checksum.sha256";
  if (!std::filesystem::exists(checksum)) {
    return Status::OK();
  }
  std::ifstream in(checksum);
  if (!in) {
    return Status::NotFound("failed to open checksum file: " +
                            checksum.string());
  }

  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    line = Trim(std::move(line));
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream iss(line);
    std::string expected;
    std::string relative_path;
    iss >> expected;
    std::getline(iss, relative_path);
    relative_path = NormalizeChecksumPath(std::move(relative_path));
    if (expected.size() != 64 || relative_path.empty()) {
      return Status::InvalidArgument("invalid checksum line " +
                                     std::to_string(line_no) + " in " +
                                     checksum.string());
    }
    const auto file = model_dir / relative_path;
    if (!std::filesystem::exists(file)) {
      return Status::NotFound("checksum target is missing: " + file.string());
    }
    const std::string actual = Sha256HexForFile(file);
    if (actual != expected) {
      return Status::InvalidArgument("checksum mismatch for " +
                                     relative_path + ": expected " + expected +
                                     ", got " + actual);
    }
  }
  return Status::OK();
}

std::string Sha256HexForFile(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return "";
  }

  std::array<uint32_t, 8> state = {
      0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::array<unsigned char, 64> block{};
  uint64_t total_bytes = 0;

  while (in) {
    in.read(reinterpret_cast<char*>(block.data()),
            static_cast<std::streamsize>(block.size()));
    const auto read = static_cast<size_t>(in.gcount());
    if (read == block.size()) {
      ProcessSha256Block(block.data(), &state);
      total_bytes += read;
      continue;
    }

    total_bytes += read;
    block[read] = 0x80;
    for (size_t i = read + 1; i < block.size(); ++i) {
      block[i] = 0;
    }
    if (read >= 56) {
      ProcessSha256Block(block.data(), &state);
      block.fill(0);
    }
    const uint64_t total_bits = total_bytes * 8;
    for (int i = 0; i < 8; ++i) {
      block[static_cast<size_t>(63 - i)] =
          static_cast<unsigned char>((total_bits >> (i * 8)) & 0xff);
    }
    ProcessSha256Block(block.data(), &state);
    break;
  }
  return HexState(state);
}

std::string HexDigestForFileForDiagnostics(const std::filesystem::path& path) {
  std::ostringstream os;
  os << std::hex << std::setw(16) << std::setfill('0') << Fnv1a64(path);
  return os.str();
}

}  // namespace wenet_sdk::internal
