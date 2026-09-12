#include <rtabmap/core/rvl_codec.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sqlite3.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: rtabmap_export_selected_depth DB OUT_DIR ID,ID,...\n";
    return 2;
  }
  std::filesystem::create_directories(argv[2]);
  sqlite3 *db = nullptr;
  if (sqlite3_open_v2(argv[1], &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) return 3;
  sqlite3_stmt *stmt = nullptr;
  sqlite3_prepare_v2(db, "SELECT depth FROM Data WHERE id=?", -1, &stmt, nullptr);
  std::stringstream list(argv[3]); std::string token;
  while (std::getline(list, token, ',')) {
    int id = std::stoi(token); sqlite3_reset(stmt); sqlite3_clear_bindings(stmt);
    sqlite3_bind_int(stmt, 1, id);
    if (sqlite3_step(stmt) != SQLITE_ROW) continue;
    const auto *bytes = static_cast<const unsigned char *>(sqlite3_column_blob(stmt, 0));
    int size = sqlite3_column_bytes(stmt, 0);
    if (!bytes || size < 16 || std::memcmp(bytes, "DEPTHRVL", 8) != 0) continue;
    uint32_t width=0, height=0; std::memcpy(&width, bytes+8, 4); std::memcpy(&height, bytes+12, 4);
    if (!width || !height || width>4096 || height>4096) continue;
    cv::Mat depth(static_cast<int>(height), static_cast<int>(width), CV_16UC1);
    rtabmap::RvlCodec codec; codec.DecompressRVL(bytes+16, depth.ptr<uint16_t>(), width*height);
    std::string out = std::string(argv[2]) + "/depth_" + std::to_string(id) + ".png";
    if (!cv::imwrite(out, depth)) { std::cerr << "failed " << out << "\n"; return 4; }
    std::cout << id << " " << width << "x" << height << " valid=" << cv::countNonZero(depth) << "\n";
  }
  sqlite3_finalize(stmt); sqlite3_close(db); return 0;
}
