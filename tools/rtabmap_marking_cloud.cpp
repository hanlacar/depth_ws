#include <rtabmap/core/rvl_codec.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sqlite3.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Point { float x, y, z; uint8_t type; int32_t node; };

static cv::Matx44f transform(const void *blob, int bytes, int floatOffset = 0) {
  cv::Matx44f output = cv::Matx44f::eye();
  if (!blob || bytes < (floatOffset + 12) * static_cast<int>(sizeof(float))) return output;
  const float *values = static_cast<const float *>(blob) + floatOffset;
  for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) output(r, c) = values[r * 4 + c];
  return output;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: rtabmap_marking_cloud DB MAP_ID START STOP OUTPUT.ply\n";
    return 2;
  }
  sqlite3 *db = nullptr;
  if (sqlite3_open_v2(argv[1], &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) return 3;
  sqlite3_stmt *stmt = nullptr;
  const char *sql = "SELECT Node.id,Node.pose,Data.image,Data.depth,Data.calibration,Data.scan_info "
                    "FROM Node JOIN Data USING(id) WHERE Node.map_id=? AND Node.id>=? AND Node.id<=? ORDER BY Node.id";
  if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return 4;
  sqlite3_bind_int(stmt, 1, std::stoi(argv[2]));
  sqlite3_bind_int(stmt, 2, std::stoi(argv[3]));
  sqlite3_bind_int(stmt, 3, std::stoi(argv[4]));
  std::vector<Point> points;
  int nodes = 0;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const int id = sqlite3_column_int(stmt, 0);
    const cv::Matx44f pose = transform(sqlite3_column_blob(stmt, 1), sqlite3_column_bytes(stmt, 1));
    const auto *imageBytes = static_cast<const unsigned char *>(sqlite3_column_blob(stmt, 2));
    const int imageSize = sqlite3_column_bytes(stmt, 2);
    const auto *depthBytes = static_cast<const unsigned char *>(sqlite3_column_blob(stmt, 3));
    const int depthSize = sqlite3_column_bytes(stmt, 3);
    const auto *cal = static_cast<const unsigned char *>(sqlite3_column_blob(stmt, 4));
    const int calSize = sqlite3_column_bytes(stmt, 4);
    const void *scanInfo = sqlite3_column_blob(stmt, 5);
    const int scanInfoSize = sqlite3_column_bytes(stmt, 5);
    if (!imageBytes || !depthBytes || !cal || imageSize <= 0 || depthSize < 16 || calSize < 48 ||
        scanInfoSize < 19 * static_cast<int>(sizeof(float)) || std::memcmp(depthBytes, "DEPTHRVL", 8) != 0) continue;
    cv::Mat image = cv::imdecode(cv::Mat(1, imageSize, CV_8U, const_cast<unsigned char *>(imageBytes)), cv::IMREAD_COLOR);
    if (image.empty()) continue;
    cv::Mat hsv; cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    uint32_t width = 0, height = 0;
    std::memcpy(&width, depthBytes + 8, 4); std::memcpy(&height, depthBytes + 12, 4);
    if (width != static_cast<uint32_t>(image.cols) || height != static_cast<uint32_t>(image.rows)) continue;
    cv::Mat depth(static_cast<int>(height), static_cast<int>(width), CV_16UC1);
    rtabmap::RvlCodec codec;
    codec.DecompressRVL(depthBytes + 16, depth.ptr<uint16_t>(), static_cast<int>(width * height));
    double fx = 388.0, fy = 388.0, cx = 317.4, cy = 247.8;
    if (calSize >= 284) {
      std::memcpy(&fx, cal + 44, 8); std::memcpy(&fy, cal + 76, 8);
      std::memcpy(&cx, cal + 60, 8); std::memcpy(&cy, cal + 84, 8);
    }
    const cv::Matx44f local = transform(scanInfo, scanInfoSize, 7);
    const cv::Matx44f world = pose * local;
    for (int v = static_cast<int>(height * 0.46); v < static_cast<int>(height * 0.95); v += 2) {
      for (int u = static_cast<int>(width * 0.06); u < static_cast<int>(width * 0.94); u += 2) {
        const cv::Vec3b color = hsv.at<cv::Vec3b>(v, u);
        const bool white = color[1] < 72 && color[2] > 155;
        const bool yellow = color[0] >= 5 && color[0] <= 35 && color[1] > 65 && color[2] > 90;
        if (!white && !yellow) continue;
        const uint16_t mm = depth.at<uint16_t>(v, u);
        if (mm < 400 || mm > 9000) continue;
        const float z = mm * 0.001f;
        const cv::Vec4f optical(static_cast<float>((u - cx) * z / fx),
                                static_cast<float>((v - cy) * z / fy), z, 1.0f);
        const cv::Vec4f p = world * optical;
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) continue;
        // Only painted surfaces close to the flat-road datum.  This rejects
        // white building/fence pixels while retaining markings and curb paint.
        if (p[2] < -0.18f || p[2] > 0.28f) continue;
        points.push_back(Point{p[0], p[1], p[2], static_cast<uint8_t>(yellow ? 2 : 1), id});
      }
    }
    ++nodes;
  }
  sqlite3_finalize(stmt); sqlite3_close(db);
  std::ofstream output(argv[5], std::ios::binary);
  if (!output) return 5;
  output << "ply\nformat binary_little_endian 1.0\n"
         << "element vertex " << points.size() << "\n"
         << "property float x\nproperty float y\nproperty float z\n"
         << "property uchar type\nproperty int node\nend_header\n";
  for (const auto &p : points) {
    output.write(reinterpret_cast<const char *>(&p.x), sizeof(float));
    output.write(reinterpret_cast<const char *>(&p.y), sizeof(float));
    output.write(reinterpret_cast<const char *>(&p.z), sizeof(float));
    output.write(reinterpret_cast<const char *>(&p.type), sizeof(uint8_t));
    output.write(reinterpret_cast<const char *>(&p.node), sizeof(int32_t));
  }
  std::cout << "nodes=" << nodes << " points=" << points.size() << " output=" << argv[5] << "\n";
  return 0;
}
