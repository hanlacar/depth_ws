#include <rtabmap/core/rvl_codec.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sqlite3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

struct Record {
  cv::Mat image;
  cv::Mat depth;
  cv::Matx44d local = cv::Matx44d::eye();
  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};

static bool load(sqlite3 *db, int id, Record &out) {
  sqlite3_stmt *stmt = nullptr;
  const char *sql = "SELECT image,depth,calibration FROM Data WHERE id=?";
  if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
  sqlite3_bind_int(stmt, 1, id);
  if (sqlite3_step(stmt) != SQLITE_ROW) { sqlite3_finalize(stmt); return false; }
  const auto *imageBytes = static_cast<const unsigned char *>(sqlite3_column_blob(stmt, 0));
  const int imageSize = sqlite3_column_bytes(stmt, 0);
  const auto *depthBytes = static_cast<const unsigned char *>(sqlite3_column_blob(stmt, 1));
  const int depthSize = sqlite3_column_bytes(stmt, 1);
  const auto *cal = static_cast<const unsigned char *>(sqlite3_column_blob(stmt, 2));
  const int calSize = sqlite3_column_bytes(stmt, 2);
  if (!imageBytes || !depthBytes || !cal || imageSize <= 0 || depthSize < 16 || calSize < 48 ||
      std::memcmp(depthBytes, "DEPTHRVL", 8) != 0) {
    sqlite3_finalize(stmt); return false;
  }
  out.image = cv::imdecode(cv::Mat(1, imageSize, CV_8U, const_cast<unsigned char *>(imageBytes)),
                           cv::IMREAD_GRAYSCALE);
  uint32_t width = 0, height = 0;
  std::memcpy(&width, depthBytes + 8, 4); std::memcpy(&height, depthBytes + 12, 4);
  out.depth = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_16UC1);
  rtabmap::RvlCodec codec;
  codec.DecompressRVL(depthBytes + 16, out.depth.ptr<uint16_t>(), static_cast<int>(width * height));
  out.fx = 388.0; out.fy = 388.0; out.cx = 317.4; out.cy = 247.8;
  if (calSize >= 284) {
    std::memcpy(&out.fx, cal + 44, 8); std::memcpy(&out.fy, cal + 76, 8);
    std::memcpy(&out.cx, cal + 60, 8); std::memcpy(&out.cy, cal + 84, 8);
  }
  const float *values = reinterpret_cast<const float *>(cal + calSize - 48);
  for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) out.local(r, c) = values[r * 4 + c];
  sqlite3_finalize(stmt);
  auto orb = cv::ORB::create(3500, 1.2f, 8, 19, 0, 2, cv::ORB::HARRIS_SCORE, 31, 7);
  orb->detectAndCompute(out.image, cv::noArray(), out.keypoints, out.descriptors);
  return !out.image.empty() && !out.descriptors.empty();
}

static bool point(const Record &record, const cv::Point2f &pixel, cv::Point2d &result) {
  const int centerU = static_cast<int>(std::lround(pixel.x));
  const int centerV = static_cast<int>(std::lround(pixel.y));
  uint16_t best = 0;
  for (int radius = 0; radius <= 7 && best == 0; ++radius) {
    for (int dv = -radius; dv <= radius; ++dv) for (int du = -radius; du <= radius; ++du) {
      const int u = centerU + du, v = centerV + dv;
      if (u < 0 || v < 0 || u >= record.depth.cols || v >= record.depth.rows) continue;
      const uint16_t value = record.depth.at<uint16_t>(v, u);
      if (value >= 400 && value <= 10000 && (best == 0 || value < best)) best = value;
    }
  }
  if (!best) return false;
  const double z = best * 0.001;
  cv::Vec4d optical((pixel.x - record.cx) * z / record.fx,
                    (pixel.y - record.cy) * z / record.fy, z, 1.0);
  const cv::Vec4d base = record.local * optical;
  result = cv::Point2d(base[0], base[1]);
  return std::isfinite(result.x) && std::isfinite(result.y);
}

static void fit(const std::vector<cv::Point2d> &source, const std::vector<cv::Point2d> &target,
                const std::vector<int> &indices, double &theta, cv::Point2d &translation) {
  cv::Point2d sourceCenter(0, 0), targetCenter(0, 0);
  for (int i : indices) { sourceCenter += source[i]; targetCenter += target[i]; }
  sourceCenter *= 1.0 / indices.size(); targetCenter *= 1.0 / indices.size();
  double cross = 0.0, dot = 0.0;
  for (int i : indices) {
    const cv::Point2d s = source[i] - sourceCenter, t = target[i] - targetCenter;
    cross += s.x * t.y - s.y * t.x; dot += s.x * t.x + s.y * t.y;
  }
  theta = std::atan2(cross, dot);
  const double c = std::cos(theta), s = std::sin(theta);
  translation = cv::Point2d(targetCenter.x - (c * sourceCenter.x - s * sourceCenter.y),
                            targetCenter.y - (s * sourceCenter.x + c * sourceCenter.y));
}

static double residual(const cv::Point2d &source, const cv::Point2d &target,
                       double theta, const cv::Point2d &translation) {
  const double c = std::cos(theta), s = std::sin(theta);
  return cv::norm(cv::Point2d(c * source.x - s * source.y + translation.x,
                              s * source.x + c * source.y + translation.y) - target);
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: rtabmap_rgbd_orb_align DB BASE TARGET\n";
    return 2;
  }
  sqlite3 *db = nullptr;
  if (sqlite3_open_v2(argv[1], &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) return 3;
  Record base, target;
  if (!load(db, std::stoi(argv[2]), base) || !load(db, std::stoi(argv[3]), target)) return 4;
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> candidates;
  matcher.knnMatch(target.descriptors, base.descriptors, candidates, 2);
  std::vector<cv::DMatch> good;
  for (const auto &matches : candidates) {
    if (matches.size() == 2 && matches[0].distance < 0.72 * matches[1].distance) good.push_back(matches[0]);
  }
  std::vector<cv::Point2f> sourcePixels, targetPixels;
  for (const auto &match : good) {
    sourcePixels.push_back(target.keypoints[match.queryIdx].pt);
    targetPixels.push_back(base.keypoints[match.trainIdx].pt);
  }
  std::vector<cv::Point2d> sourcePoints, targetPoints;
  // Depth-valid road and curb features are often rejected by a fundamental
  // matrix dominated by distant building/sky features.  Keep all ratio-test
  // matches here; the metric SE(2) RANSAC below rejects geometric outliers.
  for (size_t i = 0; i < good.size(); ++i) {
    cv::Point2d source, targetPoint;
    if (point(target, sourcePixels[i], source) && point(base, targetPixels[i], targetPoint)) {
      sourcePoints.push_back(source); targetPoints.push_back(targetPoint);
    }
  }
  if (sourcePoints.size() < 3) {
    std::cout << "{\"accepted\":false,\"reason\":\"too_few_rgbd\",\"good\":" << good.size()
              << ",\"rgbd\":" << sourcePoints.size() << "}\n";
    return 0;
  }
  std::mt19937 rng(9127);
  std::uniform_int_distribution<int> pick(0, static_cast<int>(sourcePoints.size()) - 1);
  std::vector<int> best;
  for (int iteration = 0; iteration < 6000; ++iteration) {
    int a = pick(rng), b = pick(rng); if (a == b) continue;
    std::vector<int> sample{a, b}; double theta; cv::Point2d translation;
    fit(sourcePoints, targetPoints, sample, theta, translation);
    std::vector<int> inliers;
    for (size_t i = 0; i < sourcePoints.size(); ++i) {
      if (residual(sourcePoints[i], targetPoints[i], theta, translation) < 0.20) inliers.push_back(static_cast<int>(i));
    }
    if (inliers.size() > best.size()) best.swap(inliers);
  }
  if (best.size() < 3) {
    std::cout << "{\"accepted\":false,\"reason\":\"too_few_inliers\",\"good\":" << good.size()
              << ",\"rgbd\":" << sourcePoints.size() << ",\"inliers\":" << best.size() << "}\n";
    return 0;
  }
  double theta; cv::Point2d translation; fit(sourcePoints, targetPoints, best, theta, translation);
  std::vector<double> errors; double sum = 0.0;
  for (int i : best) { const double e = residual(sourcePoints[i], targetPoints[i], theta, translation); errors.push_back(e); sum += e * e; }
  std::sort(errors.begin(), errors.end());
  const double rms = std::sqrt(sum / best.size());
  const bool accepted = best.size() >= 8 && rms < 0.14;
  std::cout << "{\"accepted\":" << (accepted ? "true" : "false") << ",\"good\":" << good.size()
            << ",\"rgbd\":" << sourcePoints.size() << ",\"inliers\":" << best.size()
            << ",\"theta_deg\":" << theta * 180.0 / M_PI << ",\"tx\":" << translation.x
            << ",\"ty\":" << translation.y << ",\"rms_m\":" << rms
            << ",\"median_m\":" << errors[errors.size() / 2] << "}\n";
  sqlite3_close(db);
  return 0;
}
