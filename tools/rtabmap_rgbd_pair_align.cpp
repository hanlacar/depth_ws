#include <rtabmap/core/rvl_codec.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <sqlite3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct Record {
  cv::Mat depth;
  cv::Matx44d local = cv::Matx44d::eye();
  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
  std::vector<cv::Point2f> pixels;
  std::vector<cv::Point2d> points;
  cv::Mat descriptors;
};

static cv::Matx33d alignNormalToUp(double slopeX, double slopeY) {
  cv::Vec3d n(-slopeX, -slopeY, 1.0);
  n /= cv::norm(n);
  const cv::Vec3d up(0.0, 0.0, 1.0);
  cv::Vec3d axis = n.cross(up);
  const double sine = cv::norm(axis);
  const double cosine = n.dot(up);
  if (sine < 1e-12) return cv::Matx33d::eye();
  axis /= sine;
  cv::Mat rotation;
  cv::Rodrigues(axis * std::atan2(sine, cosine), rotation);
  return rotation;
}

static bool load(sqlite3 *db, int id, double correctionSlopeX,
                 double correctionSlopeY, Record &out) {
  sqlite3_stmt *stmt = nullptr;
  const char *sql = "SELECT depth,calibration FROM Data WHERE id=?";
  if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;
  sqlite3_bind_int(stmt, 1, id);
  if (sqlite3_step(stmt) != SQLITE_ROW) { sqlite3_finalize(stmt); return false; }
  const auto *bytes = static_cast<const unsigned char *>(sqlite3_column_blob(stmt, 0));
  const int size = sqlite3_column_bytes(stmt, 0);
  const auto *cal = static_cast<const unsigned char *>(sqlite3_column_blob(stmt, 1));
  const int calSize = sqlite3_column_bytes(stmt, 1);
  if (!bytes || !cal || size < 16 || calSize < 48 || std::memcmp(bytes, "DEPTHRVL", 8) != 0) {
    sqlite3_finalize(stmt); return false;
  }
  uint32_t width=0, height=0;
  std::memcpy(&width, bytes+8, 4); std::memcpy(&height, bytes+12, 4);
  out.depth = cv::Mat(static_cast<int>(height), static_cast<int>(width), CV_16UC1);
  rtabmap::RvlCodec codec;
  codec.DecompressRVL(bytes+16, out.depth.ptr<uint16_t>(), static_cast<int>(width*height));
  out.fx=388.0; out.fy=388.0; out.cx=317.4; out.cy=247.8;
  if (calSize >= 284) {
    std::memcpy(&out.fx, cal+44, 8); std::memcpy(&out.fy, cal+76, 8);
    std::memcpy(&out.cx, cal+60, 8); std::memcpy(&out.cy, cal+84, 8);
  }
  const float *values = reinterpret_cast<const float *>(cal+calSize-48);
  for (int r=0;r<3;++r) for (int c=0;c<4;++c) out.local(r,c)=values[r*4+c];
  const cv::Matx33d correction = alignNormalToUp(correctionSlopeX, correctionSlopeY);
  cv::Matx33d oldRotation;
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) oldRotation(r,c)=out.local(r,c);
  const cv::Matx33d newRotation = correction*oldRotation;
  for (int r=0;r<3;++r) for (int c=0;c<3;++c) out.local(r,c)=newRotation(r,c);
  sqlite3_finalize(stmt);

  stmt = nullptr;
  if (sqlite3_prepare_v2(db, "SELECT pos_x,pos_y,depth_x,depth_y,depth_z,descriptor FROM Feature WHERE node_id=? ORDER BY rowid", -1, &stmt, nullptr) != SQLITE_OK) return false;
  sqlite3_bind_int(stmt, 1, id);
  std::vector<unsigned char> descriptors;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const void *blob = sqlite3_column_blob(stmt, 5);
    const int blobSize = sqlite3_column_bytes(stmt, 5);
    if (!blob || blobSize != 32) continue;
    out.pixels.emplace_back(static_cast<float>(sqlite3_column_double(stmt, 0)),
                            static_cast<float>(sqlite3_column_double(stmt, 1)));
    cv::Vec3d original(sqlite3_column_double(stmt, 2), sqlite3_column_double(stmt, 3),
                       sqlite3_column_double(stmt, 4));
    cv::Vec3d cameraOrigin(out.local(0,3),out.local(1,3),out.local(2,3));
    cv::Vec3d corrected=correction*(original-cameraOrigin)+cameraOrigin;
    out.points.emplace_back(corrected[0],corrected[1]);
    const auto *d = static_cast<const unsigned char *>(blob);
    descriptors.insert(descriptors.end(), d, d+32);
  }
  sqlite3_finalize(stmt);
  if (out.pixels.empty()) return false;
  out.descriptors = cv::Mat(static_cast<int>(out.pixels.size()), 32, CV_8U);
  std::memcpy(out.descriptors.data, descriptors.data(), descriptors.size());
  return true;
}

static bool point(const Record &r, const cv::Point2f &pixel, cv::Point2d &result) {
  const int centerU = static_cast<int>(std::lround(pixel.x));
  const int centerV = static_cast<int>(std::lround(pixel.y));
  uint16_t best = 0;
  for (int radius=0; radius<=2 && best==0; ++radius) {
    for (int dv=-radius; dv<=radius; ++dv) for (int du=-radius; du<=radius; ++du) {
      const int u=centerU+du, v=centerV+dv;
      if (u<0 || v<0 || u>=r.depth.cols || v>=r.depth.rows) continue;
      const uint16_t value=r.depth.at<uint16_t>(v,u);
      if (value>=300 && value<=10000 && (best==0 || value<best)) best=value;
    }
  }
  if (!best) return false;
  const double z=best*0.001;
  cv::Vec4d optical((pixel.x-r.cx)*z/r.fx, (pixel.y-r.cy)*z/r.fy, z, 1.0);
  const cv::Vec4d base=r.local*optical;
  result=cv::Point2d(base[0],base[1]);
  return std::isfinite(result.x) && std::isfinite(result.y);
}

static void fit(const std::vector<cv::Point2d> &source, const std::vector<cv::Point2d> &target,
                const std::vector<int> &indices, double &theta, cv::Point2d &translation) {
  cv::Point2d sc(0,0), tc(0,0);
  for (int i:indices) { sc+=source[i]; tc+=target[i]; }
  sc*=1.0/indices.size(); tc*=1.0/indices.size();
  double cross=0.0, dot=0.0;
  for (int i:indices) {
    const cv::Point2d s=source[i]-sc, t=target[i]-tc;
    cross += s.x*t.y-s.y*t.x; dot += s.x*t.x+s.y*t.y;
  }
  theta=std::atan2(cross,dot);
  const double c=std::cos(theta), s=std::sin(theta);
  translation=cv::Point2d(tc.x-(c*sc.x-s*sc.y), tc.y-(s*sc.x+c*sc.y));
}

static double residual(const cv::Point2d &source, const cv::Point2d &target,
                       double theta, const cv::Point2d &translation) {
  const double c=std::cos(theta), s=std::sin(theta);
  return cv::norm(cv::Point2d(c*source.x-s*source.y+translation.x,
                              s*source.x+c*source.y+translation.y)-target);
}

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "usage: rtabmap_rgbd_pair_align DB BASE TARGET BASE_SLOPE_X BASE_SLOPE_Y TARGET_SLOPE_X TARGET_SLOPE_Y\n";
    return 2;
  }
  sqlite3 *db=nullptr;
  if (sqlite3_open_v2(argv[1],&db,SQLITE_OPEN_READONLY,nullptr)!=SQLITE_OK) return 3;
  Record base,target;
  if (!load(db,std::stoi(argv[2]),std::stod(argv[4]),std::stod(argv[5]),base) ||
      !load(db,std::stoi(argv[3]),std::stod(argv[6]),std::stod(argv[7]),target)) return 4;
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> candidates;
  matcher.knnMatch(target.descriptors,base.descriptors,candidates,2);
  std::vector<cv::DMatch> good;
  for (const auto &m:candidates) if (m.size()==2 && m[0].distance < 0.72*m[1].distance) good.push_back(m[0]);
  std::vector<cv::Point2f> sourcePixels,targetPixels;
  for (const auto &m:good) { sourcePixels.push_back(target.pixels[m.queryIdx]); targetPixels.push_back(base.pixels[m.trainIdx]); }
  std::vector<unsigned char> fundamentalMask(good.size(),1);
  if (good.size()>=8) cv::findFundamentalMat(sourcePixels,targetPixels,cv::FM_RANSAC,1.5,0.999,fundamentalMask);
  std::vector<cv::Point2d> source,targetPoints;
  for (size_t i=0;i<good.size();++i) if (fundamentalMask[i]) {
    const cv::Point2d s=target.points[good[i].queryIdx], t=base.points[good[i].trainIdx];
    if (std::isfinite(s.x) && std::isfinite(s.y) && std::isfinite(t.x) && std::isfinite(t.y)) {
      source.push_back(s); targetPoints.push_back(t);
    }
  }
  if (source.size()<3) { std::cout << "{\"accepted\":false,\"reason\":\"too_few_rgbd\",\"rgbd\":" << source.size() << "}\n"; return 0; }
  std::mt19937 rng(8137);
  std::uniform_int_distribution<int> pick(0,static_cast<int>(source.size())-1);
  std::vector<int> best;
  for (int iteration=0;iteration<5000;++iteration) {
    int a=pick(rng),b=pick(rng); if(a==b) continue;
    std::vector<int> sample{a,b}; double theta; cv::Point2d translation;
    fit(source,targetPoints,sample,theta,translation);
    std::vector<int> inliers;
    for(size_t i=0;i<source.size();++i) if(residual(source[i],targetPoints[i],theta,translation)<0.18) inliers.push_back(static_cast<int>(i));
    if(inliers.size()>best.size()) best.swap(inliers);
  }
  if(best.size()<3) { std::cout << "{\"accepted\":false,\"reason\":\"too_few_inliers\",\"rgbd\":" << source.size() << ",\"inliers\":" << best.size() << "}\n"; return 0; }
  double theta; cv::Point2d translation; fit(source,targetPoints,best,theta,translation);
  std::vector<double> errors; double sum=0.0;
  for(int i:best) { const double e=residual(source[i],targetPoints[i],theta,translation); errors.push_back(e); sum+=e*e; }
  std::sort(errors.begin(),errors.end());
  std::cout << "{\"accepted\":" << (best.size()>=6 && std::sqrt(sum/best.size())<0.14?"true":"false")
            << ",\"good\":" << good.size() << ",\"rgbd\":" << source.size()
            << ",\"inliers\":" << best.size() << ",\"theta_deg\":" << theta*180.0/M_PI
            << ",\"tx\":" << translation.x << ",\"ty\":" << translation.y
            << ",\"rms_m\":" << std::sqrt(sum/best.size())
            << ",\"median_m\":" << errors[errors.size()/2] << "}\n";
  sqlite3_close(db);
  return 0;
}
