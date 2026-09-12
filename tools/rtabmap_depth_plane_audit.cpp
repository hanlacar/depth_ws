#include <rtabmap/core/rvl_codec.h>

#include <opencv2/core.hpp>
#include <sqlite3.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct PlaneResult {
  double slope_x = 0.0;
  double slope_y = 0.0;
  double intercept = 0.0;
  double rms = 0.0;
  int samples = 0;
  int inliers = 0;
};

static PlaneResult fitGround(const cv::Mat &depth, const cv::Matx44f &local,
                             double fx, double fy, double cx, double cy) {
  std::vector<cv::Point3d> points;
  const int step = 4;
  for (int v = static_cast<int>(depth.rows * 0.48); v < static_cast<int>(depth.rows * 0.93); v += step) {
    for (int u = static_cast<int>(depth.cols * 0.16); u < static_cast<int>(depth.cols * 0.84); u += step) {
      const uint16_t mm = depth.at<uint16_t>(v, u);
      if (mm < 500 || mm > 9000) continue;
      const double z = mm * 0.001;
      cv::Vec4f optical(static_cast<float>((u-cx)*z/fx),
                        static_cast<float>((v-cy)*z/fy),
                        static_cast<float>(z), 1.0f);
      cv::Vec4f base = local * optical;
      if (base[0] < 0.4f || base[0] > 8.5f || std::abs(base[1]) > 3.0f ||
          base[2] < -1.0f || base[2] > 1.2f) continue;
      points.emplace_back(base[0], base[1], base[2]);
    }
  }
  PlaneResult out;
  out.samples = static_cast<int>(points.size());
  if (points.size() < 40) return out;

  std::mt19937 rng(1937);
  std::uniform_int_distribution<size_t> pick(0, points.size()-1);
  std::vector<int> best;
  for (int iteration=0; iteration<700; ++iteration) {
    const auto &p=points[pick(rng)], &q=points[pick(rng)], &r=points[pick(rng)];
    cv::Vec3d v1(q.x-p.x,q.y-p.y,q.z-p.z), v2(r.x-p.x,r.y-p.y,r.z-p.z);
    cv::Vec3d n=v1.cross(v2);
    const double norm=cv::norm(n);
    if (norm < 1e-7) continue;
    n /= norm;
    if (std::abs(n[2]) < 0.75) continue;
    const double d=-(n[0]*p.x+n[1]*p.y+n[2]*p.z);
    std::vector<int> current;
    current.reserve(points.size());
    for (size_t i=0;i<points.size();++i) {
      const auto &x=points[i];
      if (std::abs(n[0]*x.x+n[1]*x.y+n[2]*x.z+d) < 0.045) current.push_back(static_cast<int>(i));
    }
    if (current.size()>best.size()) best.swap(current);
  }
  if (best.size()<30) return out;
  cv::Mat A(static_cast<int>(best.size()),3,CV_64F), b(static_cast<int>(best.size()),1,CV_64F);
  for (size_t j=0;j<best.size();++j) {
    const auto &p=points[best[j]];
    A.at<double>(j,0)=p.x; A.at<double>(j,1)=p.y; A.at<double>(j,2)=1.0;
    b.at<double>(j)=p.z;
  }
  cv::Mat coeff;
  cv::solve(A,b,coeff,cv::DECOMP_SVD);
  out.slope_x=coeff.at<double>(0); out.slope_y=coeff.at<double>(1); out.intercept=coeff.at<double>(2);
  double sum=0.0;
  for (int i : best) {
    const auto &p=points[i];
    const double e=p.z-(out.slope_x*p.x+out.slope_y*p.y+out.intercept);
    sum += e*e;
  }
  out.inliers=static_cast<int>(best.size());
  out.rms=std::sqrt(sum/best.size());
  return out;
}

int main(int argc,char **argv) {
  if (argc!=4) {
    std::cerr << "usage: rtabmap_depth_plane_audit DB START STOP\n";
    return 2;
  }
  sqlite3 *db=nullptr;
  if (sqlite3_open_v2(argv[1],&db,SQLITE_OPEN_READONLY,nullptr)!=SQLITE_OK) return 3;
  sqlite3_stmt *stmt=nullptr;
  const char *sql="SELECT Node.id,Data.depth,Data.calibration FROM Node JOIN Data USING(id) WHERE Node.id>=? AND Node.id<=? ORDER BY Node.id";
  sqlite3_prepare_v2(db,sql,-1,&stmt,nullptr);
  sqlite3_bind_int(stmt,1,std::stoi(argv[2])); sqlite3_bind_int(stmt,2,std::stoi(argv[3]));
  std::cout << "id,depth_pitch_deg,depth_roll_deg,intercept_m,slope_x,slope_y,rms_m,inliers,samples\n";
  while (sqlite3_step(stmt)==SQLITE_ROW) {
    const int id=sqlite3_column_int(stmt,0);
    const auto *bytes=static_cast<const unsigned char *>(sqlite3_column_blob(stmt,1));
    const int size=sqlite3_column_bytes(stmt,1);
    const auto *cal=static_cast<const unsigned char *>(sqlite3_column_blob(stmt,2));
    const int calSize=sqlite3_column_bytes(stmt,2);
    if (!bytes || !cal || calSize<48) continue;
    if (size < 16 || std::memcmp(bytes,"DEPTHRVL",8)!=0) continue;
    uint32_t width=0,height=0;
    std::memcpy(&width,bytes+8,4); std::memcpy(&height,bytes+12,4);
    if (width==0 || height==0 || width>4096 || height>4096 ||
        static_cast<uint64_t>(width)*height>16777216ULL) continue;
    cv::Mat depth(static_cast<int>(height),static_cast<int>(width),CV_16UC1);
    rtabmap::RvlCodec codec;
    codec.DecompressRVL(bytes+16,depth.ptr<uint16_t>(),static_cast<int>(width*height));
    double fx=388.0,fy=388.0,cx=317.4,cy=247.8;
    if (calSize>=284) {
      std::memcpy(&fx,cal+44,8); std::memcpy(&fy,cal+76,8);
      std::memcpy(&cx,cal+60,8); std::memcpy(&cy,cal+84,8);
    }
    cv::Matx44f local=cv::Matx44f::eye();
    const float *t=reinterpret_cast<const float *>(cal+calSize-48);
    for (int r=0;r<3;++r) for (int c=0;c<4;++c) local(r,c)=t[r*4+c];
    PlaneResult result=fitGround(depth,local,fx,fy,cx,cy);
    std::cout << id << ',' << std::atan(result.slope_x)*180.0/M_PI << ','
              << std::atan(result.slope_y)*180.0/M_PI << ',' << result.intercept << ','
              << result.slope_x << ',' << result.slope_y << ',' << result.rms << ','
              << result.inliers << ',' << result.samples << '\n';
  }
  sqlite3_finalize(stmt); sqlite3_close(db);
  return 0;
}
