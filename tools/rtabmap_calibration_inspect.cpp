#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/StereoCameraModel.h>

int main(int argc, char ** argv)
{
    if (argc != 2) {
        std::cerr << "usage: rtabmap_calibration_inspect DATABASE\n";
        return 2;
    }
    rtabmap::DBDriver * driver = rtabmap::DBDriver::create();
    if (!driver || !driver->openConnection(argv[1], false)) {
        std::cerr << "cannot open database\n";
        delete driver;
        return 1;
    }
    std::set<int> ids;
    driver->getAllNodeIds(ids, false, false, false);
    if (ids.empty()) {
        std::cerr << "no nodes\n";
        driver->closeConnection(false);
        delete driver;
        return 1;
    }
    std::vector<rtabmap::CameraModel> models;
    std::vector<rtabmap::StereoCameraModel> stereo_models;
    if (!driver->getCalibration(*ids.begin(), models, stereo_models)) {
        std::cerr << "no calibration\n";
        driver->closeConnection(false);
        delete driver;
        return 1;
    }
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "node=" << *ids.begin() << " mono_models=" << models.size()
              << " stereo_models=" << stereo_models.size() << "\n";
    for (std::size_t i = 0; i < models.size(); ++i) {
        const auto & m = models[i];
        float x, y, z, roll, pitch, yaw;
        m.localTransform().getTranslationAndEulerAngles(x, y, z, roll, pitch, yaw);
        std::cout << "mono[" << i << "] name=" << m.name()
                  << " size=" << m.imageWidth() << "x" << m.imageHeight()
                  << " fx=" << m.fx() << " fy=" << m.fy()
                  << " cx=" << m.cx() << " cy=" << m.cy()
                  << " Tx=" << m.Tx()
                  << " local_xyz=" << x << "," << y << "," << z
                  << " local_rpy=" << roll << "," << pitch << "," << yaw
                  << "\n";
    }
    driver->closeConnection(false);
    delete driver;
    return 0;
}
