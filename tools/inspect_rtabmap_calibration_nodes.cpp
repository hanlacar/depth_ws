#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/DBDriver.h>
#include <rtabmap/core/StereoCameraModel.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char ** argv)
{
    if(argc < 3)
    {
        std::cerr << "usage: inspect_rtabmap_calibration_nodes DB NODE [NODE...]\n";
        return 2;
    }
    std::unique_ptr<rtabmap::DBDriver> driver(rtabmap::DBDriver::create());
    if(!driver || !driver->openConnection(argv[1], false))
    {
        std::cerr << "cannot open database\n";
        return 1;
    }
    for(int i = 2; i < argc; ++i)
    {
        const int id = std::atoi(argv[i]);
        std::vector<rtabmap::CameraModel> mono;
        std::vector<rtabmap::StereoCameraModel> stereo;
        if(!driver->getCalibration(id, mono, stereo) || mono.empty())
        {
            std::cout << "node=" << id << " calibration=missing\n";
            continue;
        }
        const auto & model = mono.front();
        float x, y, z, roll, pitch, yaw;
        model.localTransform().getTranslationAndEulerAngles(
            x, y, z, roll, pitch, yaw);
        std::cout << "node=" << id
                  << " size=" << model.imageWidth() << "x" << model.imageHeight()
                  << " fx=" << model.fx() << " fy=" << model.fy()
                  << " cx=" << model.cx() << " cy=" << model.cy()
                  << " local_xyz=" << x << "," << y << "," << z
                  << " local_rpy=" << roll << "," << pitch << "," << yaw
                  << "\n";
    }
    driver->closeConnection(false);
    return 0;
}
