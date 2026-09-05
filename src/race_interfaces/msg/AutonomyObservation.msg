std_msgs/Header header

uint8 TRAFFIC_UNKNOWN=0
uint8 TRAFFIC_RED=1
uint8 TRAFFIC_YELLOW=2
uint8 TRAFFIC_GREEN=3

bool perception_valid
bool imu_valid
bool speed_valid
bool stop_detected
uint8 traffic_light

float32 pitch_deg
float32 roll_deg
float32 yaw_deg
float32 speed_kph
