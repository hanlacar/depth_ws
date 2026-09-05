std_msgs/Header header

uint8 STATE_INVALID=0
uint8 STATE_VALID=1
uint8 STATE_DEGRADED=2
uint8 STATE_INACTIVE=3

uint8 OWNER_NONE=0
uint8 OWNER_CAMERA=1
uint8 OWNER_GPS=2

uint16 image_width
uint16 image_height
bool path_valid
float32 path_confidence
uint8 path_state
uint8 ownership
race_interfaces/ImagePathPoint[] points
