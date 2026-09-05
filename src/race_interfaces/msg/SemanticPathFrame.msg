std_msgs/Header header
uint16 image_width
uint16 image_height

# Alternating flattened start,end-exclusive pairs for non-zero binary runs.
uint32[] road_rle
uint32[] lane_rle
uint32[] words_rle
uint32[] stop_line_rle
uint32[] c_line_rle

# Shared navigation refinement.  The legacy fields above remain populated:
# road_rle/lane_rle/words_rle/stop_line_rle/c_line_rle carry the refined
# drivable/lane/marking results so existing consumers continue to work.
uint32[] white_line_rle
uint32[] yellow_line_rle
uint32[] unknown_line_rle
uint32[] road_marking_unknown_rle
uint32[] restored_markings_rle
uint32[] raw_road_rle
float32 stop_line_confidence
float32 crosswalk_confidence
string refinement_diagnostics_json

uint32 uncompressed_bytes
uint32 encoded_bytes
