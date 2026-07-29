// Dedicated large-observation binding for the VQ2 informed visual policy.
// Keeping this separate preserves every historical 32-float drone_race ABI.
#define DRONE_RACE_OBS_SIZE DRONE_RACE_VISUAL_OBS_SIZE
#define OBS_TENSOR_T FloatTensor
#include "../drone_race/binding.c"
