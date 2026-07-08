#include "../common/common_helper.cuh"
#include "private_helper.cuh"

/// includes needed

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <iostream>
#include <cmath>


/*
mean of the whole data , wrt to col --- row wise
*/