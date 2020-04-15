#include <string>
#include <iostream>
#include <cmath>
#include <sys/time.h>
#include "user_cpu_paddle_adaptive_pool_layer.h"

namespace usercpu {

int cpu_paddle_adaptive_pool_layer::process(void *param) {
#ifdef CALC_TIME  
  struct timeval start, stop;
  gettimeofday(&start, 0);
#endif
  setParam(param);
  const int batch_size = input_shapes_[0][0];
  const int input_height = input_shapes_[0][2];
  const int input_width = input_shapes_[0][3];
  const int output_channels = output_shapes_[0][0][1];
  const int output_height = output_shapes_[0][0][2];
  const int output_width = output_shapes_[0][0][3];
  const int input_stride = input_height * input_width;
  const int output_stride = output_height * output_width;
  const float* input_data = input_tensors_[0];
  float* output_data = output_tensors_[0];
  int hstart, hend;
  int wstart, wend;
  for (int i = 0; i < batch_size; i++) {
    for (int c = 0; c < output_channels; ++c) {
      for (int ph = 0; ph < output_height; ++ph) {
        hstart = AdaptStartIndex(ph, input_height, output_height);
        hend = AdaptEndIndex(ph, input_height, output_height);
         for (int pw = 0; pw < output_width; ++pw) {
           wstart = AdaptStartIndex(pw, input_width, output_width);
           wend = AdaptEndIndex(pw, input_width, output_width);
           float ele = pool_process_->initial();
           for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                pool_process_->compute(input_data[h * input_width + w], &ele);
              }
           }
           int pool_size = (hend - hstart) * (wend - wstart);
           pool_process_->finalize(static_cast<float>(pool_size), &ele);
           output_data[ph * output_width + pw] = ele;
        }
      }
      input_data += input_stride;
      output_data += output_stride;
    }
  }

#ifdef CALC_TIME
  gettimeofday(&stop, 0);
  float timeuse = 1000000 * (stop.tv_sec - start.tv_sec) + stop.tv_usec - start.tv_usec;
  timeuse /= 1000;
  printf("**************cost time is %f ms\n", timeuse);
#endif
  return 0;
}

void cpu_paddle_adaptive_pool_layer::setParam(void *param) {
  user_cpu_adaptive_pool_param_t * adaptive_pool_param =
                static_cast<user_cpu_adaptive_pool_param_t*>(param);
  pool_process_ =
    adaptive_pool_param->is_avg ?
    static_cast<PoolProc*>(new AvgPool()) : 
    static_cast<PoolProc*>(new MaxPool());
}

int cpu_paddle_adaptive_pool_layer::reshape(
          void* param,
          const vector<vector<int>>& input_shapes,
          vector<vector<int>>& output_shapes) {
  return 0;
}

/* must register user layer
 * in macro cpu_test##_layer  == class cpu_test_layer 
 * */

REGISTER_USER_CPULAYER_CLASS(USER_PADDLE_ADAPTIVE_POOL,
                                  cpu_paddle_adaptive_pool)

}
