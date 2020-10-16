#include <string>
#include <iostream>
#include <cmath>
#include "user_cpu_paddle_density_prior_box_layer.h"
#include <sys/time.h>
namespace usercpu {

int cpu_paddle_density_prior_box_layer::process(void *param) {
#ifdef CALC_TIME  
  struct timeval start, stop;
  gettimeofday(&start, 0);
#endif
  setParam(param);

  int img_width = input_shapes_[1][3];
  int img_height = input_shapes_[1][2];
  int feature_width = input_shapes_[0][3];
  int feature_height = input_shapes_[0][2];
  float step_width, step_height;
  if (density_prior_box_params_->step_w == 0.f || density_prior_box_params_->step_h == 0.f) {
    step_width = static_cast<float>(img_width) / feature_width;
    step_height = static_cast<float>(img_height) / feature_height;
  } else {
    step_width = density_prior_box_params_->step_w;
    step_height = density_prior_box_params_->step_h;
  }

  int num_priors = 0;
  for (size_t i = 0; i < density_prior_box_params_->densities_len; ++i) {
    num_priors += (density_prior_box_params_->fixed_ratios_len) *
                      (pow(density_prior_box_params_->densities[i], 2));
  }
  density_prior_box_params_->prior_num = num_priors;
  int step_average = static_cast<int>((step_width + step_height) * 0.5);
  std::vector<float> sqrt_fixed_ratios;
  for (size_t i = 0; i < density_prior_box_params_->fixed_ratios_len; i++) {
    sqrt_fixed_ratios.push_back(sqrt(density_prior_box_params_->fixed_ratios[i]));
  }
  float* output = output_tensors_[0];
  float* b_t = output;
  for (int h = 0; h < feature_height; ++h) {
    for (int w = 0; w < feature_width; ++w) {
      float center_x = (w + density_prior_box_params_->offset) * step_width;
      float center_y = (h + density_prior_box_params_->offset) * step_height;
      for (size_t s = 0; s < density_prior_box_params_->fixed_sizes_len; ++s) {
        auto fixed_size = density_prior_box_params_->fixed_sizes[s];
        int density = density_prior_box_params_->densities[s];
        int shift = step_average / density;
        // Generate density prior boxes with fixed ratios.
        for (size_t r = 0; r < density_prior_box_params_->fixed_ratios_len; ++r) {
          float box_width_ratio = fixed_size * sqrt_fixed_ratios[r];
          float box_height_ratio = fixed_size / sqrt_fixed_ratios[r];
          float density_center_x = center_x - step_average / 2. + shift / 2.;
          float density_center_y = center_y - step_average / 2. + shift / 2.;
          for (int di = 0; di < density; ++di) {
            for (int dj = 0; dj < density; ++dj) {
              float center_x_temp = density_center_x + dj * shift;
              float center_y_temp = density_center_y + di * shift;
              b_t[0] = std::max(
                  (center_x_temp - box_width_ratio / 2.) / img_width, 0.);
              b_t[1] = std::max(
                  (center_y_temp - box_height_ratio / 2.) / img_height, 0.);
              b_t[2] = std::min(
                  (center_x_temp + box_width_ratio / 2.) / img_width, 1.);
              b_t[3] = std::min(
                  (center_y_temp + box_height_ratio / 2.) / img_height, 1.);
              b_t += 4;
            }
          }
        }
      }
    }
  }
  int32_t channel_size = feature_height * feature_width * num_priors * 4;
  if (density_prior_box_params_->clip) {
    for (int32_t d = 0; d < channel_size; ++d) {
      output[d] = std::min(std::max(output[d], 0.f), 1.f);
    }
  }
  float* ptr = output + channel_size;
  int count = 0;
  for (int32_t h = 0; h < feature_height; ++h) {
    for (int32_t w = 0; w < feature_width; ++w) {
      for (int32_t i = 0; i < num_priors; ++i) {
        for (int j = 0; j < 4; ++j) {
          ptr[count] = density_prior_box_params_->variances[j];
          ++count;
        }
      }
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

void cpu_paddle_density_prior_box_layer::setParam(void *param) {
  density_prior_box_params_ =
       static_cast<user_cpu_density_prior_box_param_t*>(param);
  USER_ASSERT(density_prior_box_params_->densities_len < 20);
  USER_ASSERT(density_prior_box_params_->fixed_ratios_len < 20);
  USER_ASSERT(density_prior_box_params_->fixed_sizes_len < 20);
  USER_ASSERT(density_prior_box_params_->variances_len < 20);
}

int cpu_paddle_density_prior_box_layer::reshape(
          void* param,
          const vector<vector<int>>& input_shapes,
          vector<vector<int>>& output_shapes) {
  return 0;
}

int cpu_paddle_density_prior_box_layer::dtype(
          void* param,
          const vector<int>& input_dtypes,
          vector<int>& output_dtypes) {
    USER_ASSERT(input_dtypes.size() == 1);
    output_dtypes = {input_dtypes[0]};
    cout << " cpu exp dtype "<< endl;
    return 0;
}

/* must register user layer
 * in macro cpu_test##_layer  == class cpu_test_layer 
 * */

REGISTER_USER_CPULAYER_CLASS(USER_PADDLE_DENSITY_PRIOR_BOX,
                                     cpu_paddle_density_prior_box)

}
