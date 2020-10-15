#include <string>
#include <iostream>
#include <cmath>
#include "user_cpu_paddle_prior_box_layer.h"
#include <sys/time.h>
namespace usercpu {

void cpu_paddle_prior_box_layer::ExpandAspectRatios() {
  constexpr float epsilon = 1e-6;
  expand_aspect_ratios_.clear();
  expand_aspect_ratios_.push_back(1.0f);

  for (size_t i = 0; i < prior_box_params_->aspect_ratios_len; ++i) {
    float ar = prior_box_params_->aspect_ratios[i];
    bool already_exist = false;
    for (size_t j = 0; j < expand_aspect_ratios_.size(); ++j) {
      if (fabs(ar - expand_aspect_ratios_[j]) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      expand_aspect_ratios_.push_back(ar);
      if (prior_box_params_->flip) {
        expand_aspect_ratios_.push_back(1.0f / ar);
      }
    }
  }
}

int cpu_paddle_prior_box_layer::process(void *param) {
#ifdef CALC_TIME  
  struct timeval start, stop;
  gettimeofday(&start, 0);
#endif
  setParam(param);
  ExpandAspectRatios();

  int img_width = input_shapes_[1][3];
  int img_height = input_shapes_[1][2];
  int feature_width = input_shapes_[0][3];
  int feature_height = input_shapes_[0][2];
  float step_width, step_height;
  if (prior_box_params_->step_w == 0.f || prior_box_params_->step_h == 0.f) {
    step_width = static_cast<float>(img_width) / feature_width;
    step_height = static_cast<float>(img_height) / feature_height;
  } else {
    step_width = prior_box_params_->step_w;
    step_height = prior_box_params_->step_h;
  }
  int num_priors = prior_box_params_->aspect_ratios_len *
                                     prior_box_params_->min_sizes_len;
  if (prior_box_params_->max_sizes_len > 0) {
    num_priors += prior_box_params_->max_sizes_len;
  }

  float* output = output_tensors_[0];
  float* b_t = output;
  for (int h = 0; h < feature_height; ++h) {
    for (int w = 0; w < feature_width; ++w) {
      float center_x = (w + prior_box_params_->offset) * step_width;
      float center_y = (h + prior_box_params_->offset) * step_height;
      float box_width, box_height;
      for (size_t s = 0; s < prior_box_params_->min_sizes_len; ++s) {
        auto min_size = prior_box_params_->min_sizes[s];
        if (prior_box_params_->min_max_aspect_ratios_order) {
          box_width = box_height = min_size / 2.;
          b_t[0] = (center_x - box_width) / img_width;
          b_t[1] = (center_y - box_height) / img_height;
          b_t[2] = (center_x + box_width) / img_width;
          b_t[3] = (center_y + box_height) / img_height;
          b_t += 4;
          if (prior_box_params_->max_sizes_len > 0) {
            auto max_size = prior_box_params_->max_sizes[s];
            // square prior with size sqrt(minSize * maxSize)
            box_width = box_height = sqrt(min_size * max_size) / 2.;
            b_t[0] = (center_x - box_width) / img_width;
            b_t[1] = (center_y - box_height) / img_height;
            b_t[2] = (center_x + box_width) / img_width;
            b_t[3] = (center_y + box_height) / img_height;
            b_t += 4;
          }
          // priors with different aspect ratios
          for (size_t r = 0; r < expand_aspect_ratios_.size(); ++r) {
            float ar = expand_aspect_ratios_[r];
            if (fabs(ar - 1.) < 1e-6) {
              continue;
            }
            box_width = min_size * sqrt(ar) / 2.;
            box_height = min_size / sqrt(ar) / 2.;
            b_t[0] = (center_x - box_width) / img_width;
            b_t[1] = (center_y - box_height) / img_height;
            b_t[2] = (center_x + box_width) / img_width;
            b_t[3] = (center_y + box_height) / img_height;
            b_t += 4;
          }
        } else {
          // priors with different aspect ratios
          for (size_t r = 0; r < expand_aspect_ratios_.size(); ++r) {
            float ar = expand_aspect_ratios_[r];
            box_width = min_size * sqrt(ar) / 2.;
            box_height = min_size / sqrt(ar) / 2.;
            b_t[0] = (center_x - box_width) / img_width;
            b_t[1] = (center_y - box_height) / img_height;
            b_t[2] = (center_x + box_width) / img_width;
            b_t[3] = (center_y + box_height) / img_height;
            b_t += 4;
          }
          if (prior_box_params_->max_sizes_len > 0) {
            auto max_size = prior_box_params_->max_sizes[s];
            // square prior with size sqrt(minSize * maxSize)
            box_width = box_height = sqrt(min_size * max_size) / 2.;
            b_t[0] = (center_x - box_width) / img_width;
            b_t[1] = (center_y - box_height) / img_height;
            b_t[2] = (center_x + box_width) / img_width;
            b_t[3] = (center_y + box_height) / img_height;
            b_t += 4;
          }
        }
      }
    }
  }
  int32_t channel_size = feature_height * feature_width * num_priors * 4;
  if (prior_box_params_->clip) {
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
          ptr[count] = prior_box_params_->variances[j];
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

void cpu_paddle_prior_box_layer::setParam(void *param) {
  prior_box_params_ = static_cast<user_cpu_prior_box_param_t*>(param);
  USER_ASSERT(prior_box_params_->aspect_ratios_len < 20);
  USER_ASSERT(prior_box_params_->max_sizes_len < 20);
  USER_ASSERT(prior_box_params_->min_sizes_len < 20);
  USER_ASSERT(prior_box_params_->variances_len < 20);
}

int cpu_paddle_prior_box_layer::reshape(
          void* param,
          const vector<vector<int>>& input_shapes,
          vector<vector<int>>& output_shapes) {
  return 0;
}

int cpu_paddle_prior_box_layer::dtype(
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

REGISTER_USER_CPULAYER_CLASS(USER_PADDLE_PRIOR_BOX, cpu_paddle_prior_box)

}
