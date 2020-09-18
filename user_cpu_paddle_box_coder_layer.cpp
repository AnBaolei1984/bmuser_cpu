#include <string>
#include <iostream>
#include <cmath>
#include "user_cpu_paddle_box_coder_layer.h"

namespace usercpu {

void cpu_paddle_box_coder_layer::encodeCenterSize() {
  int64_t row = input_shapes_[1][0];
  int64_t col = input_shapes_[0][0];
  int64_t len = input_shapes_[0][1];

  float* output = output_tensors_[0];
  float* target_box_data = input_tensors_[1];
  float* prior_box_data = input_tensors_[0];
  for (int64_t i = 0; i < row; ++i) {
    for (int64_t j = 0; j < col; ++j) {
      int64_t offset = i * col * len + j * len;
      float prior_box_width = prior_box_data[j * len + 2] -
                              prior_box_data[j * len] + (normalized_ == false);
      float prior_box_height = prior_box_data[j * len + 3] -
                               prior_box_data[j * len + 1] +
                               (normalized_ == false);
      float prior_box_center_x = prior_box_data[j * len] + prior_box_width / 2;
      float prior_box_center_y =
          prior_box_data[j * len + 1] + prior_box_height / 2;

      float target_box_center_x =
          (target_box_data[i * len + 2] + target_box_data[i * len]) / 2;
      float target_box_center_y =
          (target_box_data[i * len + 3] + target_box_data[i * len + 1]) / 2;
      float target_box_width = target_box_data[i * len + 2] -
                            target_box_data[i * len] + (normalized_ == false);
      float target_box_height = target_box_data[i * len + 3] -
                            target_box_data[i * len + 1] +
                            (normalized_ == false);
      output[offset] =
          (target_box_center_x - prior_box_center_x) / prior_box_width;
      output[offset + 1] =
          (target_box_center_y - prior_box_center_y) / prior_box_height;
      output[offset + 2] =
          std::log(std::fabs(target_box_width / prior_box_width));
      output[offset + 3] =
          std::log(std::fabs(target_box_height / prior_box_height));
    }
  }

  const float* prior_box_var_data = input_tensors_[2];
  for (int64_t i = 0; i < row; ++i) {
    for (int64_t j = 0; j < col; ++j) {
      for (int k = 0; k < 4; ++k) {
        int64_t offset = i * col * len + j * len;
        int64_t prior_var_offset = j * len;
        output[offset + k] /= prior_box_var_data[prior_var_offset + k];
      }
    }
  }
}

void cpu_paddle_box_coder_layer::decodeCenterSize() {
  int64_t row = input_shapes_[1][0];
  int64_t col = input_shapes_[1][1];
  int64_t len = input_shapes_[1][2];
  float* output = output_tensors_[0];
  float* target_box_data = input_tensors_[1];
  float* prior_box_data = input_tensors_[0];
  for (int64_t i = 0; i < row; ++i) {
    for (int64_t j = 0; j < col; ++j) {
      float var_data[4] = {1., 1., 1., 1.};
      float* var_ptr = var_data;
      int64_t offset = i * col * len + j * len;
      int64_t prior_box_offset = axis_ == 0 ? j * len : i * len;

      float prior_box_width = prior_box_data[prior_box_offset + 2] -
                              prior_box_data[prior_box_offset] +
                              (normalized_ == false);
      float prior_box_height = prior_box_data[prior_box_offset + 3] -
                               prior_box_data[prior_box_offset + 1] +
                               (normalized_ == false);
      float prior_box_center_x =
          prior_box_data[prior_box_offset] + prior_box_width / 2;
      float prior_box_center_y =
          prior_box_data[prior_box_offset + 1] + prior_box_height / 2;
      float target_box_center_x = 0, target_box_center_y = 0;
      float target_box_width = 0, target_box_height = 0;
      int64_t prior_var_offset = axis_ == 0 ? j * len : i * len;
      if (var_size_ == 2) {
        memcpy(var_ptr,
                    input_tensors_[2] + prior_var_offset,
                    4 * sizeof(float));
      } else if (var_size_ == 1) {
        var_ptr = reinterpret_cast<float*>(variance_);
      }
      float box_var_x = *var_ptr;
      float box_var_y = *(var_ptr + 1);
      float box_var_w = *(var_ptr + 2);
      float box_var_h = *(var_ptr + 3);

      target_box_center_x =
          box_var_x * target_box_data[offset] * prior_box_width +
          prior_box_center_x;
      target_box_center_y =
          box_var_y * target_box_data[offset + 1] * prior_box_height +
          prior_box_center_y;
      target_box_width =
          std::exp(box_var_w * target_box_data[offset + 2]) * prior_box_width;
      target_box_height =
          std::exp(box_var_h * target_box_data[offset + 3]) * prior_box_height;
      output[offset] = target_box_center_x - target_box_width / 2;
      output[offset + 1] = target_box_center_y - target_box_height / 2;
      output[offset + 2] =
          target_box_center_x + target_box_width / 2 - (normalized_ == false);
      output[offset + 3] =
          target_box_center_y + target_box_height / 2 - (normalized_ == false);
    }
  }
}

int cpu_paddle_box_coder_layer::process(void *param) {
  setParam(param);
  if (code_type_ == 0) {
    encodeCenterSize();
  } else {
    decodeCenterSize();
  }
  return 0;
}

void cpu_paddle_box_coder_layer::setParam(void *param) {
  user_cpu_box_coder_param_t *box_coder_param =
                static_cast<user_cpu_box_coder_param_t*>(param);
  variance_len_ = box_coder_param->variance_len;
  USER_ASSERT(variance_len_ < 2000);
  memset(variance_, 0, sizeof(float) * 2000);
  memcpy(variance_, box_coder_param->variance, variance_len_ * sizeof(float));
  code_type_ = box_coder_param->code_type;
  axis_ = box_coder_param->axis;
  normalized_ = box_coder_param->normalized;
  var_size_ = 2;
}

int cpu_paddle_box_coder_layer::reshape(
          void* param,
          const vector<vector<int>>& input_shapes,
          vector<vector<int>>& output_shapes) {
  return 0;
}

/* must register user layer
 * in macro cpu_test##_layer  == class cpu_test_layer 
 * */

REGISTER_USER_CPULAYER_CLASS(USER_PADDLE_BOX_CODER, cpu_paddle_box_coder)

}
