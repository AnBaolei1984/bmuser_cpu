#include <string>
#include <iostream>
#include <cmath>
#include "user_cpu_paddle_yolo_box_layer.h"
#include <sys/time.h>
namespace usercpu {

float cpu_paddle_yolo_box_layer::sigmoid(float x) { 
  return 1.f / (1.f + expf(-x));
}

void cpu_paddle_yolo_box_layer::get_yolo_box(float* box,
                         const float* x,
                         const int* anchors,
                         int i,
                         int j,
                         int an_idx,
                         int grid_size,
                         int input_size,
                         int index,
                         int stride,
                         int img_height,
                         int img_width) {
  box[0] = (i + sigmoid(x[index])) * img_width / grid_size;
  box[1] = (j + sigmoid(x[index + stride])) * img_height / grid_size;
  box[2] = std::exp(x[index + 2 * stride]) * anchors[2 * an_idx] * img_width /
           input_size;
  box[3] = std::exp(x[index + 3 * stride]) * anchors[2 * an_idx + 1] *
           img_height / input_size;
}

int cpu_paddle_yolo_box_layer::get_entry_index(int batch,
                           int an_idx,
                           int hw_idx,
                           int an_num,
                           int an_stride,
                           int stride,
                           int entry) {
  return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

void cpu_paddle_yolo_box_layer::calc_detection_box(float* boxes,
                               float* box,
                               const int box_idx,
                               const int img_height,
                               const int img_width) {
  boxes[box_idx] = box[0] - box[2] / 2;
  boxes[box_idx + 1] = box[1] - box[3] / 2;
  boxes[box_idx + 2] = box[0] + box[2] / 2;
  boxes[box_idx + 3] = box[1] + box[3] / 2;

  boxes[box_idx] = boxes[box_idx] > 0 ? boxes[box_idx] : static_cast<float>(0);
  boxes[box_idx + 1] =
      boxes[box_idx + 1] > 0 ? boxes[box_idx + 1] : static_cast<float>(0);
  boxes[box_idx + 2] = boxes[box_idx + 2] < img_width - 1
                           ? boxes[box_idx + 2]
                           : static_cast<float>(img_width - 1);
  boxes[box_idx + 3] = boxes[box_idx + 3] < img_height - 1
                           ? boxes[box_idx + 3]
                           : static_cast<float>(img_height - 1);
}

void cpu_paddle_yolo_box_layer::calc_label_score(float* scores,
                             const float* input,
                             const int label_idx,
                             const int score_idx,
                             const int class_num,
                             const float conf,
                             const int stride) {
  for (int i = 0; i < class_num; i++) {
    scores[score_idx + i] = conf * sigmoid(input[label_idx + i * stride]);
  }
}

int cpu_paddle_yolo_box_layer::process(void *param) {
#ifdef CALC_TIME  
  struct timeval start, stop;
  gettimeofday(&start, 0);
#endif
  setParam(param);
  const int n = input_shapes_[0][0];
  const int h = input_shapes_[0][2];
  const int w = input_shapes_[0][3];
  int b_num = output_shapes_[0][0][1];
  const int an_num = (anchors_size_ >> 1);
  int X_size = downsample_ratio_ * h;
  const int stride = h * w;
  const int an_stride = (class_num_ + 5) * stride;
  auto anchors_data = anchors_;
  const float* X_data = input_tensors_[0];
  int* ImgSize_data = (int*)input_tensors_[1];
  float* Boxes_data = output_tensors_[0];
  float* Scores_data = output_tensors_[1];
  float box[4];
  memset(Boxes_data, 0, output_shapes_[0][0][0] * output_shapes_[0][0][1] * output_shapes_[0][0][2] * sizeof(float));
  memset(Scores_data, 0, output_shapes_[0][1][0] * output_shapes_[0][1][1] * output_shapes_[0][1][2] * sizeof(float)); 
  for (int i = 0; i < n; i++) {
    int img_height = static_cast<int>(ImgSize_data[2 * i]);
    int img_width = static_cast<int>(ImgSize_data[2 * i + 1]);

    for (int j = 0; j < an_num; j++) {
      for (int k = 0; k < h; k++) {
        for (int l = 0; l < w; l++) {
          int obj_idx =
              get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 4);
          float conf = sigmoid(X_data[obj_idx]);
          if (conf < conf_thresh_) {
            continue;
          }

          int box_idx =
              get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 0);
          get_yolo_box(box,
                       X_data,
                       anchors_data,
                       l,
                       k,
                       j,
                       h,
                       X_size,
                       box_idx,
                       stride,
                       img_height,
                       img_width);
          box_idx = (i * b_num + j * stride + k * w + l) * 4;
          calc_detection_box(Boxes_data, box, box_idx, img_height, img_width);

          int label_idx =
              get_entry_index(i, j, k * w + l, an_num, an_stride, stride, 5);
          int score_idx = (i * b_num + j * stride + k * w + l) * class_num_;
          calc_label_score(Scores_data,
                           X_data,
                           label_idx,
                           score_idx,
                           class_num_,
                           conf,
                           stride);
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

void cpu_paddle_yolo_box_layer::setParam(void *param) {
  user_cpu_yolo_box_param_t * yolo_box_param =
                static_cast<user_cpu_yolo_box_param_t*>(param);
  anchors_size_ = yolo_box_param->anchors_size;
  USER_ASSERT(anchors_size_ < 2000);
  memset(anchors_, 0, sizeof(int) * 2000);
  memcpy(anchors_, yolo_box_param->anchors, anchors_size_ * sizeof(int));
  conf_thresh_ = yolo_box_param->conf_thresh;
  class_num_ = yolo_box_param->class_num;
  downsample_ratio_ = yolo_box_param->downsample_ratio;
}

int cpu_paddle_yolo_box_layer::reshape(
          void* param,
          const vector<vector<int>>& input_shapes,
          vector<vector<int>>& output_shapes) {
  return 0;
}

int cpu_paddle_yolo_box_layer::dtype(
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

REGISTER_USER_CPULAYER_CLASS(USER_PADDLE_YOLO_BOX, cpu_paddle_yolo_box)

}
