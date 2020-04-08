#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "user_cpu_paddle_multiclass_nms_layer.h"
#include <sys/time.h>

namespace usercpu {

template <class T>
static bool SortScorePairDescend(
                          const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

void cpu_paddle_multiclass_nms_layer::GetMaxScoreIndex(
                    const float* scores,
                    std::vector<std::pair<float, int>>* sorted_indices) {
  int num_boxes = boxes_dims_[1];
  for (size_t i = 0; i < num_boxes; ++i) {
    if (scores[i] > score_threshold_) {
      sorted_indices->push_back(std::make_pair(scores[i], i));
    }
  }
  // Sort the score pair according to the scores in descending order
  std::sort(sorted_indices->begin(), sorted_indices->end(),
                   SortScorePairDescend<int>);
  // Keep top_k scores if needed.
  if (nms_top_k_ > -1 && 
            nms_top_k_ < static_cast<int>(sorted_indices->size())) {
    sorted_indices->resize(nms_top_k_);
  }
}

float cpu_paddle_multiclass_nms_layer::BBoxArea(const float* box) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return 0.f;
  } else {
    const float w = box[2] - box[0];
    const float h = box[3] - box[1];
    if (normalized_) {
      return w * h;
    } else {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    }
  }
}

float cpu_paddle_multiclass_nms_layer::JaccardOverlap(
                                    const float* box1, 
                                    const float* box2) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return 0.f;
  } else {
    const float inter_xmin = std::max(box1[0], box2[0]);
    const float inter_ymin = std::max(box1[1], box2[1]);
    const float inter_xmax = std::min(box1[2], box2[2]);
    const float inter_ymax = std::min(box1[3], box2[3]);
    float norm = normalized_ ? 0.f : 1.f;
    float inter_w = inter_xmax - inter_xmin + norm;
    float inter_h = inter_ymax - inter_ymin + norm;
    const float inter_area = inter_w * inter_h;
    const float bbox1_area = BBoxArea(box1);
    const float bbox2_area = BBoxArea(box2);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

void cpu_paddle_multiclass_nms_layer::NMSFast(float* box,
                        float* score,
                        std::vector<int>* selected_indices) {
  int box_size = boxes_dims_[2];
  std::vector<std::pair<float, int>> sorted_indices;
  GetMaxScoreIndex(score, &sorted_indices);
  selected_indices->clear();
  float adaptive_threshold = nms_threshold_;
  while (sorted_indices.size() != 0) {
    const int idx = sorted_indices.front().second;
    bool keep = true;
    for (size_t k = 0; k < selected_indices->size(); ++k) {
      if (keep) {
        const int kept_idx = (*selected_indices)[k];
        float overlap = 0.f;
        // 4: [xmin ymin xmax ymax]
        if (box_size == 4) {
          overlap = JaccardOverlap(box + idx * box_size,
                                   box + kept_idx * box_size);
        }
        keep = overlap <= adaptive_threshold;
      } else {
        break;
      }
    }
    if (keep) {
      selected_indices->push_back(idx);
    }
    sorted_indices.erase(sorted_indices.begin());
    if (keep && nms_eta_ < 1 && adaptive_threshold > 0.5) {
      adaptive_threshold *= nms_eta_;
    }
  }
}

void cpu_paddle_multiclass_nms_layer::MultiClassOutput(float* box,
                      float* score,
                      float* output,
                      const std::map<int, std::vector<int>>& selected_indices,
                      const int scores_size) {
  int predict_dim = score_dims_[2];
  int box_size = boxes_dims_[2];
  int out_dim = box_size + 2;
  const float* sdata;
  int count = 0;
  for (const auto& it : selected_indices) {
    int label = it.first;
    const std::vector<int>& indices = it.second;
    sdata = score + label * predict_dim;
    for (size_t j = 0; j < indices.size(); ++j) {
      int idx = indices[j];
      output[count * out_dim] = label;  // label
      const float* bdata;
      bdata = box + idx * box_size;
      output[count * out_dim + 1] = sdata[idx];  // score
      // xmin, ymin, xmax, ymax or multi-points coordinates
      memcpy(output + count * out_dim + 2, bdata, box_size * sizeof(float));
      count++;
    }
  }
}

void cpu_paddle_multiclass_nms_layer::MultiClassNMS(float* box,
                   float* score,
                   std::map<int, std::vector<int>>* indices,
                   int* num_nmsed_out) {

  int num_det = 0;
  int class_num = score_dims_[1];
  // class
  for (int c = 0; c < class_num; ++c) {
    if (c == background_label_) continue;
    float* c_score = score + c * score_dims_[2];
    NMSFast(box, c_score, &((*indices)[c]));
    num_det += (*indices)[c].size();
  }

  *num_nmsed_out = num_det;
  if (keep_top_k_ > -1 && num_det > keep_top_k_) {
    const float* sdata;
    std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
    for (const auto& it : *indices) {
      int label = it.first;
      sdata = score + label * score_dims_[2];
      const std::vector<int>& label_indices = it.second;
      for (size_t j = 0; j < label_indices.size(); ++j) {
        int idx = label_indices[j];
        score_index_pairs.push_back(
            std::make_pair(sdata[idx], std::make_pair(label, idx)));
      }
    }
    // Keep top k results per image.
    std::sort(score_index_pairs.begin(), score_index_pairs.end(),
                     SortScorePairDescend<std::pair<int, int>>);
    score_index_pairs.resize(keep_top_k_);

    // Store the new indices.
    std::map<int, std::vector<int>> new_indices;
    for (size_t j = 0; j < score_index_pairs.size(); ++j) {
      int label = score_index_pairs[j].second.first;
      int idx = score_index_pairs[j].second.second;
      new_indices[label].push_back(idx);
    }
    new_indices.swap(*indices);
    *num_nmsed_out = keep_top_k_;
  }
}

int cpu_paddle_multiclass_nms_layer::process(void *param) {
#ifdef CALC_TIME
  struct timeval start, stop;
  gettimeofday(&start,0);
#endif
  setParam(param);
  float* boxes = input_tensors_[0];
  float* scores = input_tensors_[1];

  boxes_dims_ = input_shapes_[0];
  score_dims_ = input_shapes_[1];
  int n = boxes_dims_[0];
  int num_nmsed_out = 0;
  std::vector<std::map<int, std::vector<int>>> all_indices;
  std::vector<int> batch_starts = {0};
  //  batch
  for (int i = 0; i < n; ++i) {
    std::map<int, std::vector<int>> indices;
    float* box = boxes + i * boxes_dims_[1] * boxes_dims_[2];
    float* score = scores + i * score_dims_[1] * score_dims_[2];  
    MultiClassNMS(box, score, &indices, &num_nmsed_out);
    all_indices.push_back(indices);
    batch_starts.push_back(batch_starts.back() + num_nmsed_out);
  }

  float* output = output_tensors_[0];
  int out_size = 1;
  for (int i = 0; i < 3; i++) {
    out_size *= output_shapes_[0][0][i];
  }
  memset(output, 0, sizeof(float) * out_size);
  int num_kept = batch_starts.back();
  if (num_kept == 0) {
    output[0] = -1;
    batch_starts = {0, 1};
  } else {
    for (int i = 0; i < n; ++i) {
      float* box = boxes + i * boxes_dims_[1] * boxes_dims_[2];
      float* score = scores + i * score_dims_[1] * score_dims_[2];
      int s = static_cast<int>(batch_starts[i]);
      int e = static_cast<int>(batch_starts[i + 1]);
      if (e > s) {
        MultiClassOutput(
            box, score, output, all_indices[i], score_dims_.size());
      }
      output += output_shapes_[0][0][1] * output_shapes_[0][0][2];
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

void cpu_paddle_multiclass_nms_layer::setParam(void *param) {
  user_cpu_multiclass_nms_param_t *multiclass_nms_param =
                static_cast<user_cpu_multiclass_nms_param_t*>(param);
  score_threshold_ = multiclass_nms_param->score_threshold;
  nms_threshold_ = multiclass_nms_param->nms_threshold;
  nms_eta_ = multiclass_nms_param->nms_eta;
  keep_top_k_ = multiclass_nms_param->keep_top_k;
  nms_top_k_ = multiclass_nms_param->nms_top_k;
  normalized_ = multiclass_nms_param->normalized;
  background_label_ = multiclass_nms_param->background_label;
}

int cpu_paddle_multiclass_nms_layer::reshape(
          void* param,
          const vector<vector<int>>& input_shapes,
          vector<vector<int>>& output_shapes) {
  return 0;
}

/* must register user layer
 * in macro cpu_test##_layer  == class cpu_test_layer 
 * */

REGISTER_USER_CPULAYER_CLASS(USER_PADDLE_MULTICLASS_NMS,
                                     cpu_paddle_multiclass_nms)

}
