#ifndef _USER_CPU_PADDLE_MULTICLASS_NMS_LAYER_H
#define _USER_CPU_PADDLE_MULTICLASS_NMS_LAYER_H
#include "user_cpu_layer.h"

namespace usercpu {

/*
*  notice: define new cpu layer, must like xxx_layer 
* 
*/
class cpu_paddle_multiclass_nms_layer : public user_cpu_layer {
public:
    explicit cpu_paddle_multiclass_nms_layer() {}
    virtual ~cpu_paddle_multiclass_nms_layer() {}

    /* dowork */
    int process(void *parm);
    void setParam(void *param);

    int reshape(void* param,
                const vector<vector<int>>& input_shapes,
                vector<vector<int>>& output_shapes);

    int dtype(void* param,
              const vector<int>& input_dtypes,
              vector<int>& output_dtypes);
    virtual string get_layer_name () const {
        return "USER_PADDLE_MULTICLASS_NMS";
    }
protected:
    float score_threshold_;
    float nms_threshold_;
    float nms_eta_;
    int keep_top_k_;
    int nms_top_k_;
    int background_label_;
    bool normalized_;
    std::vector<int> boxes_dims_;
    std::vector<int> score_dims_;
    bool return_index_;

    void GetMaxScoreIndex(const float* scores,
                    std::vector<std::pair<float, int>>* sorted_indices);
    float BBoxArea(const float* box);
    float JaccardOverlap(const float* box1, const float* box2);
    void NMSFast(float* box, float* score,
                          std::vector<int>* selected_indices);
    void MultiClassOutput(float* box,
                      float* score,
                      float* output,
                      float* output_index,
                      const std::map<int, std::vector<int>>& selected_indices,
                      const int scores_size,
                      const int offset);
    void MultiClassNMS(float* box, float* score,
                   std::map<int, std::vector<int>>* indices,
                   int* num_nmsed_out);
};

} /* namespace usercpu */
#endif /* _USER_CPU_PADDLE_MULTICLASS_NMS_LAYER_H */

