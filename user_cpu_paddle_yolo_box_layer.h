#ifndef _USER_CPU_PADDLE_YOLO_BOX_LAYER_H
#define _USER_CPU_PADDLE_YOLO_BOX_LAYER_H
#include "user_cpu_layer.h"

namespace usercpu {

/*
*  notice: define new cpu layer, must like xxx_layer 
* 
*/
class cpu_paddle_yolo_box_layer : public user_cpu_layer {
public:
    explicit cpu_paddle_yolo_box_layer() {}
    virtual ~cpu_paddle_yolo_box_layer() {}

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
        return "USER_PADDLE_YOLO_BOX";
    }
protected:
    int anchors_[100];
    int anchors_size_;
    float conf_thresh_;
    int class_num_;
    int downsample_ratio_;

    float sigmoid(float x);
    void get_yolo_box(float* box,
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
                         int img_width);
    int get_entry_index(int batch,
                           int an_idx,
                           int hw_idx,
                           int an_num,
                           int an_stride,
                           int stride,
                           int entry);
    void calc_detection_box(float* boxes,
                               float* box,
                               const int box_idx,
                               const int img_height,
                               const int img_width);
    void calc_label_score(float* scores,
                             const float* input,
                             const int label_idx,
                             const int score_idx,
                             const int class_num,
                             const float conf,
                             const int stride);
};

} /* namespace usercpu */
#endif /* _USER_CPU_PADDLE_YOLO_BOX_LAYER_H */

