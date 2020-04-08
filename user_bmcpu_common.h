#ifndef _USER_CPU_COMMON_H_
#define _USER_CPU_COMMON_H_

typedef enum {
    USER_EXP = 0,
    USER_PADDLE_BOX_CODER = 1,
    USER_PADDLE_MULTICLASS_NMS = 2,
    USER_PADDLE_YOLO_BOX = 3,
    USER_CPU_UNKNOW
} USER_CPU_LAYER_TYPE_T;

typedef struct user_cpu_exp_param {
    float inner_scale_;
    float outer_scale_;
} user_cpu_exp_param_t;

typedef struct user_cpu_box_coder_param {
    float* variance;
    int code_type;
    int axis;
    bool normalized;
} user_cpu_box_coder_param_t;

typedef struct user_cpu_multiclass_nms_param {
    float score_threshold;
    float nms_threshold;
    float nms_eta;
    int keep_top_k;
    int nms_top_k;
    int background_label;
    bool normalized;
} user_cpu_multiclass_nms_param_t;

typedef struct user_cpu_yolo_box_param {
    int* anchors;
    float conf_thresh;
    int class_num;
    int downsample_ratio;
    int anchors_size;
} user_cpu_yolo_box_param_t;

union U {
  user_cpu_exp_param_t exp;
  user_cpu_box_coder_param_t box_coder_param;
  user_cpu_multiclass_nms_param_t multiclass_nms_param;
  user_cpu_yolo_box_param_t yolo_box_param;
  U(){};
  ~U(){};
};

typedef struct user_cpu_param
{
    int op_type;   /* USER_CPU_LAYER_TYPE_T */
    U u{};
} user_cpu_param_t;



#endif /* _USER_CPU_COMMON_H_ */
