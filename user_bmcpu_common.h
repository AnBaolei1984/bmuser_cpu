#ifndef _USER_CPU_COMMON_H_
#define _USER_CPU_COMMON_H_

typedef enum {
    USER_EXP = 0,
    USER_PADDLE_BOX_CODER = 1,
    USER_PADDLE_MULTICLASS_NMS = 2,
    USER_PADDLE_YOLO_BOX = 3,
    USER_PADDLE_ADAPTIVE_POOL = 4,
    USER_PADDLE_PRIOR_BOX = 5,
    USER_PADDLE_DENSITY_PRIOR_BOX = 6,
    USER_CPU_UNKNOW
} USER_CPU_LAYER_TYPE_T;

typedef struct user_cpu_exp_param {
    float inner_scale_;
    float outer_scale_;
} user_cpu_exp_param_t;

typedef struct user_cpu_box_coder_param {
    float variance[20];
    int variance_len;
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
    int anchors[100];
    float conf_thresh;
    int class_num;
    int downsample_ratio;
    int anchors_size;
} user_cpu_yolo_box_param_t;

typedef struct user_cpu_adaptive_pool_param {
    int is_avg;
}user_cpu_adaptive_pool_param_t;

typedef struct user_cpu_prior_box_param {
  float min_sizes[20];
  float max_sizes[20];
  float aspect_ratios[20];
  float variances[20];
  int max_sizes_len;
  int min_sizes_len;
  int aspect_ratios_len;
  int variances_len;
  float step_w;
  float step_h;
  float offset;
  int img_w;
  int img_h;
  int prior_num;
  bool min_max_aspect_ratios_order;
  bool clip;
  bool flip;
}user_cpu_prior_box_param_t;

typedef struct user_cpu_density_prior_box_param {
  int densities[20];
  float fixed_sizes[20];
  float fixed_ratios[20];
  float variances[20];
  int densities_len;
  int fixed_sizes_len;
  int fixed_ratios_len;
  int variances_len;
  float step_w;
  float step_h;
  float offset;
  int prior_num;
  bool flatten_to_2d;
  bool clip;
}user_cpu_density_prior_box_param_t;


union U {
  user_cpu_exp_param_t exp;
  user_cpu_box_coder_param_t box_coder_param;
  user_cpu_multiclass_nms_param_t multiclass_nms_param;
  user_cpu_yolo_box_param_t yolo_box_param;
  user_cpu_adaptive_pool_param_t adaptive_pool_parm;
  user_cpu_prior_box_param_t prior_box_param;
  user_cpu_density_prior_box_param_t density_prior_box_param;
  U(){};
  ~U(){};
};

typedef struct user_cpu_param
{
    int op_type;   /* USER_CPU_LAYER_TYPE_T */
    U u{};
} user_cpu_param_t;



#endif /* _USER_CPU_COMMON_H_ */
