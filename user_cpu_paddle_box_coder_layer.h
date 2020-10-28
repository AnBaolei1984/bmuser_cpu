#ifndef _USER_CPU_PADDLE_BOX_CODER_LAYER_H
#define _USER_CPU_PADDLE_BOX_CODER_LAYER_H 
#include "user_cpu_layer.h"

namespace usercpu {

/*
*  notice: define new cpu layer, must like xxx_layer 
* 
*/
class cpu_paddle_box_coder_layer : public user_cpu_layer {
public:
    explicit cpu_paddle_box_coder_layer() {}
    virtual ~cpu_paddle_box_coder_layer() {}

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
        return "USER_PADDLE_BOX_CODER";
    }
protected:
    void encodeCenterSize();
    void decodeCenterSize();
    float variance_[20];
    int variance_len_;
    int code_type_;
    int axis_;
    int var_size_;
    bool normalized_;
};

} /* namespace usercpu */
#endif /* _USER_CPU_PADDLE_BOX_CODER_LAYER_H */

