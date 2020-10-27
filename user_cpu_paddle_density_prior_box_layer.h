#ifndef _USER_CPU_PADDLE_DENSITY_PRIOR_BOX_LAYER_H
#define _USER_CPU_PADDLE_DENSITY_PRIOR_BOX_LAYER_H
#include "user_cpu_layer.h"

namespace usercpu {

/*
*  notice: define new cpu layer, must like xxx_layer 
* 
*/
class cpu_paddle_density_prior_box_layer : public user_cpu_layer {
public:
    explicit cpu_paddle_density_prior_box_layer() {}
    virtual ~cpu_paddle_density_prior_box_layer() {}

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
        return "USER_PADDLE_DENSITY_PRIOR_BOX";
    }
protected:
    user_cpu_density_prior_box_param_t density_prior_box_params_;
};

} /* namespace usercpu */
#endif /* _USER_CPU_PADDLE_DENSITY_PRIOR_BOX_LAYER_H */

