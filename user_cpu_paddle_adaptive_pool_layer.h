#ifndef _USER_CPU_PADDLE_ADAPTIVE_POOL_LAYER_H
#define _USER_CPU_PADDLE_ADAPTIVE_POOL_LAYER_H 
#include "user_cpu_layer.h"
#include <float.h>

namespace usercpu {

class PoolProc {
public:
  explicit PoolProc() {}
  virtual ~PoolProc() {}
  virtual float initial() = 0;
  virtual void compute(const float& x, float* y) = 0;
  virtual void finalize(const float& pool_field, float* y) = 0;
};

class MaxPool : public PoolProc{
public:
  virtual float initial() { return -FLT_MAX; }
  virtual void compute(const float& x, float* y) { *y = *y > x ? *y : x; }
  virtual void finalize(const float& pool_field, float* y) {}
};

class AvgPool : public PoolProc{
public:
  virtual float initial() { return 0; }
  virtual void compute(const float& x, float* y) { *y += x; }
  virtual void finalize(const float& pool_field, float* y) { *y /= pool_field; }
};
/*
*  notice: define new cpu layer, must like xxx_layer 
* 
*/
class cpu_paddle_adaptive_pool_layer : public user_cpu_layer {
public:
    explicit cpu_paddle_adaptive_pool_layer() {}
    virtual ~cpu_paddle_adaptive_pool_layer() {}
    /* dowork */
    int process(void *parm);
    void setParam(void *param);

    int reshape(void* param,
                const vector<vector<int>>& input_shapes,
                vector<vector<int>>& output_shapes);

    virtual string get_layer_name () const {
        return "USER_PADDLE_ADAPTIVE_POOL";
    }
    int AdaptStartIndex(
                        int ph, int input_size, int output_size) {
      return static_cast<int>(floor(static_cast<double>(ph * input_size) / output_size));
    }

    int AdaptEndIndex(
                      int ph, int input_size, int output_size) {
      return static_cast<int>(ceil(static_cast<double>((ph + 1) * input_size) / output_size));
    }
protected:
    PoolProc* pool_process_;
};

} /* namespace usercpu */
#endif /* _USER_CPU_PADDLE_ADAPTIVE_POOL_LAYER_H */

