#include <assert.h>
#include <iostream>
#include "evaluation.h"

evaluation::evaluation(const std::vector<expression> &exprs)
    : result_(0)
{
    _exprs = exprs;
}

void evaluation::add_kwargs_double(
    const char *key,
    double value)
{
    tensor t(value);
    _kwargs[key] = t;
}

void evaluation::add_kwargs_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    tensor t(dim, shape, data);
    _kwargs[key] = t;
}

int evaluation::execute()
{
    std::map<int, tensor> results;
    for (expression expr : _exprs)
    {
        if (expr._op_type == "Add")
        {
            results[expr._expr_id] = results[expr._inputs[0]] + results[expr._inputs[1]];
        }
        else if (strcmp(expr._op_type.c_str(), "Sub") == 0)
        {
            results[expr._expr_id] = results[expr._inputs[0]] - results[expr._inputs[1]];
        }
        else if (strcmp(expr._op_type.c_str(), "Mul") == 0)
        {
            results[expr._expr_id] = results[expr._inputs[0]] * results[expr._inputs[1]];
        }
        else if (strcmp(expr._op_type.c_str(), "Neg") == 0)
        {
            results[expr._expr_id] = -results[expr._inputs[0]];
        }
        else if (strcmp(expr._op_type.c_str(), "Const") == 0)
        {
            results[expr._expr_id] = expr._params["value"];
        }
        else if (strcmp(expr._op_type.c_str(), "Input") == 0)
        {
            results[expr._expr_id] = _kwargs[expr._op_name]; 
        }
        else if (strcmp(expr._op_type.c_str(), "ReLU") == 0)
        {
            results[expr._expr_id] = results[expr._inputs[0]].relu();
        }
        else if (strcmp(expr._op_type.c_str(), "Flatten") == 0)
        {
            results[expr._expr_id] = results[expr._inputs[0]].flatten(); 
        }
        else if (strcmp(expr._op_type.c_str(), "Input2d") == 0)
        {
            int H = (int)expr._params["height"];
            int W = (int)expr._params["width"];
            int C = (int)expr._params["in_channels"];
            tensor& rhs = _kwargs[expr._op_name];
            assert(rhs.get_dim() == 4);
            size_t* shape = rhs.get_shape_array();
            assert(shape[1] == H && shape[2] == W && shape[3] == C);
            results[expr._expr_id] = rhs.NHWC2NCHW(); 
        }
        else if (strcmp(expr._op_type.c_str(), "Linear") == 0)
        {
            int I = (int)expr._params["in_features"];
            int O = (int)expr._params["out_features"];
            tensor& in = results[expr._inputs[0]];
            tensor& weight = expr._params["weight"];
            tensor& bias = expr._params["bias"];
            assert(weight.get_dim() == 2 && bias.get_dim() == 1 && in.get_dim() == 2);
            assert(weight.get_shape_array()[0] == O && weight.get_shape_array()[1] == I);
            assert(bias.get_shape_array()[0] == O);
            assert(in.get_shape_array()[1] == I);
            results[expr._expr_id] = in.linear(weight, bias);
        }
        else if (strcmp(expr._op_type.c_str(), "MaxPool2d") == 0)
        {
            int k = (int)expr._params["kernel_size"];
            int s = (int)expr._params["stride"];
            tensor& in = results[expr._inputs[0]];
            results[expr._expr_id] = in.maxPool2d(k, s);
        }
        else if (strcmp(expr._op_type.c_str(), "Conv2d") == 0)
        {
            // weight: {out_channels, in_channels, kernel_size, kernel_size}
            // input: {N, in_channels, H, W}
            int I = (int)expr._params["in_channels"];
            int O = (int)expr._params["out_channels"];
            int K = (int)expr._params["kernel_size"];
            int P = (int)expr._params["padding"];
            tensor& in = results[expr._inputs[0]];
            tensor& weight = expr._params["weight"];
            tensor& bias = expr._params["bias"];
            assert(weight.get_dim() == 4 && in.get_dim() == 4);
            size_t* weightShape = weight.get_shape_array();
            assert(weightShape[0] == O);
            assert(weightShape[1] == I);
            assert(weightShape[2] == K);
            assert(weightShape[3] == K);
            assert(bias.get_dim() == 1 && bias.get_shape_array()[0] == O);
            size_t* inShape = in.get_shape_array();
            assert(inShape[1] == I);
            results[expr._expr_id] = in.conv2d(weight, bias);
        }
    }
    result_ = results[_exprs.back()._expr_id];
    return 0;
}

tensor &evaluation::get_result()
{
    return result_;
}
