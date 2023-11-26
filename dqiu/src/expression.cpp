#include "expression.h"
#include "tensor.h"
#include <iostream>

expression::expression(
    int expr_id,
    const char *op_name,
    const char *op_type,
    int *inputs,
    int num_inputs)
: _expr_id(expr_id)
, _op_name(op_name)
, _op_type(op_type)
{
    // printf("expression::expression type='%s', name='%s'\n", _op_type.c_str(), _op_name.c_str());
    for (int i = 0; i < num_inputs; i++)
    {
        _inputs.push_back(inputs[i]);
    }
}

void expression::add_op_param_double(
    const char *key,
    double value)
{
    tensor t(value);
    _params[key] = t;
}

void expression::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    tensor t(dim, shape, data);
    _params[key] = t;
}
