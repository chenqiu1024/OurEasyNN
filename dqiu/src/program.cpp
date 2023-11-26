#include "program.h"
#include "evaluation.h"

program::program()
{
}

void program::append_expression(
    int expr_id,
    const char *op_name,
    const char *op_type,
    int inputs[],
    int num_inputs)
{
    expression expr(expr_id, op_name, op_type, inputs, num_inputs);
    _exprs.push_back(expr);
}

int program::add_op_param_double(
    const char *key,
    double value)
{
    if (_exprs.size() == 0)
    {
        return -1;
    }
    _exprs.back().add_op_param_double(key, value);
    return 0;
}

int program::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    if (_exprs.size() == 0)
    {
        return -1;
    }
    _exprs.back().add_op_param_ndarray(key, dim, shape, data);
    return 0;
}

evaluation *program::build()
{
    evaluation* eval = new evaluation(_exprs);
    return eval;
}
