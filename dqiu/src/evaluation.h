#ifndef EVALUATION_H
#define EVALUATION_H

#include "expression.h"
#include "tensor.h"
class evaluation
{
public:
    evaluation(const std::vector<expression> &exprs);

    void add_kwargs_double(
        const char *key,
        double value);

    void add_kwargs_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);

    // return 0 for success
    int execute();

    // return the variable computed by the last expression
    tensor &get_result();

private:

    std::vector<expression> _exprs;
    std::map<std::string, tensor> _kwargs;
    tensor result_;
}; // class evaluation

#endif // EVALUATION_H
