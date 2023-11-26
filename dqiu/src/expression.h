#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <vector>
#include <string>
#include <map>

class evaluation;
class tensor;
class expression
{
    friend class evaluation;
public:
    expression(
        int expr_id,
        const char *op_name,
        const char *op_type,
        int *inputs,
        int num_inputs);

    void add_op_param_double(
        const char *key,
        double value);

    void add_op_param_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);

private:

    int _expr_id;
    std::string _op_name;
    std::string _op_type;
    std::vector<int> _inputs;

    std::map<std::string, tensor> _params;

}; // class expression

#endif // EXPRESSION_H
