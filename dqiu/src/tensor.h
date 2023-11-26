#ifndef TENSOR_H
#define TENSOR_H

#include "expression.h"

class tensor
{
public:
    tensor();
    explicit tensor(double v);
    tensor(int dim,size_t shape[],double data[]);
    tensor(std::vector<size_t>shape);

    int get_dim() const;
    double item() const;
    double &item();
    double at(size_t i) const;
    double at(size_t i, size_t j) const;
    size_t *get_shape_array();
    double *get_data_array();

    tensor& operator = (const tensor& rhs);

    operator double() const;

    tensor operator - (const tensor& rhs) const;
    tensor operator + (const tensor& rhs) const;

    tensor operator * (const tensor& rhs) const;
    
    tensor operator - () const;

    tensor relu() const;

    tensor flatten() const;

    tensor NHWC2NCHW() const;

    tensor linear(const tensor& weight, const tensor& bias) const;

    tensor maxPool2d(int kernelSize, int stride) const;

    tensor conv2d(const tensor& weight, const tensor& bias) const;

    void print() const;

private:
    std::vector<size_t> shape_;
    std::vector<double> data_;
};

#endif // TENSOR_H
