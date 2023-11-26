#include "tensor.h"
#include <vector>
#include <assert.h>
#include <iostream>

using namespace std;

tensor::tensor():
data_(1,0)
{
}

tensor::tensor(double v):
data_(1, v)
{
}

tensor::tensor(int dim, size_t shape[], double data[]):
shape_(shape, shape + dim)
{
    int N = 1;
    for (int i=0; i<dim; i++)
    {
        N *= shape[i];
    }
    data_.assign(data, data+N);
}

tensor::tensor(std::vector<size_t>shape):
shape_(shape)
{
    int N = 1;
    for (int i=0; i<get_dim(); i++)
    {
        N *= shape[i];
    }
    data_.resize(N);
}

double tensor::item() const
{
    assert(shape_.empty());
    return data_[0];
}

double &tensor::item()
{
    assert(shape_.empty());
    return data_[0];
}

double tensor::at(size_t i) const 
{
    assert(get_dim()==1);
    assert(i< shape_[0]);
    return data_[i];
}

double tensor::at(size_t i, size_t j) const
{
    assert(get_dim()==2);
    assert((i<shape_[0]) && (j<shape_[1]));
    return data_[i*shape_[1]+j];
}

size_t *tensor::get_shape_array()
{
    return shape_.empty()? nullptr: &shape_[0];
}

double *tensor::get_data_array()
{
    return &data_[0];
}

int tensor::get_dim() const
{
    return shape_.size();
}

tensor::operator double() const
{
    if (data_.empty())
    {
        return 0;
    }
    return data_[0];
}

tensor tensor::operator + (const tensor& rhs) const
{
    assert(get_dim()== rhs.get_dim());
    for (int i=0; i<get_dim(); i++)
    {
        assert(shape_[i]==rhs.shape_[i]);
    }
    tensor ret(shape_);

    for (int j=0;j< data_.size(); j++)
    {
        ret.data_[j] = data_[j] + rhs.data_[j];
    }
    return ret;
}

tensor tensor::operator - (const tensor& rhs) const
{
    assert(get_dim()== rhs.get_dim());
    for (int i=0; i<get_dim(); i++)
    {
        assert(shape_[i]==rhs.shape_[i]);
    }
    tensor ret(shape_);
    
    for (int j=0;j< data_.size(); j++)
    {
        ret.data_[j] = data_[j] - rhs.data_[j];
    }
    return ret;
}

tensor& tensor::operator = (const tensor& rhs)
{
    shape_ = rhs.shape_;
    data_ = rhs.data_;
    return *this;
}

tensor tensor::operator * (const tensor& rhs) const
{
    if (get_dim() == 0)
    {
        tensor ret(rhs.shape_);
        for (int j=0;j< rhs.data_.size(); j++)
        {
            ret.data_[j] = data_[0] * rhs.data_[j];
        }
        return ret;
    }
    else if (rhs.get_dim() == 0)
    {
        tensor ret(shape_);
        for (int j=0;j< data_.size(); j++)
        {
            ret.data_[j] = rhs.data_[0] * data_[j];
        }
        return ret;
    }
    assert(get_dim() == 2 && rhs.get_dim() == 2);
    const size_t r1 = shape_[0];
    const size_t c1 = shape_[1];
    const size_t r2 = rhs.shape_[0];
    const size_t c2 = rhs.shape_[1];
    assert(c1 == r2);
    vector<size_t> shapeC = {r1, c2};
    vector<double> dataC(r1 * c2);
    //matrices
    for (int row = 0; row < r1; row++)
    {
        for (int col = 0; col < c2; col++)
        {
            double v = 0;
            for (int j = 0; j < c1; j++)
            {
                v += data_[row * c1 + j] * rhs.data_[j * c2 + col];
            }
            dataC[row * c2 + col] = v;
        }
    }
    tensor ret(shapeC);
    ret.data_.assign(dataC.begin(), dataC.end());
    return ret;
}

tensor tensor::operator - () const
{
    tensor ret(shape_);
    for (int j=0;j< data_.size(); j++)
    {
        ret.data_[j] = -data_[j];
    }
    return ret;
}

tensor tensor::relu() const
{
    tensor ret(shape_);
    for (int j = 0; j < data_.size(); j++)
    {
        ret.data_[j] = data_[j] >= 0 ? data_[j] : 0;
    }
    return ret;
}

tensor tensor::flatten() const
{
    vector<size_t> shape = {shape_[0], 1};
    for (int i = 1; i < shape_.size(); i++)
    {
        shape[1] *= shape_[i];
    }
    tensor ret(shape);
    ret.data_.assign(data_.begin(), data_.end());
    return ret;
}
// input2d, transpose
tensor tensor::NHWC2NCHW() const
{
    const size_t N = shape_[0];
    const size_t H = shape_[1];
    const size_t W = shape_[2];
    const size_t C = shape_[3];
    // n, h, w, c
    // data_[c * 1 + w * C + h * (C * W) + n * (H * W * C)]
    // NCHW:
    // [w *1 + h * W + c * (W * H) + n * (C * H * W)]
    vector<size_t> shape = {N, C, H, W};
    tensor ret(shape);
    for (int iN = 0; iN < N; iN++)
    {
        for (int iH = 0; iH < H; iH++)
        {
            for (int iW = 0; iW < W; iW++)
            {
                for (int iC = 0; iC < C; iC++)
                {
                    ret.data_[iW + iH * W + iC * (W * H) + iN * (W * H * C)] = data_[iC + iW * C + iH * (C * W) + iN * (C * W * H)];
                }
            }
        }
    }
    return ret;
}

tensor tensor::linear(const tensor& weight, const tensor& bias) const
{
    // out[n, o] = sum(i=0~I-1){in[n, i] * weight[o, i]} + bias[o]

    assert(get_dim() == 2 && weight.get_dim() == 2 && bias.get_dim() == 1);
    const size_t N = shape_[0];
    const size_t I = shape_[1];
    assert(weight.shape_[1] == I);
    const size_t O = weight.shape_[0];
    assert(bias.shape_[0] == O);
    vector<size_t> outShape = {N, O};
    tensor out(outShape);
    double* ptrOut = &out.data_[0];
    for (int iN = 0; iN < N; iN++)
    {
        for (int iO = 0; iO < O; iO++)
        {
            double sum = bias.data_[iO];
            for (int iI = 0; iI < I; iI++)
            {
                sum += weight.data_[iO * I + iI] * data_[iN * I + iI];
            }
            *ptrOut++ = sum;
        }
    }
    return out;
}

tensor tensor::maxPool2d(int kernelSize, int stride) const
{
    assert(shape_.size() == 4);
    size_t N = shape_[0];
    size_t C = shape_[1];
    size_t H = shape_[2];
    size_t W = shape_[3];
    assert(kernelSize <= H && kernelSize <= W);
    size_t PH = H/stride; //(H + stride - 1) / stride;
    size_t PW = W/stride; //(W + stride - 1) / stride;
    vector<size_t> poolShape = {N, C, PH, PW};
    tensor pool(poolShape);
    // N,C,H,W
    // src: [(iw + iPW * stride) + (ih + iPH * stride) * W + iC * (W * H) + iN * (W * H * C)]
    // dst: [iPW + iPH * PW + iC * (PW * PH) + iN * (PW * PH * C)]
    double *dst = &pool.data_[0];
    const double *src = &data_[0];
    for (int iN = 0; iN < N; iN++)
    {
        for (int iC = 0; iC < C; iC++)
        {
            for (int iPH = 0; iPH < PH; iPH++)
            {
                for (int iPW = 0; iPW < PW; iPW++)
                {
                    double maxValue = *src;
                    for (int ih = 0; ih < kernelSize; ih++)
                    {
                        for (int iw = 0; iw < kernelSize; iw++)
                        {
                            double value = src[iw + ih * W];
                            if (maxValue < value)
                            {
                                maxValue = value;
                            }
                        }
                    }
                    *dst++ = maxValue;
                    src += stride;
                }
                src += (stride - 1) * W;
            }
        }
    }
    return pool;
}

tensor tensor::conv2d(const tensor& weight, const tensor& bias) const
{
    // weight: {O, I, K, K}, bias: {O}
    // input: {N, I, H, W}
    // output: {N, O, H-K+1, W-K+1}
    // output[n, o, h, w] = bias[o] + sum(i=0~I-1, y=0~K-1, x=0~K-1){input[n, i, h+y, w+x] * weight[o,i,y,x]}
    assert(get_dim() == 4 && weight.get_dim() == 4 && bias.get_dim() == 1);
    const size_t I = weight.shape_[1];
    assert(shape_[1] == I);
    const size_t K = weight.shape_[2];
    assert(K == weight.shape_[3]);
    const size_t O = weight.shape_[0];
    const size_t N = shape_[0];
    const size_t H = shape_[2];
    const size_t W = shape_[3];
    const size_t OH = H - K + 1;
    const size_t OW = W - K + 1;
    vector<size_t> outShape = {N, O, OH, OW};
    tensor out(outShape);
    for (int n=0; n<N; n++)
    {
        for (int o=0; o<O; o++)
        {
            for (int h=0; h<OH; h++)
            {
                for (int w=0; w<OW; w++)
                {
                    double sum = bias.data_[o];
                    for (int i=0; i<I; i++)
                    {
                        for (int x=0; x<K; x++)
                        {
                            for (int y=0; y<K; y++)
                            {
                                double vInput = data_[w+x + (h+y) * W + i * (W * H) + n * (W * H * I)];
                                double vWeight = weight.data_[x + y * K + i * (K * K) + o * (K * K * I)];
                                sum += vInput * vWeight;
                            }
                        }
                    }
                    out.data_[w + h * OW + o * (OW * OH) + n * (OW * OH * O)] = sum;
                }
            }
        }
    }
    return out;
}

void tensor::print() const
{
    printf("tensor(0x%lx): dim=%d, shape=[", (long)this, get_dim());
    if (get_dim() > 0)
    {
        printf("%d", shape_[0]);
        for (int i = 1; i < get_dim(); i++)
        {
            printf(", %d", shape_[i]);
        }
    }
    printf("]\n");
}
