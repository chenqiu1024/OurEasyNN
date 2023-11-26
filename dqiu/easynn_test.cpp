/**
 * A simple test program helps you to debug your easynn implementation.
 */

#include <stdio.h>
#include "src/libeasynn.h"
#include "src/tensor.h"
#include <map>

using namespace std;

int evaluate(evaluation *eval) {
    int dim = 0;
    size_t *shape = nullptr;
    double *data = nullptr;
    if (execute(eval, &dim, &shape, &data) != 0)
    {
        printf("evaluation fails\n");
        return -1;
    }

    printf("evaluate: dim=%d, shape=[", dim);
    for (int iDim = 0; iDim < dim; iDim++)
    {
        printf("%d  ", shape[iDim]);
    }
    printf("]\n");
    if (dim == 0)
        printf("res = %f\n", data[0]);
    else
    {
        int N = 1;
        for (int i=0; i<dim; i++)
        {
            N *= shape[i];
        }
        printf("res = [%.3f", data[0]);
        for (int i = 1; i < N; i++)
        {
            printf(", %.3f", data[i]);
        }
        printf("]\n");
    }
    return 0;
}

void testP3Q1a() {
    // x = nn.Input("x")
    // return is_same(x, 1, x = (9,)) and is_same(x, 1, x = (9, 9))
    program *prog = create_program();
    int inputs0[] = {};
    append_expression(prog, 0, "x", "Input", inputs0, 0);
    evaluation *eval = build(prog);
    // add_kwargs_double(eval, "x", 5);
    size_t shapeX[] = {9};
    double dataX[] = {1,2,3,4,5,6,7,8,9};
    add_kwargs_ndarray(eval, "x", 1, shapeX, dataX);
    // add_kwargs_double(eval, "x", 5);

    if (0 != evaluate(eval))
    {
        printf("testP3Q1a failed\n");
    }
    delete eval;
    delete prog;
}

void testP3Q1b() {
    // x = nn.Input("x")
    // return is_same(x, 1, x = (9,)) and is_same(x, 1, x = (9, 9))
    program *prog = create_program();
    int inputs0[] = {};
    append_expression(prog, 0, "x", "Input", inputs0, 0);
    evaluation *eval = build(prog);
    // add_kwargs_double(eval, "x", 5);
    size_t shapeX[] = {9, 9};
    double dataX[81];
    for (int i = 0; i < 81; i++) { dataX[i] = i; }
    add_kwargs_ndarray(eval, "x", 2, shapeX, dataX);

    if (0 != evaluate(eval))
    {
        printf("testP3Q1b failed\n");
    }
    delete eval;
    delete prog;
}

int testP3Q2a() {
    // c1 = nn.Const(np.random.random((10,)))
    // c2 = nn.Const(np.random.random((10, 10)))
    // return is_same(c1, 1) and is_same(c2, 1)
    program *prog = create_program();
    int inputs0[] = {};
    append_expression(prog, 0, "c", "Const", inputs0, 0);
    size_t shapeC[] = {10};
    double dataC[] = {1,2,3,4,5,6,7,8,9,10};
    add_op_param_ndarray(prog, "value", 1, shapeC, dataC);
    evaluation *eval = build(prog);

    return evaluate(eval);
}

int testP3Q6() {
    program *prog = create_program();
    int inputs0[] = {};
    int inputs2[] = {0, 1};
    append_expression(prog, 0, "x", "Input", inputs0, 0);
    append_expression(prog, 1, "y", "Input", inputs0, 0);
    append_expression(prog, 2, "z", "Mul", inputs2, 2);
    evaluation *eval = build(prog);

    size_t zeroShape[] = {};
    double dataX = 2;
    size_t shape[] = {2, 3};
    double data[] = {1,2,3,4,5,6};

    add_kwargs_ndarray(eval, "x", 0, zeroShape, &dataX);
    add_kwargs_ndarray(eval, "y", 2, shape, data);
    
    return evaluate(eval);
}

int main()
{
    testP3Q6();
    // testP3Q1b();
    return 0;
}
