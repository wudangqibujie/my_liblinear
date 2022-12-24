//
// Created by jay on 2022/12/19.
//

#ifndef MY_FM_LINEAR_H
#define MY_FM_LINEAR_H

struct feature_node{
    int index;
    double value;
};

struct problem{
    int l, n;
    double *y;
    struct feature_node **x;
    double bias;
};

enum { L2R_LR,
        L2R_L2LOSS_SVC_DUAL,
        L2R_L2LOSS_SVC,
        L2R_L1LOSS_SVC_DUAL,
        MCSVM_CS,
        L1R_L2LOSS_SVC,
        L1R_LR,
        L2R_LR_DUAL,
        L2R_L2LOSS_SVR = 11,
        L2R_L2LOSS_SVR_DUAL,
        L2R_L1LOSS_SVR_DUAL,
        ONECLASS_SVM = 21 }; /* solver_type */


struct parameter{
    int solver_type;

    double eps;
    double C;
    int nr_weight;
    int *weight_label;
    double *weight;
    double p;
    double nu;
    double *init_sol;
    int regularize_bias;
};

void set_print_string_function(void (*print_func) (const char*));

#endif //MY_FM_LINEAR_H
