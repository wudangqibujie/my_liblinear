#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "cstring"
#include "linear.h"
#include <ctype.h>
#define Malloc(type, n) (type *)malloc((n) * sizeof(type))
#define INF HUGE_VAL

static char *line = NULL;
static int max_line_len;

static char *readline(FILE *input){
    int len;
    if (fgets(line, max_line_len, input) == NULL)
        return NULL;
    while (strrchr(line, '\n') == NULL){
        max_line_len *= 2;
        line = (char *) realloc(line, max_line_len);
        len = (int) strlen(line);
        if (fgets(line + len, max_line_len - len, input) == NULL)
            break;
    }
    return line;
}

void exit_with_help()
{
    printf(
            "Usage: train [options] training_set_file [model_file]\n"
            "options:\n"
            "-s type : set type of solver (default 1)\n"
            "  for multi-class classification\n"
            "        0 -- L2-regularized logistic regression (primal)\n"
            "        1 -- L2-regularized L2-loss support vector classification (dual)\n"
            "        2 -- L2-regularized L2-loss support vector classification (primal)\n"
            "        3 -- L2-regularized L1-loss support vector classification (dual)\n"
            "        4 -- support vector classification by Crammer and Singer\n"
            "        5 -- L1-regularized L2-loss support vector classification\n"
            "        6 -- L1-regularized logistic regression\n"
            "        7 -- L2-regularized logistic regression (dual)\n"
            "  for regression\n"
            "       11 -- L2-regularized L2-loss support vector regression (primal)\n"
            "       12 -- L2-regularized L2-loss support vector regression (dual)\n"
            "       13 -- L2-regularized L1-loss support vector regression (dual)\n"
            "  for outlier detection\n"
            "       21 -- one-class support vector machine (dual)\n"
            "-c cost : set the parameter C (default 1)\n"
            "-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
            "-n nu : set the parameter nu of one-class SVM (default 0.5)\n"
            "-e epsilon : set tolerance of termination criterion\n"
            "       -s 0 and 2\n"
            "               |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
            "               where f is the primal function and pos/neg are # of\n"
            "               positive/negative data (default 0.01)\n"
            "       -s 11\n"
            "               |f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.0001)\n"
            "       -s 1, 3, 4, 7, and 21\n"
            "               Dual maximal violation <= eps; similar to libsvm (default 0.1 except 0.01 for -s 21)\n"
            "      -s 5 and 6\n"
            "               |f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
            "               where f is the primal function (default 0.01)\n"
            "       -s 12 and 13\n"
            "               |f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
            "               where f is the dual function (default 0.1)\n"
            "-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
            "-R : not regularize the bias; must with -B 1 to have the bias; DON'T use this unless you know what it is\n"
            "       (for -s 0, 2, 5, 6, 11)\n"
            "-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
            "-v n: n-fold cross validation mode\n"
            "-C : find parameters (C for -s 0, 2 and C, p for -s 11)\n"
            "-q : quiet mode (no outputs)\n"
    );
    exit(1);
}
void print_null(const char *s){};
void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
struct feature_node *x_space;
struct parameter param;
struct problem prob;
int flag_cross_validation;
int flag_find_parameters;
int flag_C_specified;
int flag_p_specified;
int flag_solver_specified;
int nr_fold;
double bias;


int main(int argc, char **argv) {
    std::cout << "start jayliu liblinear" << std::endl;
    std::cout << "start jayliu liblinear" << std::endl;
    std::cout << "Argc NUM: " << argc << std::endl;
    std::cout << "Arg 0: " << argv[3] << std::endl;
    char input_file_name[1024];
    std::cout << input_file_name << std::endl;
    char model_file_name[1024];
    const char *error_msg;
    parse_command_line(argc, argv, input_file_name, model_file_name);
    read_problem(input_file_name);
    return 0;
}

void parse_command_line(int argc, char *argv[], char *input_file_name, char *model_file_name){
    int i;
    void (*print_func)(const char*) = NULL;
    // default values
    param.solver_type = L2R_L2LOSS_SVC_DUAL;
    param.C = 1;
    param.p = 0.1;
    param.nu = 0.5;
    param.eps = INF;
    param.nr_weight = 0;
    param.regularize_bias = 1;
    param.weight_label = NULL;
    param.weight = NULL;
    flag_cross_validation = 0;
    flag_C_specified = 0;
    flag_p_specified = 0;
    flag_solver_specified = 0;
    flag_find_parameters = 0;
    bias = -1;

    for(i=1;i<argc;i++)
    {
        if(argv[i][0] != '-') break;
        if(++i>=argc)
            exit_with_help();
        switch(argv[i-1][1])
        {
            case 's':
                param.solver_type = atoi(argv[i]);
                flag_solver_specified = 1;
                break;

            case 'c':
                param.C = atof(argv[i]);
                flag_C_specified = 1;
                break;

            case 'p':
                flag_p_specified = 1;
                param.p = atof(argv[i]);
                break;

            case 'n':
                param.nu = atof(argv[i]);
                break;

            case 'e':
                param.eps = atof(argv[i]);
                break;

            case 'B':
                bias = atof(argv[i]);
                break;

            case 'w':
                ++param.nr_weight;
                param.weight_label = (int *) realloc(param.weight_label,sizeof(int)*param.nr_weight);
                param.weight = (double *) realloc(param.weight,sizeof(double)*param.nr_weight);
                param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
                param.weight[param.nr_weight-1] = atof(argv[i]);
                break;
            case 'v':
                flag_cross_validation = 1;
                nr_fold = atoi(argv[i]);
                if(nr_fold < 2)
                {
                    fprintf(stderr,"n-fold cross validation: n must >= 2\n");
                    exit_with_help();
                }
                break;
            case 'q':
                print_func = &print_null;
                i--;
                break;
            case 'C':
                flag_find_parameters = 1;
                i--;
                break;
            case 'R':
                param.regularize_bias = 0;
                i--;
                break;
            default:
                fprintf(stderr,"unknown option: -%c\n", argv[i-1][1]);
                exit_with_help();
                break;
        }
    }

    set_print_string_function(print_func);
    if (i >= argc){
        exit_with_help();
    }
    strcpy(input_file_name, argv[i]);
    if (i < argc - 1)
        strcpy(model_file_name, argv[i + 1]);
    else{
        char *p = strrchr(argv[i], '/');
        if(p == NULL)
            p = argv[i];
        else
            ++p;
        sprintf(model_file_name, "%s.model", p);
    }

    if(flag_find_parameters)
    {
        if(!flag_cross_validation)
            nr_fold = 5;
        if(!flag_solver_specified)
        {
            fprintf(stderr, "Solver not specified. Using -s 2\n");
            param.solver_type = L2R_L2LOSS_SVC;
        }
        else if(param.solver_type != L2R_LR && param.solver_type != L2R_L2LOSS_SVC && param.solver_type != L2R_L2LOSS_SVR)
        {
            fprintf(stderr, "Warm-start parameter search only available for -s 0, -s 2 and -s 11\n");
            exit(1);
        }
    }
    if (param.eps == INF){
        switch (param.solver_type) {
            case L2R_LR:
            case L2R_L2LOSS_SVC:
                param.eps = 0.01;
                break;
            case L2R_L2LOSS_SVR:
                param.eps = 0.0001;
                break;
            case L2R_L2LOSS_SVC_DUAL:
            case L2R_L1LOSS_SVC_DUAL:
            case MCSVM_CS:
            case L2R_LR_DUAL:
                param.eps = 0.1;
                break;
            case L1R_L2LOSS_SVC:
            case L1R_LR:
                param.eps = 0.01;
                break;
            case L2R_L1LOSS_SVR_DUAL:
            case L2R_L2LOSS_SVR_DUAL:
                param.eps = 0.1;
                break;
            case ONECLASS_SVM:
                param.eps = 0.01;
                break;
        }
    }
}

void read_problem(const char *filename){
    int max_index, inst_max_index, i;
    size_t elements, j;
    FILE *fp = fopen(filename, "r");
    char *endptr;
    char *idx, *val, *label;
    if(fp == NULL){
        std::cout << "can't open input file" << std::endl;
        fprintf(stderr,"can't open input file %s\n", filename);
        exit(1);
    }
    prob.l = 0;
    elements = 0;
    max_line_len = 1024;
    line = Malloc(char, max_line_len);
    while(readline(fp)!=NULL)
    {
        char *p = strtok(line," \t"); // label
        // features
        while(1)
        {
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            elements++;
        }
        elements++; // for bias term
        prob.l++;
    }
    rewind(fp);
    prob.bias = bias;

    prob.y = Malloc(double, prob.l);
    prob.x = Malloc(struct feature_node *, prob.l);
    x_space = Malloc(struct feature_node, elements + prob.l);
    
    max_index = 0;
    j = 0;
    for (int k = 0; k < prob.l; ++k) {
        inst_max_index = 0;
        readline(fp);
        prob.x[k] = &x_space[i];
    }
}