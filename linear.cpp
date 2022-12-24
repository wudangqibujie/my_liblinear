//
// Created by jay on 2022/12/20.
//
#include "linear.h"
#include <stdio.h>
#include <stdlib.h>
#include "math.h"

static void print_string_stdout(const char *s){
    fputs(s, stdout);
    fflush(stdout);
}
static void print_null(const char *s) {}
static void (*liblinear_print_string) (const char *) = &print_string_stdout;

void set_print_string_function(void (*print_func)(const char*)){
    if(print_func == NULL)
        liblinear_print_string = &print_string_stdout;
    else
        liblinear_print_string = print_func;
}