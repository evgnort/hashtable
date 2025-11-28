# Copyright (C) Evgeniy Buevich

#ifndef _CSVPARSE_H
#define _CSVPARSE_H

#include <immintrin.h>
#include <setjmp.h>

#define CACHE_LINE_SIZE 64

typedef struct FParseParamsTg {
   char delimiter;
   char src_delimiter;
   int key_col_num;
   int value_col_mask;
   int value_col_count;
   } FParseParams;

typedef struct FSourceTg {
   char *input_start;
   char *input_end;
   char *input_pos;
   FParseParams *pp;
   __m128i lastpart;
   int lp_size;
   jmp_buf eod_exit;
   } FSource;

typedef struct FParseStateTg {
   char *key_buf; 
   char *key_pos;
   int *key_size_ref;
   char *value_start;
   char *value_pos;
   int *value_size_ref;
   int col_num;
   } FParseState;

typedef struct FProcessStateTg FProcessState;

void init_source(FSource *source,char *input,int len);
int process_row(FSource *source,FParseState *state);

typedef struct FNormalRqSetTg FNormalRqSet;
typedef struct FProcessStateTg FProcessState;

int process_csv_plain(FSource *source,FProcessState *states,FNormalRqSet *empty_set,FNormalRqSet *loaded_set);

#endif 
