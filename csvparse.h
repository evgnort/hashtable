#ifndef _CSVPARSE_H
#define _CSVPARSE_H

#include <setjmp.h>

#define CACHE_LINE_SIZE 64

typedef struct FSourceTg {
   char *input;
   int inputlen;
   int inputpos;
   __m128i lastpart;
   int lp_size;
   jmp_buf eod_exit;
   } FSource;

typedef struct FParseParamsTg {
   char delimiter;
   char src_delimiter;
   int key_col_num;
   int value_col_mask;
   int value_col_count;
   } FParseParams;

typedef struct FProcessStateTg FProcessState;

void init_source(FSource *source,char *input,int len);
int process_row(FSource *source,FProcessState *state);


#endif 
