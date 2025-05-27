#ifndef _HASHTABLE_H
#define _HASHTABLE_H

#define CACHE_LINE_SIZE 64

typedef struct FParseParamsTg FParseParams;

typedef struct FProcessStateTg {
   char key_buf[128]; // For case when key column is not in value
   char *key_pos;
   int key_size;
   char value_start[128];
   char *value_pos;
   int value_size;
   int col_num;
   FParseParams *pp;

   uint64_t tick;

   char *data_refs[10];
   char *chain_ref;
   char **data_ref;
   } FProcessState;

#define SET_SIZE 128
typedef struct FProcessStateSetTg
   {
   FProcessState *states[SET_SIZE];
   int first;
   int last;
   int count;
   } FProcessStateSet;

typedef struct FHashTableTg {
   char *table;

   uint32_t *data;
   uint32_t *data_pos;

   uint64_t tick;
   int pcnt;
   } FHashTable;

#endif // !_HASHTABLE_H
