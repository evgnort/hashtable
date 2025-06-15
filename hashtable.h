#ifndef _HASHTABLE_H
#define _HASHTABLE_H

#include <stdint.h>

#define CACHE_LINE_SIZE 64
#define MAX_PREFETCH 16

#define MEM_DELAY 200

#define ITEMS_COUNT 1000000
#define TABLE_SIZE (ITEMS_COUNT / 8)

#define ITER_COUNT 10
#define STATES_COUNT 1024

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

   char *chain_ref;
   char *data_refs[12];
   char **data_ref;
   char *last_chain_ref;
   int num;
   int offset;
   } FProcessState;

#define BIG_SET_SIZE 1024
typedef struct FProcessStateBigSetTg
   {
   FProcessState *states[BIG_SET_SIZE];
   int first;
   int last;
   int count;
   } FProcessStateBigSet;

#if MAX_PREFETCH > 16
#define SMALL_SET_SIZE MAX_PREFETCH
#else
#define SMALL_SET_SIZE 16
#endif

typedef struct FProcessStateSmallSetTg
   {
   int offsets[SMALL_SET_SIZE];
   FProcessState *states[SMALL_SET_SIZE];
   int count;
   } FProcessStateSmallSet;

typedef struct FOutputTg
   {
   char *start;
   char *pos;
   } FOutput;

typedef struct FHashTableTg {
   char *table;
   uint32_t table_size;
   uint32_t unlocated;

   uint32_t *data;
   uint32_t data_pos;

   uint64_t tick;
   int pcnt;

#ifdef DEBUG_COUNTERS
   int fp;
   int found;
   int not_found;
   int not_found2;
   int chain;
#endif

   FProcessStateBigSet *unpref;
   FProcessStateBigSet *empty;
   FProcessStateBigSet *loaded;
   FProcessStateBigSet *h_req;
   FProcessStateBigSet *d_req;
   } FHashTable;

#endif // !_HASHTABLE_H