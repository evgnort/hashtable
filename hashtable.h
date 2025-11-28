# Copyright (C) Evgeniy Buevich

#ifndef _HASHTABLE_H
#define _HASHTABLE_H

#include <stdint.h>

#include "request.h"

#define CACHE_LINE_SIZE 64
#define PAGE_SIZE 4096
#define MAX_PREFETCH 16

#define MEM_DELAY 200

#define DEF_ITEMS_COUNT 1000000

typedef struct FParseParamsTg FParseParams;

// 128 byte on state data

typedef struct FRequestSetTg
   {
   FProcessState *states[STATES_COUNT];
   int first;
   int last;
   int count;
   } FRequestSet;

extern int lut8[256][8];

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

typedef enum ERequestStateTg {
   RS_Empty,
   RS_Loaded,
   RS_Unpref,
   RS_Header,
   RS_Data,
   RS_MAX
   } ERequestState;

typedef struct FHashTableTg {
   char *table;
   uint32_t table_size;
   uint32_t unlocated;

   uint32_t *data;
   uint32_t data_pos;

   uint64_t tick;

   int pcnt;
   uint32_t items_count;

   uint32_t table_large;
   uint32_t data_large;

   FProcessState *states;
   char *value_store;

#ifdef DEBUG_COUNTERS
   int fp;
   int found;
   int not_found;
   int not_found2;
   int chain;
#endif

   FRequestSet sets[RS_MAX];
   } FHashTable;

typedef struct FParseStateTg FParseState;

void reset_state(FParseState *pstate,FProcessState *state);
int64_t get_nanotime(void);

void make_hash(FHashTable *ht,FProcessState *state,int ssize);

#endif // !_HASHTABLE_H