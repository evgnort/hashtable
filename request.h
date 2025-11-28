# Copyright (C) Evgeniy Buevich

#ifndef _REQUEST_H
#define _REQUEST_H

#include "defines.h"

typedef struct FPStateParamsTg
   {
   uint32_t offset;
   uint32_t inc;
   } FPStateParams;

typedef struct FProcessStateTg {
   char key_buf[64]; // One cache line

   // Here can be up to 16 bytes overwrite, so hashing fields go first
   uint32_t chain_cont;
   uint32_t items_mask;

   char *cur_data_ref;
   uint32_t *cur_chain_ref;
   uint32_t *next_chain_ref;
   
   uint64_t tick;

   FPStateParams sp;

   char *value_start;
   int key_size;
   int value_size;
   } FProcessState;

static inline void reset_state_search_data(FProcessState *state)
   {
   _mm256_storeu_si256((__m256i *)&state->chain_cont,_mm256_setzero_si256());
   state->tick = 0LL;
   }

#define SET_STATESREF_COUNT 512

typedef struct FNormalRqSetTg
   {
   int indexes[SET_STATESREF_COUNT];
   int first;
   int last;
   int count; // Used only in RS_Loaded, unsupported in other queues
   } FNormalRqSet;

static inline FProcessState *nrs_get_first(FProcessState *base_states,FNormalRqSet *set)
   {
   return (FProcessState *)(((char *)base_states) + set->indexes[set->first++]);
   }

typedef struct FHashedRqSetTg
   {
   int indexes[SET_STATESREF_COUNT];
   uint32_t hashes[SET_STATESREF_COUNT];
   int first;
   int last;
   } FHashedRqSet;


#endif // !_REQUEST_H

