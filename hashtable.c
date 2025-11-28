# Copyright (C) Evgeniy Buevich

#include "port.h"

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>

#include <setjmp.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

//#define LOGFILE
//#define DEBUG_COUNTERS
#include "utils.h"

#include "hashtable.h"
#include "csvparse.h"

#define HASH_PRIME 591798841
#define HASH_INITIAL 2166136261

static inline FProcessState *get_state(FRequestSet *set)
   { 
   set->count--; 
   FProcessState *rv = set->states[set->first & STATES_COUNT_MASK];
   set->first++;
   return rv;
   }

static inline FProcessState *get_state_link(FRequestSet *set,int num)
   { 
   return set->states[(set->first + num) & STATES_COUNT_MASK];
   }

static inline void add_state(FRequestSet *set,FProcessState *state)
   {
//   assert(set->count < STATES_COUNT);
   set->count++;
   set->states[set->last & STATES_COUNT_MASK] = state;
   set->last++;
   }

void reset_set(FRequestSet *set)
   {
   set->first = set->last = set->count = 0;
   }

static inline void add_unpref(FHashTable *ht,FProcessState *state)
   {
   add_state(&ht->sets[RS_Unpref],state);
   assert(state->cur_data_ref);
   }

static inline void process_unpref(FHashTable *ht)
   {
   if (ht->sets[RS_Unpref].count)
      {
      FProcessState *state = get_state(&ht->sets[RS_Unpref]);
      _mm_prefetch(state->cur_data_ref,_MM_HINT_T2);
      state->tick = ht->tick + MEM_DELAY;
      add_state(&ht->sets[RS_Data - (state->cur_data_ref == (char *)state->next_chain_ref)],state);
      assert(state->cur_data_ref);
      return;
      }
   ht->pcnt--;
   }

static inline void max_process_unpref(FHashTable *ht)
   {
   while (ht->sets[RS_Unpref].count && ht->pcnt < MAX_PREFETCH)
      {
      FProcessState *state = get_state(&ht->sets[RS_Unpref]);
      _mm_prefetch(state->cur_data_ref,_MM_HINT_T2);
      state->tick = ht->tick + MEM_DELAY;
      add_state(&ht->sets[RS_Data - (state->cur_data_ref == (char *)state->next_chain_ref)],state);
      assert(state->cur_data_ref);
      ht->pcnt++;
      }
   }

static inline int update_state_refs(FHashTable *ht,FProcessState *state,uint32_t mask,int sp)
   {
   state->items_mask = _blsr_u32(mask);

   if (!(mask + state->cur_chain_ref[15]))
      { // Unpredictable branch
      add_state(&ht->sets[RS_Empty],state);
      return 1;
      }
   ht->tick += 4;

   state->next_chain_ref = state->cur_chain_ref[15] ? (uint32_t *)&ht->table[state->cur_chain_ref[15] * CACHE_LINE_SIZE] : NULL;
   state->cur_data_ref = mask ? (char *)&ht->data[state->cur_chain_ref[_tzcnt_u32(mask) + sp]] : (char *)state->next_chain_ref;
   add_unpref(ht,state);
   return 0;
   }

static inline int update_state_refs_prefetch(FHashTable *ht,FProcessState *state,uint32_t mask,int sp)
   {
   state->items_mask = _blsr_u32(mask);
   ht->tick += 5;

   if (!(mask + state->cur_chain_ref[15]))
      { // Unpredictable branch
      add_state(&ht->sets[RS_Empty],state);
      return 1;
      }
   ht->tick += 4;

   state->next_chain_ref = state->cur_chain_ref[15] ? (uint32_t *)&ht->table[state->cur_chain_ref[15] * CACHE_LINE_SIZE] : NULL;
   state->cur_data_ref = mask ? (char *)&ht->data[state->cur_chain_ref[_tzcnt_u32(mask) + sp]] : (char *)state->next_chain_ref;

   assert(state->cur_data_ref);
   _mm_prefetch(state->cur_data_ref,_MM_HINT_T2);
   state->tick = ht->tick + MEM_DELAY;
   add_state(&ht->sets[RS_Data - (state->cur_data_ref == (char *)state->next_chain_ref)],state);
   return 0;
   }

static inline void make_hashes16(FHashTable *ht,int *base,FProcessStateSmallSet *reqs,int ssize)
   {
   __m256i indexes1 = _mm256_loadu_si256((__m256i *)&reqs->offsets[0]);
   __m256i indexes2 = _mm256_loadu_si256((__m256i *)&reqs->offsets[8]);
   __m256i a1 = _mm256_i32gather_epi32(base,indexes1,1);
   __m256i a2 = _mm256_i32gather_epi32(base,indexes2,1);
   __m256i incs = _mm256_set1_epi32(4);
   indexes1 = _mm256_add_epi32(indexes1,incs);
   indexes2 = _mm256_add_epi32(indexes2,incs);
   int size = ssize;

   __m256i hashes11,hashes12;
   __m256i hashes21,hashes22;
   __m256i leftshifts = _mm256_set1_epi32(5);
   __m256i rightshifts = _mm256_set1_epi32(27);
   __m256i primes = _mm256_set1_epi32(HASH_PRIME);
   __m256i hashes1 = _mm256_set1_epi32(HASH_INITIAL);
   __m256i hashes2 = _mm256_set1_epi32(HASH_INITIAL);
   hashes1 = _mm256_xor_si256(hashes1,a1);
   hashes2 = _mm256_xor_si256(hashes2,a2);
   hashes1 = _mm256_mullo_epi32(hashes1,primes);
   hashes2 = _mm256_mullo_epi32(hashes2,primes);
   while (size--)
      {
      a1 = _mm256_i32gather_epi32(base,indexes1,1);
      a2 = _mm256_i32gather_epi32(base,indexes2,1);
      indexes1 = _mm256_add_epi32(indexes1,incs);
      indexes2 = _mm256_add_epi32(indexes2,incs);

      hashes11 = _mm256_sllv_epi32(hashes1,leftshifts);
      hashes21 = _mm256_sllv_epi32(hashes2,leftshifts);
      hashes12 = _mm256_srlv_epi32(hashes1,rightshifts);
      hashes22 = _mm256_srlv_epi32(hashes2,rightshifts);
      hashes1 = _mm256_or_si256(hashes11,hashes12);
      hashes2 = _mm256_or_si256(hashes21,hashes22);
      hashes1 = _mm256_xor_si256(hashes1,a1);
      hashes2 = _mm256_xor_si256(hashes2,a2);
      hashes1 = _mm256_mullo_epi32(hashes1,primes);
      hashes2 = _mm256_mullo_epi32(hashes2,primes);
      }

   uint32_t i,res[16];
   _mm256_storeu_si256((__m256i *)&res[0],hashes1);
   _mm256_storeu_si256((__m256i *)&res[8],hashes2);
   for(i = 0; i < 16; i++)
      {
      reqs->states[i]->next_chain_ref = (uint32_t *)&ht->table[(res[i] % ht->table_size + 1) * CACHE_LINE_SIZE];
      reqs->states[i]->cur_data_ref = (char *)reqs->states[i]->next_chain_ref;
      }
   ht->tick += 102 + 21 * ssize;
   }

void make_hash(FHashTable *ht,FProcessState *state,int ssize)
   {
   int pos = 0;
   int size = ssize;
   uint32_t hash = HASH_INITIAL;
   uint32_t a = *(uint32_t *)&state->key_buf[0];
   hash ^= a;
   hash *= HASH_PRIME;

   while (size--)
      {
      pos += 4;
      uint32_t hash1 = hash << 5;
      uint32_t hash2 = hash >> 27;
      hash = hash1 | hash2;
      a = *(uint32_t *)&state->key_buf[pos];
      hash ^= a;
      hash *= HASH_PRIME;
      }
   state->next_chain_ref = (uint32_t *)&ht->table[(hash % ht->table_size + 1) * CACHE_LINE_SIZE];
   state->cur_data_ref = (char *)state->next_chain_ref;
   ht->tick += 20 + 2 * ssize;
   }

int lut8[256][8] = {0};

const int pnums[8] = {0,1,4,5,2,3,6,7};
void make_lut8(void)
   {
   int i;
   for (i = 0; i < 256; i++)
      {
      int v = i, bn, pos = 0, offt = 0;
      while ((bn = _tzcnt_u32(v)) != 32)
         {
         lut8[i][pnums[pos]] = offt + bn;
         pos++;
         v >>= bn + 1;
         offt += bn + 1;
         }
      }
   }

// 8 keys in line
static inline int look_ht_8(FHashTable *ht,FProcessState *state)
   {
   uint32_t *chain_ref = state->next_chain_ref;
   state->cur_chain_ref = chain_ref;
   __m256i headers = _mm256_loadu_si256((__m256i *)chain_ref);
   process_unpref(ht);
   
   __m256i search = _mm256_set1_epi32(*((int *)state->key_buf));
   __m256i cmpres = _mm256_cmpeq_epi32(headers,search);
   uint32_t res = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres)) & 0x7F;
   state->items_mask = _blsr_u32(res);
   ht->tick += 5;

   return update_state_refs(ht,state,res,8);
   }

static inline int look_ht_12(FHashTable *ht,FProcessState *state)
   {
   uint32_t *chain_ref = state->next_chain_ref;
   state->cur_chain_ref = chain_ref;
   __m256i headers = _mm256_loadu_si256((__m256i *)chain_ref);
   process_unpref(ht);
   
   __m256i search = _mm256_set1_epi8(state->key_buf[0]);
   uint32_t bits = chain_ref[3];
   uint32_t bit8 = 0x7FF * (state->key_buf[1] & 0x1);
   uint32_t bit9 = 0x7FF * ((state->key_buf[1] & 0x2) >> 1);
//   uint32_t src_bits = 0x7FF * ((state->key_buf[1] & 0x1) + ((state->key_buf[1] & 0x2) << 15));
//   uint32_t bit_res = ~(chain_ref[3] ^ src_bits); // Low half for bit8, up half for bit 9

   __m256i cmpres = _mm256_cmpeq_epi8(headers,search);
   uint32_t res = _mm256_movemask_epi8(cmpres);
   res = res & ~(bits ^ bit8) & (~(bits >> 16) ^ bit9) & 0x7FF;

//   res = res & bit_res & (bit_res >> 16) & 0x7FF;
   ht->tick += 7;

   return update_state_refs(ht,state,res,4);
   }

void ht_add_8(FHashTable *ht,FProcessState *state)
   {
   int isize = state->value_size / 4 + ((state->value_size % 4) ? 1 : 0);
   int align = ht->data_pos % 16;
   ht->data_pos += 15 - align;
   uint32_t dpos = ht->data_pos;
   ht->data_pos += isize + 1;
   ht->data[dpos++] = isize;
   memcpy(&ht->data[dpos],state->value_start,isize * 4);

   uint32_t *chain_ref = state->cur_chain_ref;
   __m256i headers,cmpres0;

ht_add_8_repeat:
   headers = _mm256_loadu_si256((__m256i *)chain_ref);
   cmpres0 = _mm256_cmpeq_epi32(headers,_mm256_setzero_si256());
   int res0 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres0)) & 0xFF;
   int pos = _tzcnt_u32(res0);

   if (pos == 7)
      {
      if (chain_ref[15])
         {
         chain_ref = (uint32_t *)&ht->table[chain_ref[15] * CACHE_LINE_SIZE];
         goto ht_add_8_repeat;
         }
      chain_ref[15] = ht->unlocated;
      chain_ref = (uint32_t *)&ht->table[ht->unlocated * CACHE_LINE_SIZE];
      ht->unlocated++;
      pos = 0;
      }

   chain_ref[pos] = *((int *)state->key_buf);
   chain_ref[pos+8] = dpos;
   ht->tick += 10;
   }

void ht_add_12(FHashTable *ht,FProcessState *state)
   {
   int isize = state->value_size / 4 + ((state->value_size % 4) ? 1 : 0);
   int align = ht->data_pos % 16;
   ht->data_pos += 15 - align;
   uint32_t dpos = ht->data_pos;
   ht->data_pos += isize + 1;
   ht->data[dpos++] = isize;
   memcpy(&ht->data[dpos],state->value_start,isize * 4);

   uint32_t *chain_ref = state->cur_chain_ref;
   uint32_t bits;
   __m256i headers,cmpres0;

ht_add_12_repeat:
   bits = chain_ref[3];
   headers = _mm256_loadu_si256((__m256i *)chain_ref);
   cmpres0 = _mm256_cmpeq_epi8(headers,_mm256_setzero_si256());
   int res0 = _mm256_movemask_epi8(cmpres0);
   res0 &= ~(bits | (bits >> 16));
   res0 &= 0xFFF;
   int pos = _tzcnt_u32(res0);

   if (pos == 11)
      {
      if (chain_ref[15])
         {
         chain_ref = (uint32_t *)&ht->table[chain_ref[15] * CACHE_LINE_SIZE];
         goto ht_add_12_repeat;
         }
      chain_ref[15] = ht->unlocated;
      chain_ref = (uint32_t *)&ht->table[ht->unlocated * CACHE_LINE_SIZE];
      ht->unlocated++;
      pos = 0;
      }

   chain_ref[3] &= ~(0x10001 << pos);
   chain_ref[3] |= (state->key_buf[1] & 0x1) << pos;
   chain_ref[3] |= (state->key_buf[1] & 0x2) << (pos + 16 - 1);
   ((char *)chain_ref)[pos] = state->key_buf[0];
   chain_ref[pos+4] = dpos;
   ht->tick += 10;
   }

int look_ht_8_add(FHashTable *ht,FProcessState *state)
   {
   uint32_t *chain_ref = state->next_chain_ref;
   state->cur_chain_ref = chain_ref;

   __m256i headers = _mm256_loadu_si256((__m256i *)chain_ref);
   process_unpref(ht);
   
   __m256i search = _mm256_set1_epi32(*((int *)state->key_buf));
   __m256i cmpres = _mm256_cmpeq_epi32(headers,search);
   uint32_t res = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres)) & 0x7F;

   ht->tick += 10;

   if (!update_state_refs(ht,state,res,8))
      return 0;
   ht_add_8(ht,state);
   return 1;
   }

static inline int look_ht_12_add(FHashTable *ht,FProcessState *state)
   {
   uint32_t *chain_ref = state->next_chain_ref;
   state->cur_chain_ref = chain_ref;

   __m256i headers = _mm256_loadu_si256((__m256i *)chain_ref);
   process_unpref(ht);
   
   __m256i search = _mm256_set1_epi8(state->key_buf[0]);
   uint32_t bits = chain_ref[3];
   uint32_t bit8 = 0x7FF * (state->key_buf[1] & 0x1);
   uint32_t bit9 = 0x7FF * ((state->key_buf[1] & 0x2) >> 1);
   __m256i cmpres = _mm256_cmpeq_epi8(headers,search);

   uint32_t res = _mm256_movemask_epi8(cmpres);
   res &= ~(bits ^ bit8) & (~(bits >> 16) ^ bit9) & 0x7FF;
    
   ht->tick += 12;

   if (!update_state_refs(ht,state,res,4))
      return 0;
   ht_add_12(ht,state);
   return 1;
   }

// 4 by step
int look_in_data_4_states(FHashTable *ht,FOutput *output,int sp)
   {
   uint64_t res_mask,key_mask;
   int rv = 0;

   FProcessState *state0 = get_state(&ht->sets[RS_Data]);
   __m256i data00 = _mm256_loadu_si256((__m256i *)(state0->cur_data_ref));
   __m256i data01 = _mm256_loadu_si256((__m256i *)(state0->cur_data_ref + 32));
   __m256i key00 = _mm256_loadu_si256((__m256i *)state0->key_buf); 
   __m256i key01 = _mm256_loadu_si256((__m256i *)&state0->key_buf[32]); 

   FProcessState *state1 = get_state(&ht->sets[RS_Data]);
   __m256i data10 = _mm256_loadu_si256((__m256i *)(state1->cur_data_ref));
   __m256i data11 = _mm256_loadu_si256((__m256i *)(state1->cur_data_ref + 32));
   __m256i key10 = _mm256_loadu_si256((__m256i *)state1->key_buf); 
   __m256i key11 = _mm256_loadu_si256((__m256i *)&state1->key_buf[32]); 

   key_mask = (1LL << state0->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key01,data01)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key00,data00));
   if ((res_mask & key_mask) == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)output->pos,key00), _mm256_storeu_si256((__m256i *)(output->pos+32),key01);
      output->pos[state0->key_size] = '\n';
      output->pos += state0->key_size + 1;
      add_state(&ht->sets[RS_Empty],state0);
      rv++;
      }
   else
      rv += update_state_refs_prefetch(ht,state0,state0->items_mask,sp);

   FProcessState *state2 = get_state(&ht->sets[RS_Data]);
   __m256i data20 = _mm256_loadu_si256((__m256i *)(state2->cur_data_ref));
   __m256i data21 = _mm256_loadu_si256((__m256i *)(state2->cur_data_ref + 32));
   __m256i key20 = _mm256_loadu_si256((__m256i *)state2->key_buf); 
   __m256i key21 = _mm256_loadu_si256((__m256i *)&state2->key_buf[32]);  

   key_mask = (1LL << state1->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key11,data11)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key10,data10));
   if ((res_mask & key_mask) == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)output->pos,key10), _mm256_storeu_si256((__m256i *)(output->pos+32),key11);
      output->pos[state1->key_size] = '\n';
      output->pos += state1->key_size + 1;
      add_state(&ht->sets[RS_Empty],state1);
      rv++;
      }
   else
      rv += update_state_refs_prefetch(ht,state1,state1->items_mask,sp);

   FProcessState *state3 = get_state(&ht->sets[RS_Data]);
   __m256i data30 = _mm256_loadu_si256((__m256i *)(state3->cur_data_ref));
   __m256i data31 = _mm256_loadu_si256((__m256i *)(state3->cur_data_ref + 32));
   __m256i key30 = _mm256_loadu_si256((__m256i *)state3->key_buf); 
   __m256i key31 = _mm256_loadu_si256((__m256i *)&state3->key_buf[32]); 

   key_mask = (1LL << state2->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key21,data21)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key20,data20));
   if ((res_mask & key_mask) == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)output->pos,key20), _mm256_storeu_si256((__m256i *)(output->pos+32),key21);
      output->pos[state2->key_size] = '\n';
      output->pos += state2->key_size + 1;
      add_state(&ht->sets[RS_Empty],state2);
      rv++;
      }
   else
      rv += update_state_refs_prefetch(ht,state2,state2->items_mask,sp);

   key_mask = (1LL << state3->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key31,data31)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key30,data30));
   if ((res_mask & key_mask) == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)output->pos,key30), _mm256_storeu_si256((__m256i *)(output->pos+32),key31);
      output->pos[state3->key_size] = '\n';
      output->pos += state3->key_size + 1;
      add_state(&ht->sets[RS_Empty],state3);
      rv++;
      }
   else
      rv += update_state_refs_prefetch(ht,state3,state3->items_mask,sp);

   ht->tick += 38;
   ht->pcnt -= rv;
   return rv;
   }

int look_in_data(FHashTable *ht,FOutput *output, int sp)
   {
   uint64_t res_mask,key_mask;

   FProcessState *state0 = get_state(&ht->sets[RS_Data]);
   __m256i data00 = _mm256_loadu_si256((__m256i *)(state0->cur_data_ref));
   __m256i data01 = _mm256_loadu_si256((__m256i *)(state0->cur_data_ref + 32));
   __m256i key00 = _mm256_loadu_si256((__m256i *)state0->key_buf); 
   __m256i key01 = _mm256_loadu_si256((__m256i *)&state0->key_buf[32]); 

   key_mask = (1LL << state0->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key01,data01)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key00,data00));
   ht->tick += 20;
   if ((res_mask & key_mask) == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)output->pos,key00), _mm256_storeu_si256((__m256i *)(output->pos+32),key01);
      output->pos[state0->key_size] = '\n';
      output->pos += state0->key_size + 1;
      add_state(&ht->sets[RS_Empty],state0);
      ht->pcnt--;
      return 1;
      }
   int res = update_state_refs_prefetch(ht,state0,state0->items_mask,sp);
   ht->pcnt -= res;
   return res;
   }

int look_in_data_add(FHashTable *ht,FProcessState *state, int sp)
   {
   uint64_t res_mask,key_mask;

   __m256i data0 = _mm256_loadu_si256((__m256i *)(state->cur_data_ref));
   __m256i data1 = _mm256_loadu_si256((__m256i *)(state->cur_data_ref + 32));
   process_unpref(ht);
   ht->tick += 15;
   __m256i key0 = _mm256_loadu_si256((__m256i *)state->key_buf); 
   __m256i key1 = _mm256_loadu_si256((__m256i *)&state->key_buf[32]); 

   key_mask = (1LL << state->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key1,data1)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key0,data0));
   if ((res_mask & key_mask) == key_mask)
      return add_state(&ht->sets[RS_Empty],state),0;
   return update_state_refs(ht,state,state->items_mask,sp);
   }

void reset_state(FParseState *pstate,FProcessState *state)
   {
   state->next_chain_ref = state->cur_chain_ref = NULL;
   state->cur_data_ref = NULL;
   pstate->col_num = 0;
   state->key_size = 0;
   pstate->key_size_ref = &state->key_size;
   pstate->value_size_ref = &state->value_size;
   pstate->key_pos = pstate->key_buf = state->key_buf;
   pstate->value_pos = pstate->value_start = state->value_start;
   state->sp.inc = 0;
   }

void reset_ht_search(FHashTable *ht,FProcessState *states)
   {
   int i;
   for (i = 0; i < RS_MAX; i++)
      reset_set(&ht->sets[i]);
   ht->pcnt = 0;
   ht->tick = 0;
   for (i = 0; i < STATES_COUNT; i++)
      add_state(&ht->sets[RS_Empty],&states[i]);
#ifdef DEBUG_COUNTERS
   ht->found = ht->not_found = ht->not_found2 = ht->chain = ht->fp = 0;
#endif
   }

void reset_ht_data(FHashTable *ht)
   {
   ht->unlocated = ht->table_size + 1;
   memset(ht->table,0,ht->table_size * 2 * CACHE_LINE_SIZE);
   memset(ht->data,0,ht->items_count * 128);
   ht->data_pos = 31;
   }

double NANOTIME_COST = 0;

void process_add_8(FHashTable *ht,FSource *source, FProcessState *states)
   {
   FProcessState *state;
   int i,j;

   FProcessStateSmallSet set_by_size[16] = {0};
   reset_ht_data(ht);
   reset_ht_search(ht,states);
   FParseState pstate;

   while (source->input_pos < source->input_end)
      {
      while (ht->sets[RS_Empty].count && source->input_pos < source->input_end)
         {
         state = get_state(&ht->sets[RS_Empty]);
         reset_state(&pstate,state);
         process_row(source,&pstate);

         int ks = (state->key_size - 1) / 4;
         set_by_size[ks].offsets[set_by_size[ks].count] = state->sp.offset + offsetof(FProcessState,key_buf);
         set_by_size[ks].states[set_by_size[ks].count++] = state;

         if (set_by_size[ks].count == 16)
            {
            make_hashes16(ht,(int *)states,&set_by_size[ks],ks);
            for(j = 0; j < 16; j++)
               {
               add_unpref(ht,set_by_size[ks].states[j]);
               }
            set_by_size[ks].count = 0;
            }
         if (source->input_pos < source->input_end) break;
         }
      max_process_unpref(ht);
      while(ht->sets[RS_Header].count && get_state_link(&ht->sets[RS_Header],0)->tick <= ht->tick)
         look_ht_8_add(ht,get_state(&ht->sets[RS_Header]));

      while(ht->sets[RS_Data].count && get_state_link(&ht->sets[RS_Data],0)->tick <= ht->tick)
         {
         state = get_state(&ht->sets[RS_Data]);
         if (look_in_data_add(ht,state,8))
            ht_add_8(ht,state);
         }
      ht->tick += 20;
      }
   // Process tails
   for (i = 0; i < 8; i++)
      {
      for (j = 0; j < set_by_size[i].count; j++)
         {
         make_hash(ht,set_by_size[i].states[j],i);
         add_unpref(ht,set_by_size[i].states[j]);
         }
      }
   while (ht->sets[RS_Empty].count != STATES_COUNT)
      {
      max_process_unpref(ht);
      while(ht->sets[RS_Header].count && get_state_link(&ht->sets[RS_Header],0)->tick <= ht->tick)
         look_ht_8_add(ht,get_state(&ht->sets[RS_Header]));

      while(ht->sets[RS_Data].count && get_state_link(&ht->sets[RS_Data],0)->tick <= ht->tick)
         {
         state = get_state(&ht->sets[RS_Data]);
         if (look_in_data_add(ht,state,8))
            ht_add_8(ht,state);
         }
      ht->tick += 20;
      }
   }

void process_add_12(FHashTable *ht,FSource *source, FProcessState *states)
   {
   FProcessState *state;
   int i,j;
   FProcessStateSmallSet set_by_size[16] = {0};

   reset_ht_data(ht);
   reset_ht_search(ht,states);
   FParseState pstate;

   while (source->input_pos < source->input_end)
      {
      while (ht->sets[RS_Empty].count && source->input_pos < source->input_end)
         {
         state = get_state(&ht->sets[RS_Empty]);
         reset_state(&pstate,state);
         process_row(source,&pstate);

         int ks = (state->key_size - 1) / 4;

         set_by_size[ks].offsets[set_by_size[ks].count] = state->sp.offset + offsetof(FProcessState,key_buf);
         set_by_size[ks].states[set_by_size[ks].count++] = state;

         if (set_by_size[ks].count == 16)
            {
            make_hashes16(ht,(int *)states,&set_by_size[ks],ks);
            for(j = 0; j < 16; j++)
               add_unpref(ht,set_by_size[ks].states[j]);
            set_by_size[ks].count = 0;
            }
         if (source->input_pos < source->input_end) break;
         }
      max_process_unpref(ht);
      while(ht->sets[RS_Header].count && get_state_link(&ht->sets[RS_Header],0)->tick <= ht->tick)
         look_ht_12_add(ht,get_state(&ht->sets[RS_Header]));
      while(ht->sets[RS_Data].count && get_state_link(&ht->sets[RS_Data],0)->tick <= ht->tick)
         {
         state = get_state(&ht->sets[RS_Data]);
         if (look_in_data_add(ht,state,4))
            ht_add_12(ht,state);
         }
      ht->tick += 20;
      }
   // Process tails
   for (i = 0; i < 8; i++)
      {
      for (j = 0; j < set_by_size[i].count; j++)
         {
         make_hash(ht,set_by_size[i].states[j],i);
         add_unpref(ht,set_by_size[i].states[j]);
         }
      }
   while (ht->sets[RS_Empty].count != STATES_COUNT)
      {
      max_process_unpref(ht);
      while(ht->sets[RS_Header].count && get_state_link(&ht->sets[RS_Header],0)->tick <= ht->tick)
         look_ht_12_add(ht,get_state(&ht->sets[RS_Header]));

      while(ht->sets[RS_Data].count && get_state_link(&ht->sets[RS_Data],0)->tick <= ht->tick)
         {
         state = get_state(&ht->sets[RS_Data]);
         if (look_in_data_add(ht,state,4))
            ht_add_12(ht,state);
         }
      ht->tick += 20;
      }
   }

double process_search_8(FHashTable *ht,FSource *source, FProcessState *states,FOutput *output)
   {
   FProcessState *state;
   int64_t tm = 0;
   int j;
   FProcessStateSmallSet set_by_size[16] = {0};
   int processed = 0;
   FParseState pstate;

   reset_ht_search(ht,states);

   while (source->input_pos < source->input_end)
      {
      while (ht->sets[RS_Empty].count && source->input_pos < source->input_end)
         {
         state = get_state(&ht->sets[RS_Empty]);
         reset_state(&pstate,state);
         process_row(source,&pstate);
         add_state(&ht->sets[RS_Loaded],state);
         ht->tick += 200;
         }
      int64_t t1 = get_nanotime();
      while (ht->sets[RS_Loaded].count)
         {
         state = get_state(&ht->sets[RS_Loaded]);
         int ks = (state->key_size - 1) / 4;
         set_by_size[ks].offsets[set_by_size[ks].count] = state->sp.offset + offsetof(FProcessState,key_buf);
         set_by_size[ks].states[set_by_size[ks].count++] = state;

         if (set_by_size[ks].count == 16)
            {
            make_hashes16(ht,(int *)states,&set_by_size[ks],ks);
            for(j = 0; j < 16; j++)
               add_unpref(ht,set_by_size[ks].states[j]);
            set_by_size[ks].count = 0;
            }

         max_process_unpref(ht);
         while(ht->sets[RS_Header].count && get_state_link(&ht->sets[RS_Header],0)->tick <= ht->tick)
            processed += look_ht_8(ht,get_state(&ht->sets[RS_Header]));
         while(ht->sets[RS_Data].count >= 4 && get_state_link(&ht->sets[RS_Data],3)->tick <= ht->tick)
            processed += look_in_data_4_states(ht,output,8);
         ht->tick += 20;
         }
      tm += get_nanotime() - t1;
      }

   return (double)tm/processed;
   }

double process_search_12(FHashTable *ht,FSource *source, FProcessState *states,FOutput *output)
   {
   FProcessState *state;
   int64_t tm = 0;
   int j;
   FProcessStateSmallSet set_by_size[16] = {0};
   int processed = 0;
   FParseState pstate;

   reset_ht_search(ht,states);

   while (source->input_pos < source->input_end)
      {
      while (ht->sets[RS_Empty].count && source->input_pos < source->input_end)
         {
         state = get_state(&ht->sets[RS_Empty]);
         reset_state(&pstate,state);
         process_row(source,&pstate);
         add_state(&ht->sets[RS_Loaded],state);
         ht->tick += 200;
         }
      int64_t t1 = get_nanotime();
      while (ht->sets[RS_Loaded].count)
         {
         state = get_state(&ht->sets[RS_Loaded]);

         int ks = (state->key_size - 1) / 4;
         set_by_size[ks].offsets[set_by_size[ks].count] = state->sp.offset + offsetof(FProcessState,key_buf);
         set_by_size[ks].states[set_by_size[ks].count++] = state;

         if (set_by_size[ks].count == 16)
            {
            make_hashes16(ht,(int *)states,&set_by_size[ks],ks);
            for(j = 0; j < 16; j++)
               {
               LOG_RECORD("%d added to unpref",set_by_size[ks].states[j]->offset / sizeof(FProcessState));
               add_unpref(ht,set_by_size[ks].states[j]);
               }
            set_by_size[ks].count = 0;
            }

         max_process_unpref(ht);
         while(ht->sets[RS_Header].count && get_state_link(&ht->sets[RS_Header],0)->tick <= ht->tick)
            processed += look_ht_12(ht,get_state(&ht->sets[RS_Header]));
         while(ht->sets[RS_Data].count && get_state_link(&ht->sets[RS_Data],0)->tick <= ht->tick)
            processed += look_in_data(ht,output,12);
         ht->tick += 20;
         }
      tm += get_nanotime() - t1;
      }

   return (double)tm/processed;
   }

double pipeline(FHashTable *ht, FSource *source, char *output, double *ctm, int stat);

typedef struct FItemTg
	{
	int next;
	int payload[(CACHE_LINE_SIZE - sizeof(int)) / sizeof(int)];
	} FItem;
	
#define ARRAY_SIZE 1024 * 1024
#define ARRAY_COUNT 8

FItem *init_array()
	{
	FItem *arr = aligned_alloc(CACHE_LINE_SIZE,ARRAY_SIZE * sizeof(FItem));
	int *numbers = (int *)malloc(ARRAY_SIZE * sizeof(int));
	int i;
	for (i = 0; i < ARRAY_SIZE - 1; i++)
		numbers[i] = i+1;
	int pos = 0;
	int cnt = ARRAY_SIZE - 1;
	for (i = 0; i < ARRAY_SIZE - 1; i++)
		{
		int npos = rand() % cnt;
		arr[pos].next = numbers[npos];
		pos = numbers[npos];
		numbers[npos] = numbers[--cnt];
		}
	arr[pos].next = 0;
	return arr;
	}
	
uint64_t traverse8(FItem **items,int *res)
	{
	int i,j;
	int ipos[ARRAY_COUNT] = {0};
	int val = rand();
   uint64_t t1 = get_nanotime();
	for (i = 0; i < 100000; i++)
		{
		ipos[0] = items[0][ipos[0]].next;
		ipos[1] = items[1][ipos[1]].next;
		ipos[2] = items[2][ipos[2]].next;
		ipos[3] = items[3][ipos[3]].next;
		ipos[4] = items[4][ipos[4]].next;
		ipos[5] = items[5][ipos[5]].next;
		ipos[6] = items[6][ipos[6]].next;
		ipos[7] = items[7][ipos[7]].next;
		}
	t1 = get_nanotime() - t1;
	for (j = 0; j < ARRAY_COUNT; j++)
		val += ipos[j];
	*res = val;
	return t1;
	}
	
void get_mem_time(void)
	{
	FItem *arrays[ARRAY_COUNT];
	srand((uint32_t)time(NULL));
	int i,res = 0;
	uint64_t tm;
	for (i = 0; i < ARRAY_COUNT; i++)
		arrays[i] = init_array();
	
	tm = traverse8(arrays,&res);
	
	for (i = 0; i < ARRAY_COUNT; i++)
		aligned_free(arrays[i]);
	printf("Memory delay: %.2f\n",(double)tm / 100000);
	}

int init_ht(FHashTable *ht,int items_count,int try_large_pages)
   {
   ht->items_count = items_count;
   ht->table_size = items_count / 8;
   ht->unlocated = ht->table_size + 1;
   ht->table_large = ht->data_large = 0;
   size_t lpsize = try_large_pages ? enable_large_pages() : 0;

   ht->table = NULL;
   ht->data = NULL;
   if (lpsize)
      {
      if ((ht->table = alloc_large_pages((ht->table_size * 2 + 1) * CACHE_LINE_SIZE,lpsize)))
         ht->table_large = 1;
      if ((ht->data = alloc_large_pages(ht->items_count * 128,lpsize)))
         ht->data_large = 1;
      }

   if (!ht->table && !(ht->table = aligned_alloc(CACHE_LINE_SIZE,(ht->table_size * 2 + 1) * CACHE_LINE_SIZE)))
      return 0;

   if (!ht->data && !(ht->data = aligned_alloc(CACHE_LINE_SIZE,ht->items_count * 128)))
      return 0;

   memset(ht->table,0,CACHE_LINE_SIZE);
   if (!(ht->value_store = (char *)malloc(ht->items_count * 128)))
      return 0;

   return 1;
   }

void deinit_ht(FHashTable *ht)
   {
   if (ht->table)
      {
      if (ht->table_large)
         free_large_pages(ht->table);
      else
         aligned_free(ht->table);
      }
   if (ht->data)
      {
      if (ht->data_large)
         free_large_pages(ht->data);   
      else
         aligned_free(ht->data);
      }
   free(ht->value_store);
   }

int main(int argc, char *argv[ ])
   {
   int i;
   FHashTable ht;
   char *filedata = NULL, *filedata2 = NULL;
   FOutput output = {0};
   FSource source,source2;
   FParseParams parse_params = {'\t','\t',0,3,__builtin_popcount(3)};
   source.pp = source2.pp = &parse_params;
   FProcessState *states = NULL;

   char *filename1 = "input1.csv", *filename2 = "input2.csv";
   int try_large_pages = 0, fnum = 0, items_count = DEF_ITEMS_COUNT, itcnt = 0;
   
   for (i = 1; i < argc; i++)
      {
      if (!strcmp(argv[i],"large"))
         try_large_pages = 1;
      else if (!strcmp(argv[i] + strlen(argv[i]) - 4,".csv"))
         {
         if (!fnum)
            filename1 = argv[i],fnum++;
         else if (fnum == 1)
            filename2 = argv[i],fnum++;
         }
      else if ((itcnt = atoi(argv[i])) && itcnt > 0)
         items_count = itcnt;
      }

   if (!init_ht(&ht,items_count,try_large_pages))
      goto main_exit;
   
   make_lut8();

   get_mem_time();
   printf("Table large: %d, Data large: %d\n",ht.table_large,ht.data_large);

   INIT_LOG("ht.log");

   off_t sz1 = file_size(filename1);
   if (sz1 < 0)
      return printf("file %s not found\n",filename1),1;

   off_t sz2 = file_size(filename2);
   if (sz2 < 0)
      return printf("file %s not found\n",filename2),1;

   filedata = read_file(filename1,sz1);
   filedata2 = read_file(filename2,sz2);

   if (!filedata || !filedata2)
      goto main_exit;


   states = aligned_alloc(CACHE_LINE_SIZE,sizeof(FProcessState) * STATES_COUNT);
   if (!states)
      goto main_exit;
      
   for (i = 0; i < STATES_COUNT; i++)
      {
      states[i].sp.offset = i * sizeof(FProcessState);
      states[i].value_start = &ht.value_store[i * 128];
      }

   output.start = output.pos = (char *)malloc(sz1 * 2 + 32);

// 8 per line
/*
   init_source(&source,filedata,sz1);
   process_add_8(&ht,&source,states);

   double tm8_1 = 10000000.0;
   for (i = 0; i < ITER_COUNT; i++)
      {
      init_source(&source,filedata,sz1);
      output.pos = output.start;
      double tm = process_search_8(&ht,&source,states,&output);
      if (tm < tm8_1)
         tm8_1 = tm;
      }

#ifdef DEBUG_COUNTERS
   printf("present  8: found %d, not found %d, long %d, false positive %d\n",ht.found,ht.not_found + ht.not_found2,ht.chain,ht.fp);
#endif

   double tm8_2 = 10000000.0;
   for (i = 0; i < ITER_COUNT; i++)
      {
      init_source(&source2,filedata2,sz2);
      output.pos = output.start;
      double tm = process_search_8(&ht,&source2,states,&output);
      if (tm < tm8_2)
         tm8_2 = tm;
      }

#ifdef DEBUG_COUNTERS
   printf("absent   8: found %d, not found %d, long %d, false positive %d\n",ht.found,ht.not_found + ht.not_found2,ht.chain,ht.fp);
#endif
*/
// 12 per line

   init_source(&source,filedata,sz1);
   process_add_12(&ht,&source,states);

   LOG_RECORD("12 PRESENT");

   double tm12_1 = 10000000.0;
   double tm12_1c = 10000000.0;
   printf("Present:\n");
   for (i = 0; i < ITER_COUNT; i++)
      {
      init_source(&source,filedata,sz1);
      source.input_pos = source.input_start;
      output.pos = output.start;
//      double tm = process_search_12(&ht,&source,states,&output);
      double ac;
      double tm = pipeline(&ht,&source,output.start,&ac,!i);
      if (tm < tm12_1) tm12_1 = tm;
      if (ac < tm12_1c) tm12_1c = ac;
      }

#ifdef DEBUG_COUNTERS
   printf("present 12: found %d, not found %d, long %d, false positive %d\n",ht.found,ht.not_found + ht.not_found2,ht.chain,ht.fp);
#endif

   LOG_RECORD("12 ABSENT");

   double tm12_2 = 10000000.0;
   double tm12_2c = 10000000.0;
   printf("Absent:\n");
   for (i = 0; i < ITER_COUNT; i++)
      {
      init_source(&source2,filedata2,sz2);
      source2.input_pos = source2.input_start;
      output.pos = output.start;
      double ac;
      double tm = pipeline(&ht,&source2,output.start,&ac,!i);
      if (tm < tm12_2) tm12_2 = tm;
      if (ac < tm12_2c) tm12_2c = ac;
      }

#ifdef DEBUG_COUNTERS
   printf("absent  12: found %d, not found %d, long %d, false positive %d\n",ht.found,ht.not_found + ht.not_found2,ht.chain,ht.fp);
#endif

//   printf("present  8: %.3f\n",tm8_1);
//   printf("absent   8: %.3f\n",tm8_2);
   printf("present 12: %.3f  %.1f\n",tm12_1,tm12_1c);
   printf("absent  12: %.3f  %.1f\n",tm12_2,tm12_2c);
//   printf("present ratio: %.3f\n",tm8_1/tm12_1);
//   printf("absent ratio: %.3f\n",tm8_2/tm12_2);

main_exit:
   deinit_ht(&ht);
   if (filedata) free(filedata);
   if (filedata2) free(filedata2);
   if (states) aligned_free(states);
   if (output.start) free(output.start);

   CLOSE_LOG;
   return 0;
   }
