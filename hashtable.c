#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
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

static inline FProcessState *get_state(FProcessStateBigSet *set)
   { 
   set->count--; 
   FProcessState *rv = set->states[set->first];
   set->first = (set->first + 1) % BIG_SET_SIZE;
   return rv;
   }

static inline FProcessState *get_state_link(FProcessStateBigSet *set,int num)
   { 
   return set->states[(set->first + num) % BIG_SET_SIZE];
   }

static inline void add_state(FProcessStateBigSet *set,FProcessState *state)
   {
   assert(set->count < BIG_SET_SIZE);
   set->count++;
   set->states[set->last] = state;
   set->last = (set->last + 1) % BIG_SET_SIZE;
   }

void reset_set(FProcessStateBigSet *set)
   {
   set->first = set->last = set->count = 0;
   }

static inline void add_unpref(FHashTable *ht,FProcessState *state)
   {
   add_state(ht->unpref,state);
   }

static inline void process_unpref(FHashTable *ht)
   {
   if (ht->unpref->count)
      {
      FProcessState *state = get_state(ht->unpref);
      _mm_prefetch(*state->data_ref,_MM_HINT_T2);
      state->tick = ht->tick + MEM_DELAY;
      add_state((state->data_ref == &state->chain_ref) ? ht->h_req : ht->d_req,state);
      LOG_RECORD("%d state %d prefetched, moved to %s, unpref %d, pcnt %d",ht->tick,state->num,(state->data_ref == &state->chain_ref)?"lh":"ld",ht->unpref->count,ht->pcnt);
      return;
      }
   ht->pcnt--;
   }

static inline void max_process_unpref(FHashTable *ht)
   {
   while (ht->unpref->count && ht->pcnt < MAX_PREFETCH)
      {
      FProcessState *state = get_state(ht->unpref);
      _mm_prefetch(*state->data_ref,_MM_HINT_T2);
      state->tick = ht->tick + MEM_DELAY;
      add_state((state->data_ref == &state->chain_ref) ? ht->h_req : ht->d_req,state);
      ht->pcnt++;
      LOG_RECORD("%d state %d prefetched, moved to %s, max, unpref %d, pcnt %d",ht->tick,state->num,(state->data_ref == &state->chain_ref)?"lh":"ld",ht->unpref->count,ht->pcnt);
      }
   }

static inline void make_hashes(FHashTable *ht,int *base,FProcessStateSmallSet *reqs,int size)
   {
   __m256i indexes = _mm256_loadu_si256((__m256i *)&reqs->offsets[0]);
   __m256i a = _mm256_i32gather_epi32(base,indexes,1);
   __m256i incs = _mm256_set1_epi32(4);
   indexes = _mm256_add_epi32(indexes,incs);
   size = (size - 1) * 2;
   __m256i b = _mm256_i32gather_epi32(base,indexes,1);
   indexes = _mm256_add_epi32(indexes,incs);

   __m256i hashes1,hashes2;
   __m256i leftshifts = _mm256_set1_epi32(5);
   __m256i rightshifts = _mm256_set1_epi32(27);
   __m256i primes = _mm256_set1_epi32(HASH_PRIME);
   __m256i hashes = _mm256_set1_epi32(HASH_INITIAL);
   hashes = _mm256_xor_si256(hashes,a);
   hashes = _mm256_mullo_epi32(hashes,primes);
   while (size--)
      {
      a = b;
      b = _mm256_i32gather_epi32(base,indexes,1);
      indexes = _mm256_add_epi32(indexes,incs);

      hashes1 = _mm256_sllv_epi32(hashes,leftshifts);
      hashes2 = _mm256_srlv_epi32(hashes,rightshifts);
      hashes = _mm256_or_si256(hashes1,hashes2);
      hashes = _mm256_xor_si256(hashes,a);
      hashes = _mm256_mullo_epi32(hashes,primes);
      }
   hashes1 = _mm256_sllv_epi32(hashes,leftshifts);
   hashes2 = _mm256_srlv_epi32(hashes,rightshifts);
   hashes = _mm256_or_si256(hashes1,hashes2);
   hashes = _mm256_xor_si256(hashes,b);
   hashes = _mm256_mullo_epi32(hashes,primes);

   uint32_t i,res[8];
   _mm256_storeu_si256((__m256i *)res,hashes);
   for(i = 0; i < 8; i++)
      {
      reqs->states[i]->chain_ref = &ht->table[(res[i] % ht->table_size) * CACHE_LINE_SIZE];
      LOG_RECORD("%d state %d hash calculation",ht->tick,reqs->states[i]->num);
      }
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
   int size = (ssize - 1) * 2;
   __m256i b1 = _mm256_i32gather_epi32(base,indexes1,1);
   __m256i b2 = _mm256_i32gather_epi32(base,indexes2,1);
   indexes1 = _mm256_add_epi32(indexes1,incs);
   indexes2 = _mm256_add_epi32(indexes2,incs);

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
      a1 = b1;
      a2 = b2;
      b1 = _mm256_i32gather_epi32(base,indexes1,1);
      b2 = _mm256_i32gather_epi32(base,indexes2,1);
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
   hashes11 = _mm256_sllv_epi32(hashes1,leftshifts);
   hashes21 = _mm256_sllv_epi32(hashes2,leftshifts);
   hashes12 = _mm256_srlv_epi32(hashes1,rightshifts);
   hashes22 = _mm256_srlv_epi32(hashes2,rightshifts);
   hashes1 = _mm256_or_si256(hashes11,hashes12);
   hashes2 = _mm256_or_si256(hashes21,hashes22);
   hashes1 = _mm256_xor_si256(hashes1,b1);
   hashes2 = _mm256_xor_si256(hashes2,b2);
   hashes1 = _mm256_mullo_epi32(hashes1,primes);
   hashes2 = _mm256_mullo_epi32(hashes2,primes);

   uint32_t i,res[16];
   _mm256_storeu_si256((__m256i *)&res[0],hashes1);
   _mm256_storeu_si256((__m256i *)&res[8],hashes2);
   for(i = 0; i < 16; i++)
      {
      reqs->states[i]->chain_ref = &ht->table[(res[i] % ht->table_size) * CACHE_LINE_SIZE];
      LOG_RECORD("%d state %d hash calculation",ht->tick,reqs->states[i]->num);
      }
   ht->tick += 102 + 21 * ssize;
   }

static inline void make_hash(FHashTable *ht,FProcessState *state,int ssize)
   {
   int pos = 0;
   int size = (ssize + 1) * 2 - 1;
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
   state->chain_ref =  &ht->table[(hash % ht->table_size) * CACHE_LINE_SIZE];
   ht->tick += 20 + 2 * ssize;
   }

int lut8[128][8] = {0};

const int pnums[8] = {0,1,4,5,2,3,6,7};
void make_lut8(void)
   {
   int i;
   for (i = 0; i < 128; i++)
      {
      int v = i, bn, pos = 0, offt = 0;
      while ((bn = __builtin_ffsl(v)))
         {
         lut8[i][pnums[pos]] = offt + bn - 1;
         pos++;
         v >>= bn;
         offt += bn;
         }
      }
   }

// 8 keys in line
static inline int look_ht_8(FHashTable *ht,FProcessState *state)
   {
   __m256i shifts = _mm256_set1_epi32(2);
   LOG_RECORD("Processed at %d, should be %d",ht->tick,state->tick);
   uint32_t *chain_ref = (uint32_t *)state->chain_ref;
   __m256i headers = _mm256_loadu_si256((__m256i *)state->chain_ref);
   process_unpref(ht);
   __m256i zeropad = _mm256_setzero_si256();
   __m256i base = _mm256_set_epi64x((uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data);
   
   __m256i search = _mm256_set1_epi32(*((int *)state->key_buf));
   __m256i cmpres = _mm256_cmpeq_epi32(headers,search);
   int res = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres)) & 0x7F;
   int bcnt = __builtin_popcount(res); 
   uint32_t nidx = chain_ref[15];
   ht->tick += 5;

   if (!(bcnt + nidx))
      { // Unpredictable branch
      LOG_RECORD("%d state %d not found in table",ht->tick,state->num);
      add_state(ht->empty,state);
#ifdef DEBUG_COUNTERS
      ht->not_found2++;
#endif
      return 1;
      }

   ht->tick += 5;
   __m256i refs = _mm256_loadu_si256((__m256i *)&chain_ref[8]);
   refs = _mm256_permutevar8x32_epi32(refs,_mm256_loadu_si256((__m256i *)&lut8[res][0]));

   __m256i links1 = _mm256_unpacklo_epi32(refs,zeropad);
   __m256i links2 = _mm256_unpackhi_epi32(refs,zeropad);
   links1 = _mm256_sllv_epi32(links1,shifts);
   links2 = _mm256_sllv_epi32(links2,shifts);

   links1 = _mm256_add_epi64(links1,base);
   links2 = _mm256_add_epi64(links2,base);

   _mm256_storeu_si256((__m256i *)&state->data_refs[0],links1);
   _mm256_storeu_si256((__m256i *)&state->data_refs[4],links2);
   state->chain_ref = nidx ? &ht->table[nidx * CACHE_LINE_SIZE] : NULL;
   state->data_ref = &state->data_refs[bcnt - 1];
   LOG_RECORD("%d state %d found in table, %d links, chain_ref %s",ht->tick,state->num,bcnt,state->chain_ref ? "present" : "absent");
   add_unpref(ht,state);
#ifdef DEBUG_COUNTERS
   if (state->data_ref == &state->chain_ref) ht->chain++;
#endif
   return 0;
   }

int lut12_1[16][8] = {0};

void make_lut12(void)
   {
   int i;
   for (i = 0; i < 16; i++)
      {
      int v = i, bn, pos = 0, offt = 0;;
      while ((bn = __builtin_ffsl(v)))
         {
         lut12_1[i][pos++] = offt + bn - 1;
         v >>= bn;
         offt += bn;
         }
      }
   }

static inline int look_ht_12(FHashTable *ht,FProcessState *state)
   {
   __m256i shifts = _mm256_set1_epi32(2);
   uint32_t *chain_ref = (uint32_t *)state->chain_ref;
   __m256i headers = _mm256_loadu_si256((__m256i *)state->chain_ref);
   process_unpref(ht);
   __m256i zeropad = _mm256_setzero_si256();
   __m256i base = _mm256_set_epi64x((uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data);
   
   __m256i search = _mm256_set1_epi8(state->key_buf[0]);
   uint32_t bits = chain_ref[3];
   uint32_t bit8 = 0x7FF * (state->key_buf[1] & 0x1);
   uint32_t bit9 = 0x7FF * ((state->key_buf[1] & 0x2) >> 1);
   __m256i cmpres = _mm256_cmpeq_epi8(headers,search);

   int res = _mm256_movemask_epi8(cmpres);
   res &= ~(bits ^ bit8) & (~(bits >> 16) ^ bit9) & 0x7FF;

   int bcnt = __builtin_popcount(res); 
   uint32_t nidx = chain_ref[15];
   ht->tick += 7;
   if (!(bcnt + nidx))
      { // Unpredictable branch
      LOG_RECORD("%d state %d not found in table",ht->tick,state->num);
      add_state(ht->empty,state);
#ifdef DEBUG_COUNTERS
      ht->not_found2++;
#endif
      return 1;
      }
      
   __m256i refs1 = _mm256_loadu_si256((__m256i *)&chain_ref[4]);
   __m256i refs2 = _mm256_loadu_si256((__m256i *)&chain_ref[8]);
   int refs1size = __builtin_popcount(res & 0xF);
   refs1 = _mm256_permutevar8x32_epi32(refs1,_mm256_loadu_si256((__m256i *)&lut12_1[res & 0xF][0]));
   refs2 = _mm256_permutevar8x32_epi32(refs2,_mm256_loadu_si256((__m256i *)&lut8[res >> 4][0]));

   __m256i links1 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(refs1));
   __m256i links21 = _mm256_unpacklo_epi32(refs2,zeropad);
   __m256i links22 = _mm256_unpackhi_epi32(refs2,zeropad);
   links1 = _mm256_sllv_epi32(links1,shifts);
   links21 = _mm256_sllv_epi32(links21,shifts);
   links22 = _mm256_sllv_epi32(links22,shifts);

   links1 = _mm256_add_epi64(links1,base);
   links21 = _mm256_add_epi64(links21,base);
   links22 = _mm256_add_epi64(links22,base);

   _mm256_storeu_si256((__m256i *)&state->data_refs[0],links1);
   _mm256_storeu_si256((__m256i *)&state->data_refs[refs1size],links21);
   _mm256_storeu_si256((__m256i *)&state->data_refs[refs1size+4],links22);
   ht->tick += 12;

   state->chain_ref = nidx ? &ht->table[nidx * CACHE_LINE_SIZE] : NULL;
   state->data_ref = &state->data_refs[bcnt - 1];
   LOG_RECORD("%d found in table, %d links, chain_ref %s",state->num,bcnt,state->chain_ref ? "present" : "absent");
   add_unpref(ht,state);
#ifdef DEBUG_COUNTERS
   if (state->data_ref == &state->chain_ref) ht->chain++;
#endif
   return 0;
   }

void ht_add_8(FHashTable *ht,FProcessState *state)
   {
   int isize = state->value_size / 4 + ((state->value_size % 4) ? 1 : 0);
   uint32_t dpos = ht->data_pos;
   ht->data_pos += isize + 1;
   ht->data[dpos] = isize;
   memcpy(&ht->data[dpos + 1],state->value_start,isize * 4);

   uint32_t *chain_ref = (uint32_t *)state->last_chain_ref;
   __m256i headers,cmpres0;

ht_add_8_repeat:
   headers = _mm256_loadu_si256((__m256i *)chain_ref);
   cmpres0 = _mm256_cmpeq_epi32(headers,_mm256_setzero_si256());
   int res0 = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres0)) & 0xFF;
   int pos = __builtin_ctzl(res0);

   if (pos == 7)
      {
      if (chain_ref[15])
         {
         chain_ref = (uint32_t *)&ht->table[chain_ref[15] * CACHE_LINE_SIZE];
         goto ht_add_8_repeat;
         }
      LOG_RECORD("%d state %d %s add in table in new line",ht->tick,state->num,state->key_buf);
      chain_ref[15] = ht->unlocated;
      chain_ref = (uint32_t *)&ht->table[ht->unlocated * CACHE_LINE_SIZE];
      ht->unlocated++;
      pos = 0;
      }
   else
      {
      LOG_RECORD("%d state %d %s add in table in pos %d",ht->tick,state->num,state->key_buf,pos);
      }
   chain_ref[pos] = *((int *)state->key_buf);
   chain_ref[pos+8] = dpos;
   ht->tick += 10;
   }

void ht_add_12(FHashTable *ht,FProcessState *state)
   {
   int isize = state->value_size / 4 + ((state->value_size % 4) ? 1 : 0);
   uint32_t dpos = ht->data_pos;
   ht->data_pos += isize + 1;
   ht->data[dpos] = isize;
   memcpy(&ht->data[dpos + 1],state->value_start,isize * 4);

   uint32_t *chain_ref = (uint32_t *)state->last_chain_ref;
   uint32_t bits;
   __m256i headers,cmpres0;

ht_add_12_repeat:
   bits = chain_ref[3];
   headers = _mm256_loadu_si256((__m256i *)chain_ref);
   cmpres0 = _mm256_cmpeq_epi8(headers,_mm256_setzero_si256());
   int res0 = _mm256_movemask_epi8(cmpres0);
   res0 &= ~(bits | (bits >> 16));
   res0 &= 0xFFF;
   int pos = __builtin_ctzl(res0);

   if (pos == 11)
      {
      if (chain_ref[15])
         {
         chain_ref = (uint32_t *)&ht->table[chain_ref[15] * CACHE_LINE_SIZE];
         goto ht_add_12_repeat;
         }
      LOG_RECORD("%d state %d %s add in table in new line",ht->tick,state->num,state->key_buf,state->num);
      chain_ref[15] = ht->unlocated;
      chain_ref = (uint32_t *)&ht->table[ht->unlocated * CACHE_LINE_SIZE];
      ht->unlocated++;
      pos = 0;
      }
   else
      {
      LOG_RECORD("%d state %d %s add in table in pos %d",ht->tick,state->num,state->key_buf,pos);
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
   __m256i shifts = _mm256_set1_epi32(2);
   uint32_t *chain_ref = (uint32_t *)state->chain_ref;

   __m256i headers = _mm256_loadu_si256((__m256i *)state->chain_ref);
   process_unpref(ht);
   __m256i zeropad = _mm256_setzero_si256();
   __m256i base = _mm256_set_epi64x((uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data);
   
   __m256i search = _mm256_set1_epi32(*((int *)state->key_buf));
   __m256i cmpres = _mm256_cmpeq_epi32(headers,search);
   int res = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres)) & 0x7F;
   int bcnt = __builtin_popcount(res); 

   state->last_chain_ref = state->chain_ref;

   uint32_t nidx = chain_ref[15];
   ht->tick += 25;

   if (bcnt + nidx)
      { // Unpredictable branch
      __m256i refs = _mm256_loadu_si256((__m256i *)&chain_ref[8]);
      refs = _mm256_permutevar8x32_epi32(refs,_mm256_loadu_si256((__m256i *)&lut8[res][0]));

      __m256i links1 = _mm256_unpacklo_epi32(refs,zeropad);
      __m256i links2 = _mm256_unpackhi_epi32(refs,zeropad);
      links1 = _mm256_sllv_epi32(links1,shifts);
      links2 = _mm256_sllv_epi32(links2,shifts);

      links1 = _mm256_add_epi64(links1,base);
      links2 = _mm256_add_epi64(links2,base);

      _mm256_storeu_si256((__m256i *)&state->data_refs[0],links1);
      _mm256_storeu_si256((__m256i *)&state->data_refs[4],links2);

      state->chain_ref = nidx ? &ht->table[nidx * CACHE_LINE_SIZE] : NULL;
      state->data_ref = &state->data_refs[bcnt - 1];

      add_unpref(ht,state);
      LOG_RECORD("%d state %d found in table, %d links, chain_ref %s",ht->tick,state->num,bcnt,state->chain_ref ? "present" : "absent");
      return 0;
      }
   LOG_RECORD("%d state %d not found in table",ht->tick,state->num);
   ht_add_8(ht,state);
   add_state(ht->empty,state);
   return 1;
   }

static inline int look_ht_12_add(FHashTable *ht,FProcessState *state)
   {
   __m256i shifts = _mm256_set1_epi32(2);
   uint32_t *chain_ref = (uint32_t *)state->chain_ref;
   __m256i headers = _mm256_loadu_si256((__m256i *)state->chain_ref);
   process_unpref(ht);
   __m256i zeropad = _mm256_setzero_si256();
   __m256i base = _mm256_set_epi64x((uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data);
   
   __m256i search = _mm256_set1_epi8(state->key_buf[0]);
   uint32_t bits = chain_ref[3];
   uint32_t bit8 = 0x7FF * (state->key_buf[1] & 0x1);
   uint32_t bit9 = 0x7FF * ((state->key_buf[1] & 0x2) >> 1);
   __m256i cmpres = _mm256_cmpeq_epi8(headers,search);

   int res = _mm256_movemask_epi8(cmpres);
   res &= ~(bits ^ bit8) & (~(bits >> 16) ^ bit9) & 0x7FF;
   int bcnt = __builtin_popcount(res);
    
   state->last_chain_ref = state->chain_ref;

   uint32_t nidx = chain_ref[15];
   ht->tick += 25;

   if (bcnt + nidx)
      { // Unpredictable branch, no prefetch here
      int refs1size = __builtin_popcount(res & 0xF);
      __m256i refs1 = _mm256_loadu_si256((__m256i *)&chain_ref[4]);
      __m256i refs2 = _mm256_loadu_si256((__m256i *)&chain_ref[8]);

      refs1 = _mm256_permutevar8x32_epi32(refs1,_mm256_loadu_si256((__m256i *)&lut12_1[res & 0xF][0]));
      refs2 = _mm256_permutevar8x32_epi32(refs2,_mm256_loadu_si256((__m256i *)&lut8[res >> 4][0]));

      __m256i links1 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(refs1));
      __m256i links21 = _mm256_unpacklo_epi32(refs2,zeropad);
      __m256i links22 = _mm256_unpackhi_epi32(refs2,zeropad);
      links1 = _mm256_sllv_epi32(links1,shifts);
      links21 = _mm256_sllv_epi32(links21,shifts);
      links22 = _mm256_sllv_epi32(links22,shifts);

      links1 = _mm256_add_epi64(links1,base);
      links21 = _mm256_add_epi64(links21,base);
      links22 = _mm256_add_epi64(links22,base);

      _mm256_storeu_si256((__m256i *)&state->data_refs[0],links1);
      _mm256_storeu_si256((__m256i *)&state->data_refs[refs1size],links21);
      _mm256_storeu_si256((__m256i *)&state->data_refs[refs1size+4],links22);

      state->chain_ref = nidx ? &ht->table[nidx * CACHE_LINE_SIZE] : NULL;
      state->data_ref = &state->data_refs[bcnt - 1];

      add_unpref(ht,state);
      LOG_RECORD("%d found in table, %d links, chain_ref %s",state->num,bcnt,state->chain_ref ? "present" : "absent");
      return 0;
      }

   LOG_RECORD("%d state %d not found in table",ht->tick,state->num);
   ht_add_12(ht,state);
   add_state(ht->empty,state);
   return 1;
   }

// 4 by step
int look_in_data_4_states(FHashTable *ht,FOutput *output)
   {
   uint64_t res_mask,key_mask;
   int rv = 0;

   FProcessState *state0 = get_state(ht->d_req);
   __m256i data00 = _mm256_loadu_si256((__m256i *)(*state0->data_ref + 4));
   __m256i data01 = _mm256_loadu_si256((__m256i *)(*state0->data_ref + 36));
   process_unpref(ht);
   __m256i key00 = _mm256_loadu_si256((__m256i *)state0->key_buf); 
   __m256i key01 = _mm256_loadu_si256((__m256i *)&state0->key_buf[32]); 

   FProcessState *state1 = get_state(ht->d_req);
   __m256i data10 = _mm256_loadu_si256((__m256i *)(*state1->data_ref + 4));
   __m256i data11 = _mm256_loadu_si256((__m256i *)(*state1->data_ref + 36));
   process_unpref(ht);
   __m256i key10 = _mm256_loadu_si256((__m256i *)state1->key_buf); 
   __m256i key11 = _mm256_loadu_si256((__m256i *)&state1->key_buf[32]); 

   key_mask = (1LL << state0->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key01,data01)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key00,data00));
   if ((res_mask & key_mask) == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)output->pos,key00), _mm256_storeu_si256((__m256i *)(output->pos+32),key01);
      output->pos[state0->key_size] = '\n';
      output->pos += state0->key_size + 1;
      rv++;
      LOG_RECORD("%d state %d found in table",ht->tick,state0->num);
      add_state(ht->empty,state0);
#ifdef DEBUG_COUNTERS
      ht->found++;
#endif
      }
   else if (*(--state0->data_ref))
      {
      add_unpref(ht,state0);
#ifdef DEBUG_COUNTERS
      if (state0->data_ref == &state0->chain_ref) ht->chain++;
      else ht->fp++;
#endif
      }
   else
      {
      rv++;
      LOG_RECORD("%d state %d false positive, not found in table",ht->tick,state0->num);
      add_state(ht->empty,state0);
#ifdef DEBUG_COUNTERS
      ht->not_found++;
      ht->fp++;
#endif
      }

   FProcessState *state2 = get_state(ht->d_req);
   __m256i data20 = _mm256_loadu_si256((__m256i *)(*state2->data_ref + 4));
   __m256i data21 = _mm256_loadu_si256((__m256i *)(*state2->data_ref + 36));
   process_unpref(ht);
   __m256i key20 = _mm256_loadu_si256((__m256i *)state2->key_buf); 
   __m256i key21 = _mm256_loadu_si256((__m256i *)&state2->key_buf[32]);  

   key_mask = (1LL << state1->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key11,data11)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key10,data10));
   if ((res_mask & key_mask) == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)output->pos,key10), _mm256_storeu_si256((__m256i *)(output->pos+32),key11);
      output->pos[state1->key_size] = '\n';
      output->pos += state1->key_size + 1;
      rv++;
      LOG_RECORD("%d state %d found in table",ht->tick,state1->num);
      add_state(ht->empty,state1);
#ifdef DEBUG_COUNTERS
      ht->found++;
#endif
      }
   else if (*(--state1->data_ref))
      {
      add_unpref(ht,state1);
#ifdef DEBUG_COUNTERS
      if (state1->data_ref == &state1->chain_ref) ht->chain++;
      else ht->fp++;
#endif
      }
   else
      {
      rv++;
      LOG_RECORD("%d state %d false positive, not found in table",ht->tick,state1->num);
      add_state(ht->empty,state1);
#ifdef DEBUG_COUNTERS
      ht->not_found++;
      ht->fp++;
#endif
      }

   FProcessState *state3 = get_state(ht->d_req);
   __m256i data30 = _mm256_loadu_si256((__m256i *)(*state3->data_ref + 4));
   __m256i data31 = _mm256_loadu_si256((__m256i *)(*state3->data_ref + 36));
   process_unpref(ht);
   __m256i key30 = _mm256_loadu_si256((__m256i *)state3->key_buf); 
   __m256i key31 = _mm256_loadu_si256((__m256i *)&state3->key_buf[32]); 

   key_mask = (1LL << state2->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key21,data21)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key20,data20));
   if ((res_mask & key_mask) == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)output->pos,key20), _mm256_storeu_si256((__m256i *)(output->pos+32),key21);
      output->pos[state2->key_size] = '\n';
      output->pos += state2->key_size + 1;
      rv++;
      LOG_RECORD("%d state %d found in table",ht->tick,state2->num);
      add_state(ht->empty,state2);
#ifdef DEBUG_COUNTERS
      ht->found++;
#endif
      }
   else if (*(--state2->data_ref))
      {
      add_unpref(ht,state2);
#ifdef DEBUG_COUNTERS
      if (state2->data_ref == &state2->chain_ref) ht->chain++;
      else ht->fp++;
#endif
      }
   else
      {
      rv++;
      LOG_RECORD("%d state %d false positive, not found in table",ht->tick,state2->num);
      add_state(ht->empty,state2);
#ifdef DEBUG_COUNTERS
      ht->not_found++;
      ht->fp++;
#endif
      }

   key_mask = (1LL << state3->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key31,data31)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key30,data30));
   if ((res_mask & key_mask) == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)output->pos,key30), _mm256_storeu_si256((__m256i *)(output->pos+32),key31);
      output->pos[state3->key_size] = '\n';
      output->pos += state3->key_size + 1;
      rv++;
      LOG_RECORD("%d state %d found in table",ht->tick,state3->num);
      add_state(ht->empty,state3);
#ifdef DEBUG_COUNTERS
      ht->found++;
#endif
      }
   else if (*(--state3->data_ref))
      {
      add_unpref(ht,state3);
#ifdef DEBUG_COUNTERS
      if (state3->data_ref == &state3->chain_ref) ht->chain++;
      else ht->fp++;
#endif
      }
   else
      {
      rv++;
      LOG_RECORD("%d state %d false positive, not found in table",ht->tick,state3->num);
      add_state(ht->empty,state3);
#ifdef DEBUG_COUNTERS
      ht->not_found++;
      ht->fp++;
#endif
      }
   ht->tick += 38;
   return rv;
   }

int look_in_data_add(FHashTable *ht,FProcessState *state)
   {
   uint64_t res_mask,key_mask;

   __m256i data0 = _mm256_loadu_si256((__m256i *)(*state->data_ref + 4));
   __m256i data1 = _mm256_loadu_si256((__m256i *)(*state->data_ref + 36));
   process_unpref(ht);
   ht->tick += 15;
   __m256i key0 = _mm256_loadu_si256((__m256i *)state->key_buf); 
   __m256i key1 = _mm256_loadu_si256((__m256i *)&state->key_buf[32]); 

   key_mask = (1LL << state->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key1,data1)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key0,data0));
   if ((res_mask & key_mask) == key_mask)
      { // Unlikely
      LOG_RECORD("%d state %d found in table, not added",ht->tick,state->num);
      return 1;
      }
   if (*(--state->data_ref))
      {
      add_unpref(ht,state);
      return 0;
      }
   LOG_RECORD("%d state %d not found in table and has no links",ht->tick,state->num);
   return 2;
   }

void reset_state(FProcessState *state)
   {
   state->chain_ref = NULL;
   state->data_ref = &state->chain_ref;
   state->col_num = 0;
   state->key_size = 0;
   state->key_pos = state->key_buf;
   state->value_pos = state->value_start;
   }

int64_t get_nanotime(void)
   {
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC,&t);
   return t.tv_sec * 1000000000 + t.tv_nsec;
   }

void reset_ht_search(FHashTable *ht,FProcessState *states)
   {
   int i;
   reset_set(ht->d_req);
   reset_set(ht->h_req);
   reset_set(ht->empty);
   reset_set(ht->loaded);
   reset_set(ht->unpref);
   ht->pcnt = 0;
   ht->tick = 0;
   for (i = 0; i < BIG_SET_SIZE; i++)
      add_state(ht->empty,&states[i]);
#ifdef DEBUG_COUNTERS
   ht->found = ht->not_found = ht->not_found2 = ht->chain = ht->fp = 0;
#endif
   }

void reset_ht_data(FHashTable *ht)
   {
   ht->unlocated = TABLE_SIZE;
   memset(ht->table,0,ht->table_size * 2 * CACHE_LINE_SIZE);
   memset(ht->data,0,ITEMS_COUNT * 128);
   ht->data_pos = 1;
   }

double NANOTIME_COST = 0;

void process_add_8(FHashTable *ht,FSource *source, FProcessState *states)
   {
   FProcessState *state;
   int i,j;

   FProcessStateSmallSet set_by_size[8] = {0};
   reset_ht_data(ht);
   reset_ht_search(ht,states);

   while (source->inputpos < source->inputlen)
      {
      while (ht->empty->count && source->inputpos < source->inputlen)
         {
         state = get_state(ht->empty);
         reset_state(state);
         process_row(source,state);

         int ks = state->key_size / 8;
         set_by_size[ks].offsets[set_by_size[ks].count] = state->offset;
         set_by_size[ks].states[set_by_size[ks].count++] = state;

         if (set_by_size[ks].count == 16)
            {
            make_hashes16(ht,(int *)states,&set_by_size[ks],ks+1);
            for(j = 0; j < 16; j++)
               add_unpref(ht,set_by_size[ks].states[j]);
            set_by_size[ks].count = 0;
            }
         if (source->inputpos >= source->inputlen) break;
         }
      max_process_unpref(ht);
      while(ht->h_req->count && get_state_link(ht->h_req,0)->tick <= ht->tick)
         look_ht_8_add(ht,get_state(ht->h_req));

      while(ht->d_req->count && get_state_link(ht->d_req,0)->tick <= ht->tick)
         {
         state = get_state(ht->d_req);
         switch (look_in_data_add(ht,state))
            {
            case 2: ht_add_8(ht,state);
            case 1: add_state(ht->empty,state);
            }
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
   while (ht->unpref->count || ht->d_req->count || ht->h_req->count)
      {
      max_process_unpref(ht);
      while(ht->h_req->count && get_state_link(ht->h_req,0)->tick <= ht->tick)
         look_ht_8_add(ht,get_state(ht->h_req));

      while(ht->d_req->count && get_state_link(ht->d_req,0)->tick <= ht->tick)
         {
         state = get_state(ht->d_req);
         switch (look_in_data_add(ht,state))
            {
            case 2: ht_add_8(ht,state);
            case 1: add_state(ht->empty,state);
            }
         }
      ht->tick += 20;
      }
   }

void process_add_12(FHashTable *ht,FSource *source, FProcessState *states)
   {
   FProcessState *state;
   int i,j;
   FProcessStateSmallSet set_by_size[8] = {0};

   reset_ht_data(ht);
   reset_ht_search(ht,states);

   while (source->inputpos < source->inputlen)
      {
      while (ht->empty->count && source->inputpos < source->inputlen)
         {
         state = get_state(ht->empty);
         reset_state(state);
         process_row(source,state);

         int ks = state->key_size / 8;
         set_by_size[ks].offsets[set_by_size[ks].count] = state->offset;
         set_by_size[ks].states[set_by_size[ks].count++] = state;

         if (set_by_size[ks].count == 16)
            {
            make_hashes16(ht,(int *)states,&set_by_size[ks],ks+1);
            for(j = 0; j < 16; j++)
               add_unpref(ht,set_by_size[ks].states[j]);
            set_by_size[ks].count = 0;
            }
         if (source->inputpos >= source->inputlen) break;
         }
      max_process_unpref(ht);
      while(ht->h_req->count && get_state_link(ht->h_req,0)->tick <= ht->tick)
         look_ht_12_add(ht,get_state(ht->h_req));
      while(ht->d_req->count && get_state_link(ht->d_req,0)->tick <= ht->tick)
         {
         state = get_state(ht->d_req);
         switch (look_in_data_add(ht,state))
            {
            case 2: ht_add_12(ht,state);
            case 1: add_state(ht->empty,state);
            }
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
   while (ht->unpref->count || ht->d_req->count || ht->h_req->count)
      {
      max_process_unpref(ht);
      while(ht->h_req->count && get_state_link(ht->h_req,0)->tick <= ht->tick)
         look_ht_12_add(ht,get_state(ht->h_req));

      while(ht->d_req->count && get_state_link(ht->d_req,0)->tick <= ht->tick)
         {
         state = get_state(ht->d_req);
         switch (look_in_data_add(ht,state))
            {
            case 2: ht_add_12(ht,state);
            case 1: add_state(ht->empty,state);
            }
         }
      ht->tick += 20;
      }
   }

double process_search_8(FHashTable *ht,FSource *source, FProcessState *states,FOutput *output)
   {
   FProcessState *state;
   int64_t tm = 0;
   int j;
   FProcessStateSmallSet set_by_size[8] = {0};
   int processed = 0;

   reset_ht_search(ht,states);

   while (source->inputpos < source->inputlen)
      {
      while (ht->empty->count && source->inputpos < source->inputlen)
         {
         state = get_state(ht->empty);
         reset_state(state);
         process_row(source,state);
         add_state(ht->loaded,state);
         ht->tick += 200;
         }
      int64_t t1 = get_nanotime();
      while (ht->loaded->count)
         {
         state = get_state(ht->loaded);
         int ks = state->key_size / 8;
         set_by_size[ks].offsets[set_by_size[ks].count] = state->offset;
         set_by_size[ks].states[set_by_size[ks].count++] = state;

         if (set_by_size[ks].count == 16)
            {
            make_hashes16(ht,(int *)states,&set_by_size[ks],ks+1);
            for(j = 0; j < 16; j++)
               {
               LOG_RECORD("%d added to unpref",set_by_size[ks].states[j]->num);
               add_state(ht->unpref,set_by_size[ks].states[j]);
               }
            set_by_size[ks].count = 0;
            }

         max_process_unpref(ht);
         while(ht->h_req->count && get_state_link(ht->h_req,0)->tick <= ht->tick)
            processed += look_ht_8(ht,get_state(ht->h_req));
         while(ht->d_req->count >= 4 && get_state_link(ht->d_req,3)->tick <= ht->tick)
            processed += look_in_data_4_states(ht,output);
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
   FProcessStateSmallSet set_by_size[8] = {0};
   int processed = 0;

   reset_ht_search(ht,states);

   while (source->inputpos < source->inputlen)
      {
      while (ht->empty->count && source->inputpos < source->inputlen)
         {
         state = get_state(ht->empty);
         reset_state(state);
         process_row(source,state);
         add_state(ht->loaded,state);
         ht->tick += 200;
         }

      int64_t t1 = get_nanotime();
      while (ht->loaded->count)
         {
         state = get_state(ht->loaded);
         int ks = state->key_size / 8;
         set_by_size[ks].offsets[set_by_size[ks].count] = state->offset;
         set_by_size[ks].states[set_by_size[ks].count++] = state;

         if (set_by_size[ks].count == 16)
            {
            make_hashes16(ht,(int *)states,&set_by_size[ks],ks+1);
            for(j = 0; j < 16; j++)
               add_unpref(ht,set_by_size[ks].states[j]);
            set_by_size[ks].count = 0;
            }

         max_process_unpref(ht);
         while(ht->h_req->count && get_state_link(ht->h_req,0)->tick <= ht->tick)
            processed += look_ht_12(ht,get_state(ht->h_req));
         while(ht->d_req->count >= 4 && get_state_link(ht->d_req,3)->tick <= ht->tick)
            processed += look_in_data_4_states(ht,output);
         ht->tick += 20;
         }
      tm += get_nanotime() - t1;
      }

   return (double)tm/processed;
   }

int main(int argc, char *argv[ ])
   {
   int i;
   FHashTable ht;
   ht.table_size = ht.unlocated = TABLE_SIZE;
   ht.table = aligned_alloc(CACHE_LINE_SIZE,ht.table_size * 2 * CACHE_LINE_SIZE);
   ht.data = aligned_alloc(CACHE_LINE_SIZE,ITEMS_COUNT * 128);

   FProcessStateBigSet unpref={0},lh={0}, ld={0}, empty={0}, loaded = {0};
   ht.h_req = &lh;
   ht.d_req = &ld;
   ht.unpref = &unpref;
   ht.empty = &empty;
   ht.loaded = &loaded;

   if (argc < 3)
      return printf("Format: ./hashtable.out file1 file2\n"),1;

   INIT_LOG("ht.log");

   off_t sz1 = file_size(argv[1]);
   if (sz1 < 0)
      return printf("file %s not found\n",argv[1]),1;

   off_t sz2 = file_size(argv[2]);
   if (sz2 < 0)
      return printf("file %s not found\n",argv[2]),1;

   char *filedata = read_file(argv[1],sz1);
   char *filedata2 = read_file(argv[2],sz2);

   make_lut8();
   make_lut12();

   FParseParams parse_params = {'\t','\t',0,3,__builtin_popcount(3)};

   FProcessState *states = aligned_alloc(CACHE_LINE_SIZE,sizeof(FProcessState) * STATES_COUNT);
   for (i = 0; i < STATES_COUNT; i++)
      states[i].pp = &parse_params, states[i].num = i, states[i].offset = i * sizeof(FProcessState);

   FOutput output;
   output.start = output.pos = (char *)malloc(sz1 + 32);
   FSource source,source2;

// 8 per line

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

// 12 per line

   init_source(&source,filedata,sz1);
   process_add_12(&ht,&source,states);

   double tm12_1 = 10000000.0;
   for (i = 0; i < ITER_COUNT; i++)
      {
      init_source(&source,filedata,sz1);
      output.pos = output.start;
      double tm = process_search_12(&ht,&source,states,&output);
      if (tm < tm12_1)
         tm12_1 = tm;
      }

#ifdef DEBUG_COUNTERS
   printf("present 12: found %d, not found %d, long %d, false positive %d\n",ht.found,ht.not_found + ht.not_found2,ht.chain,ht.fp);
#endif

   double tm12_2 = 10000000.0;
   for (i = 0; i < ITER_COUNT; i++)
      {
      init_source(&source2,filedata2,sz2);
      output.pos = output.start;
      double tm = process_search_12(&ht,&source2,states,&output);
      if (tm < tm12_2)
         tm12_2 = tm;
      }

#ifdef DEBUG_COUNTERS
   printf("absent  12: found %d, not found %d, long %d, false positive %d\n",ht.found,ht.not_found + ht.not_found2,ht.chain,ht.fp);
#endif

   free(ht.table);
   free(ht.data);
   free(filedata);
   free(filedata2);
   free(states);
   free(output.start);
   printf("present  8: %.3f\n",tm8_1);
   printf("absent   8: %.3f\n",tm8_2);
   printf("present 12: %.3f\n",tm12_1);
   printf("absent  12: %.3f\n",tm12_2);
   printf("present ratio: %.3f\n",tm8_1/tm12_1);
   printf("absent ratio: %.3f\n",tm8_2/tm12_2);
   CLOSE_LOG;
   return 0;
   }
