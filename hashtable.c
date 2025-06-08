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
#include "utils.h"

#include "hashtable.h"
#include "csvparse.h"

#define HASH_PRIME 591798841
#define HASH_INITIAL 2166136261

static inline void make_hashes(FHashTable *ht,int *base,FProcessStateSmallSet *reqs,int size)
   {
   int iindexes[8] = {  ((int *)reqs->states[0]) - base, ((int *)reqs->states[1]) - base, ((int *)reqs->states[2]) - base, ((int *)reqs->states[3]) - base, 
                        ((int *)reqs->states[4]) - base, ((int *)reqs->states[5]) - base, ((int *)reqs->states[6]) - base, ((int *)reqs->states[7]) - base };
   __m256i indexes = _mm256_loadu_si256((__m256i *)iindexes);
   __m256i a = _mm256_i32gather_epi32(base,indexes,4);
   __m256i incs = _mm256_set1_epi32(1);
   indexes = _mm256_add_epi32(indexes,incs);
   size = (size - 1) * 2;
   __m256i b = _mm256_i32gather_epi32(base,indexes,4);
   indexes = _mm256_add_epi32(indexes,incs);

   __m256i hashes1,hashes2;
   __m256i leftshifts = _mm256_set1_epi32(5);
   __m256i rightshifts = _mm256_set1_epi32(27);
   __m256i primes = _mm256_set1_epi32(HASH_PRIME);
   __m256i hashes = _mm256_set1_epi32(HASH_INITIAL);
   hashes = _mm256_xor_si256(hashes,a);
   hashes = _mm256_mul_epi32(hashes,primes);
   while (size--)
      {
      a = b;
      b = _mm256_i32gather_epi32(base,indexes,4);
      indexes = _mm256_add_epi32(indexes,incs);

      hashes1 = _mm256_sllv_epi32(hashes,leftshifts);
      hashes2 = _mm256_srlv_epi32(hashes,rightshifts);
      hashes = _mm256_or_si256(hashes1,hashes2);
      hashes = _mm256_xor_si256(hashes,a);
      hashes = _mm256_mul_epi32(hashes,primes);
      }
   hashes1 = _mm256_sllv_epi32(hashes,leftshifts);
   hashes2 = _mm256_srlv_epi32(hashes,rightshifts);
   hashes = _mm256_or_si256(hashes1,hashes2);
   hashes = _mm256_xor_si256(hashes,b);
   hashes = _mm256_mul_epi32(hashes,primes);

   uint32_t i,res[8];
   _mm256_storeu_si256((__m256i *)res,hashes);
   for(i = 0; i < 8; i++)
      {
      reqs->states[i]->data_ref[0] = reqs->states[i]->chain_ref = &ht->table[(res[i] % ht->table_size) * CACHE_LINE_SIZE];
      LOG_RECORD("%d hash calculation",reqs->states[i]->num);
      }
   }

static inline void make_hashes16(FHashTable *ht,int *base,FProcessStateSmallSet *reqs,int ssize)
   {
   int iindexes1[8] = {  ((int *)reqs->states[0]) - base, ((int *)reqs->states[1]) - base, ((int *)reqs->states[2]) - base, ((int *)reqs->states[3]) - base, 
                        ((int *)reqs->states[4]) - base, ((int *)reqs->states[5]) - base, ((int *)reqs->states[6]) - base, ((int *)reqs->states[7]) - base };
   int iindexes2[8] = {  ((int *)reqs->states[8]) - base, ((int *)reqs->states[9]) - base, ((int *)reqs->states[10]) - base, ((int *)reqs->states[11]) - base, 
                        ((int *)reqs->states[12]) - base, ((int *)reqs->states[13]) - base, ((int *)reqs->states[14]) - base, ((int *)reqs->states[15]) - base };
   __m256i indexes1 = _mm256_loadu_si256((__m256i *)iindexes1);
   __m256i indexes2 = _mm256_loadu_si256((__m256i *)iindexes2);
   __m256i a1 = _mm256_i32gather_epi32(base,indexes1,4);
   __m256i a2 = _mm256_i32gather_epi32(base,indexes2,4);
   __m256i incs = _mm256_set1_epi32(1);
   indexes1 = _mm256_add_epi32(indexes1,incs);
   indexes2 = _mm256_add_epi32(indexes2,incs);
   int size = (ssize - 1) * 2;
   __m256i b1 = _mm256_i32gather_epi32(base,indexes1,4);
   __m256i b2 = _mm256_i32gather_epi32(base,indexes2,4);
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
   hashes1 = _mm256_mul_epi32(hashes1,primes);
   hashes2 = _mm256_mul_epi32(hashes2,primes);
   while (size--)
      {
      a1 = b1;
      a2 = b2;
      b1 = _mm256_i32gather_epi32(base,indexes1,4);
      b2 = _mm256_i32gather_epi32(base,indexes2,4);
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
      hashes1 = _mm256_mul_epi32(hashes1,primes);
      hashes2 = _mm256_mul_epi32(hashes2,primes);
      }
   hashes11 = _mm256_sllv_epi32(hashes1,leftshifts);
   hashes21 = _mm256_sllv_epi32(hashes2,leftshifts);
   hashes12 = _mm256_srlv_epi32(hashes1,rightshifts);
   hashes22 = _mm256_srlv_epi32(hashes2,rightshifts);
   hashes1 = _mm256_or_si256(hashes11,hashes12);
   hashes2 = _mm256_or_si256(hashes21,hashes22);
   hashes1 = _mm256_xor_si256(hashes1,b1);
   hashes2 = _mm256_xor_si256(hashes2,b2);
   hashes1 = _mm256_mul_epi32(hashes1,primes);
   hashes2 = _mm256_mul_epi32(hashes2,primes);

   uint32_t i,res[16];
   _mm256_storeu_si256((__m256i *)&res[0],hashes1);
   _mm256_storeu_si256((__m256i *)&res[8],hashes2);
   for(i = 0; i < 16; i++)
      {
      reqs->states[i]->data_ref[0] = reqs->states[i]->chain_ref = &ht->table[(res[i] % ht->table_size) * CACHE_LINE_SIZE];
      LOG_RECORD("%d hash calculation",reqs->states[i]->num);
      }
   ht->tick += 40 + 10 * size;
   }

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
   set->count++;
   set->states[set->last] = state;
   set->last = (set->last + 1) % BIG_SET_SIZE;
   }

static inline void set_inc_by(FProcessStateBigSet *set,int count)
   {
   set->count += count;
   set->last = (set->last + count) % BIG_SET_SIZE;
   }

static inline void set_dec_by(FProcessStateBigSet *set,int count)
   {
   set->count += count;
   set->first = (set->first + count) % BIG_SET_SIZE;
   }

int lut8[128][8] = {0};

void make_lut8(void)
   {
   int i;
   for (i = 0; i < 128; i++)
      {
      int v = i, bn, pos = 0;
      while ((bn = __builtin_ffsl(v)))
         {
         lut8[i][pos++] = bn - 1;
         v >>= bn;
         }
      }
   }

// 8 keys in line
static inline int look_ht_8(FHashTable *ht,FProcessState *state,FProcessStateSmallSet *set_data)
   {
   __m256i shifts = _mm256_set1_epi32(2);
   uint32_t *block_ref = (uint32_t *)state->chain_ref;
   __m256i headers = _mm256_loadu_si256((__m256i *)state->chain_ref);
   __m256i zeropad = _mm256_setzero_si256();
   __m256i base = _mm256_set_epi64x((uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data);
   
   __m256i search = _mm256_set1_epi32(*((int *)state->key_buf));
   __m256i cmpres = _mm256_cmpeq_epi32(headers,search);
   int res = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres));
   int bcnt = __builtin_popcount(res); 
   uint32_t nidx = block_ref[15];
   state->data_ref = &state->data_refs[0];
   ht->tick += 5;
   ht->pcnt--;

   if (bcnt + nidx)
      {
      ht->tick += 5;
      __m256i refs = _mm256_loadu_si256((__m256i *)&block_ref[8]);
      refs = _mm256_permutevar8x32_epi32(refs,_mm256_loadu_si256((__m256i *)&lut8[res][0]));

      __m256i links1 = _mm256_unpacklo_epi32(refs,zeropad);
      __m256i links2 = _mm256_unpackhi_epi32(refs,zeropad);
      links1 = _mm256_sllv_epi32(links1,shifts);
      links2 = _mm256_sllv_epi32(links2,shifts);

      links1 = _mm256_add_epi64(links1,base);
      links2 = _mm256_add_epi64(links2,base);

      _mm256_storeu_si256((__m256i *)&state->data_refs[0],links1);
      _mm256_storeu_si256((__m256i *)&state->data_refs[4],links2);
      state->chain_ref = state->data_refs[bcnt] = nidx ? &ht->table[nidx * CACHE_LINE_SIZE] : NULL;
      state->data_refs[bcnt + 1] = NULL;
      LOG_RECORD("%d found in table, %d links, chain_ref %s",state->num,bcnt,state->chain_ref ? "present" : "absent");
      set_data->states[set_data->count++] = state;
      return 0;
      }
   state->chain_ref = state->data_refs[0] = NULL;
   LOG_RECORD("%d not found in table",state->num);
   return 1;
   }

int lut12_1[16][8] = {0};
int lut12_2[128][8] = {0};

void make_lut12(void)
   {
   int i;
   for (i = 0; i < 16; i++)
      {
      int v = i, bn, pos = 0;
      while ((bn = __builtin_ffsl(v)))
         {
         lut12_1[i][pos++] = bn - 1;
         v >>= bn;
         }
      }
   for (i = 0; i < 128; i++)
      {
      int v = i, bn, pos = 0;
      while ((bn = __builtin_ffsl(v)))
         {
         lut12_2[i][pos++] = bn - 1;
         v >>= bn;
         }
      }
   }

static inline int look_ht_12(FHashTable *ht,FProcessState *state,FProcessStateSmallSet *set_data)
   {
   __m256i shifts = _mm256_set1_epi32(2);
   uint32_t *block_ref = (uint32_t *)state->chain_ref;
   __m256i headers = _mm256_loadu_si256((__m256i *)state->chain_ref);
   __m256i zeropad = _mm256_setzero_si256();
   __m256i base = _mm256_set_epi64x((uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data);
   
   __m256i search = _mm256_set1_epi8(state->key_buf[0]);
   uint32_t bits = block_ref[3];
   uint32_t bit8 = 0xEFF * (state->key_buf[1] & 0x1);
   uint32_t bit9 = 0xEFF * ((state->key_buf[1] & 0x2) >> 1);
   __m256i cmpres = _mm256_cmpeq_epi8(headers,search);

   int res = _mm256_movemask_epi8(cmpres);
   res &= ~(bits ^ bit8) & (~(bits >> 16) ^ bit9) & 0xEFF;

   int bcnt = __builtin_popcount(res); 
   uint32_t nidx = block_ref[15];
   state->data_ref = &state->data_refs[0];
   ht->tick += 7;
   ht->pcnt--;
   if (bcnt + nidx)
      {   // Unpredictable branch
      __m256i refs1 = _mm256_loadu_si256((__m256i *)&block_ref[4]);
      __m256i refs2 = _mm256_loadu_si256((__m256i *)&block_ref[8]);
      int refs1size = __builtin_popcount(res & 0xF);
      refs1 = _mm256_permutevar8x32_epi32(refs1,_mm256_loadu_si256((__m256i *)&lut12_1[res & 0xF][0]));
      refs2 = _mm256_permutevar8x32_epi32(refs2,_mm256_loadu_si256((__m256i *)&lut12_2[res >> 4][0]));

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

      state->chain_ref = state->data_refs[bcnt] = nidx ? &ht->table[nidx * CACHE_LINE_SIZE] : NULL;
      state->data_refs[bcnt + 1] = NULL;
      LOG_RECORD("%d found in table, %d links, chain_ref %s",state->num,bcnt,state->chain_ref ? "present" : "absent");
      set_data->states[set_data->count++] = state;
      return 0;
      }
   state->chain_ref = state->data_refs[0] = NULL;
   LOG_RECORD("%d not found in table",state->num);
   return 1;
   }

void ht_add_8(FHashTable *ht,FProcessState *state)
   {
   int isize = state->value_size / 4 + ((state->value_size % 4) ? 1 : 0);
   uint32_t dpos = ht->data_pos;
   ht->data_pos += isize;
   ht->data[dpos] = isize;
   memcpy(&ht->data[dpos + 1],state->value_start,isize * 4);

   uint32_t *chain_ref;
   int pos = state->last_pos;
   chain_ref = (uint32_t *)state->last_chain_ref;
   if (pos == 7)
      {
      LOG_RECORD("%d %s add in table in new line",state->num,state->key_buf);
      chain_ref[15] = ht->unlocated;
      chain_ref = (uint32_t *)&ht->table[ht->unlocated * CACHE_LINE_SIZE];
      ht->unlocated++;
      pos = 0;
      }
   else
      {
      LOG_RECORD("%d %s add in table in pos %d",state->num,state->key_buf,pos);
      }
   chain_ref[pos] = *((int *)state->key_buf);
   chain_ref[pos+8] = dpos;
   ht->tick += 10;
   }

void ht_add_12(FHashTable *ht,FProcessState *state)
   {
   int isize = state->value_size / 4 + ((state->value_size % 4) ? 1 : 0);
   uint32_t dpos = ht->data_pos;
   ht->data_pos += isize;
   ht->data[dpos] = isize;
   memcpy(&ht->data[dpos + 1],state->value_start,isize * 4);

   uint32_t *chain_ref;
   int pos = state->last_pos;
   chain_ref = (uint32_t *)state->last_chain_ref;
   if (pos == 11)
      {
      LOG_RECORD("%d %s add in table in new line",state->num,state->key_buf,state->num);
      chain_ref[11] = ht->unlocated;
      chain_ref = (uint32_t *)&ht->table[ht->unlocated * CACHE_LINE_SIZE];
      ht->unlocated++;
      pos = 0;
      }
   else
      {
      LOG_RECORD("%d %s add in table in pos %d",state->num,state->key_buf,pos);
      }
   chain_ref[3] &= ~(0x10001 << pos);
   chain_ref[3] |= (state->key_buf[1] & 0x1) << pos;
   chain_ref[3] |= (state->key_buf[1] & 0x2) << (pos + 16 - 1);
   state->last_chain_ref[pos] = state->key_buf[0];
   chain_ref[pos+4] = dpos;
   ht->tick += 10;
   }

int look_ht_8_add(FHashTable *ht,FProcessState *state,FProcessStateSmallSet *set_data,FProcessStateBigSet *empty)
   {
   uint32_t *block_ref = (uint32_t *)state->chain_ref;
   __m256i headers = _mm256_loadu_si256((__m256i *)state->chain_ref);
   __m256i zeropad = _mm256_setzero_si256();
   __m256i base = _mm256_set_epi64x((uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data);
   
   __m256i search = _mm256_set1_epi32(*((int *)state->key_buf));
   __m256i cmpres = _mm256_cmpeq_epi32(headers,search);
   __m256i refs = _mm256_loadu_si256((__m256i *)&block_ref[8]);
   int res = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres));
   refs = _mm256_permutevar8x32_epi32(refs,_mm256_loadu_si256((__m256i *)&lut8[res][0]));

   __m256i links1 = _mm256_unpacklo_epi32(refs,zeropad);
   __m256i links2 = _mm256_unpackhi_epi32(refs,zeropad);

   links1 = _mm256_add_epi64(links1,base);
   links2 = _mm256_add_epi64(links2,base);

   _mm256_storeu_si256((__m256i *)&state->data_refs[0],links1);
   _mm256_storeu_si256((__m256i *)&state->data_refs[4],links2);
   int bcnt = __builtin_popcount(res); // 1 for chain link
   uint32_t nidx = block_ref[15];
   state->last_chain_ref = state->chain_ref;
   state->chain_ref = state->data_refs[bcnt] = nidx ? &ht->table[nidx * CACHE_LINE_SIZE] : NULL;
   state->data_refs[bcnt + 1] = NULL;
   ht->tick += 25;
   ht->pcnt--;
   if (state->data_refs[0])
      { // Unpredictable branch, no prefetch here
      state->data_ref = &state->data_refs[0];
      set_data->states[set_data->count++] = state;
      LOG_RECORD("%d found in table, %d links, chain_ref %s",state->num,bcnt,state->chain_ref ? "present" : "absent");
      return 0;
      }
   cmpres = _mm256_cmpeq_epi32(headers,zeropad);
   res = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres));
   state->last_pos = __builtin_ctzl(res);
   LOG_RECORD("%d not found in table",state->num);
   ht_add_8(ht,state);
   add_state(empty,state);
   return 1;
   }

static inline int look_ht_12_add(FHashTable *ht,FProcessState *state,FProcessStateSmallSet *set_data,FProcessStateBigSet *empty)
   {
   __m256i shifts = _mm256_set1_epi32(2);
   uint32_t *block_ref = (uint32_t *)state->chain_ref;
   __m256i headers = _mm256_loadu_si256((__m256i *)state->chain_ref);
   __m256i zeropad = _mm256_setzero_si256();
   __m256i base = _mm256_set_epi64x((uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data,(uint64_t)ht->data);
   
   __m256i search = _mm256_set1_epi8(state->key_buf[0]);
   uint32_t bits = block_ref[3];
   uint32_t bit8 = 0xEFF * (state->key_buf[1] & 0x1);
   uint32_t bit9 = 0xEFF * ((state->key_buf[1] & 0x2) >> 1);
   __m256i cmpres = _mm256_cmpeq_epi8(headers,search);
   __m256i refs1 = _mm256_loadu_si256((__m256i *)&block_ref[4]);
   __m256i refs2 = _mm256_loadu_si256((__m256i *)&block_ref[8]);

   int res = _mm256_movemask_epi8(cmpres);
   res &= ~(bits ^ bit8) & (~(bits >> 16) ^ bit9) & 0xEFF;
   
   int refs1size = __builtin_popcount(res & 0xF);
   refs1 = _mm256_permutevar8x32_epi32(refs1,_mm256_loadu_si256((__m256i *)&lut12_1[res & 0xF][0]));
   refs2 = _mm256_permutevar8x32_epi32(refs2,_mm256_loadu_si256((__m256i *)&lut12_2[res >> 4][0]));

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
   int bcnt = __builtin_popcount(res); // 1 for chain link
   uint32_t nidx = block_ref[15];
   state->last_chain_ref = state->chain_ref;
   state->chain_ref = state->data_refs[bcnt] = nidx ? &ht->table[nidx * CACHE_LINE_SIZE] : NULL;
   state->data_refs[bcnt + 1] = NULL;
   ht->tick += 25;
   ht->pcnt--;

   if (state->data_refs[0])
      { // Unpredictable branch, no prefetch here
      state->data_ref = &state->data_refs[0];
      set_data->states[set_data->count++] = state;
      LOG_RECORD("%d found in table, %d links, chain_ref %s",state->num,bcnt,state->chain_ref ? "present" : "absent");
      return 0;
      }
   cmpres = _mm256_cmpeq_epi8(headers,zeropad);
   res = _mm256_movemask_epi8(cmpres);
   res &= ~(bits | (bits >> 16)) & 0xEFF;

   state->last_pos = __builtin_ctzl(res);
   ht_add_12(ht,state);
   add_state(empty,state);

   if (state->data_refs[0])
      { // Unpredictable branch, no prefetch here
      state->data_ref = &state->data_refs[0];
      set_data->states[set_data->count++] = state;
      return 0;
      }
   return 1;
   }

// 4 by step
int look_in_data_4_states(FHashTable *ht,FProcessStateBigSet *set_data,FProcessStateBigSet *set_headers,FOutput *output)
   {
   uint64_t res_mask,key_mask;
   int rv = 0;

   ht->tick += 15;
   FProcessState *state0 = get_state(set_data);
   __m256i data00 = _mm256_loadu_si256((__m256i *)(*state0->data_ref + 4));
   __m256i data01 = _mm256_loadu_si256((__m256i *)(*state0->data_ref + 36));
   __m256i key00 = _mm256_loadu_si256((__m256i *)state0->key_buf); 
   __m256i key01 = _mm256_loadu_si256((__m256i *)&state0->key_buf[32]); 

   FProcessState *state1 = get_state(set_data);
   __m256i data10 = _mm256_loadu_si256((__m256i *)(*state1->data_ref + 4));
   __m256i data11 = _mm256_loadu_si256((__m256i *)(*state1->data_ref + 36));
   __m256i key10 = _mm256_loadu_si256((__m256i *)state1->key_buf); 
   __m256i key11 = _mm256_loadu_si256((__m256i *)&state1->key_buf[32]); 

   key_mask = (1LL << state0->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key01,data01)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key00,data00));
   if ((res_mask & key_mask) == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)output->pos,key00), _mm256_storeu_si256((__m256i *)(output->pos+32),key01);
      output->pos[state0->key_size] = '\n';
      output->pos += state0->key_size + 1;
      ht->pcnt--;
      rv++;
      LOG_RECORD("%d found in table",state0->num);
      }
   else if (*(++state0->data_ref))
      {
      _mm_prefetch(*state0->data_ref,_MM_HINT_T2);
      state0->tick = ht->tick += 5;
      add_state((*state0->data_ref == state0->chain_ref) ? set_headers : set_data,state0);
      LOG_RECORD("%d false positive, goto %s",state0->num,(*state0->data_ref == state0->chain_ref)?"lh":"ld");
      }
   else
      {
      ht->pcnt--, rv++;
      LOG_RECORD("%d false positive, not found in table",state0->num);
      }

   FProcessState *state2 = get_state(set_data);
   __m256i data20 = _mm256_loadu_si256((__m256i *)(*state2->data_ref + 4));
   __m256i data21 = _mm256_loadu_si256((__m256i *)(*state2->data_ref + 36));
   __m256i key20 = _mm256_loadu_si256((__m256i *)state2->key_buf); 
   __m256i key21 = _mm256_loadu_si256((__m256i *)&state2->key_buf[32]);  

   key_mask = (1LL << state1->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key10,data10)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key11,data11));
   if ((res_mask & key_mask) == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)output->pos,key10), _mm256_storeu_si256((__m256i *)(output->pos+32),key11);
      output->pos[state1->key_size] = '\n';
      output->pos += state1->key_size + 1;
      ht->pcnt--;
      rv++;
      LOG_RECORD("%d found in table",state1->num);
      }
   else if (*(++state1->data_ref))
      {
      _mm_prefetch(*state1->data_ref,_MM_HINT_T2);
      state1->tick = ht->tick += 5;
      add_state((*state1->data_ref == state1->chain_ref) ? set_headers : set_data,state1);
      LOG_RECORD("%d false positive, goto %s",state1->num,(*state1->data_ref == state1->chain_ref)?"lh":"ld");
      }
   else
      {
      ht->pcnt--, rv++;
      LOG_RECORD("%d false positive, not found in table",state1->num);
      }

   FProcessState *state3 = get_state(set_data);
   __m256i data30 = _mm256_loadu_si256((__m256i *)(*state3->data_ref + 4));
   __m256i data31 = _mm256_loadu_si256((__m256i *)(*state3->data_ref + 36));
   __m256i key30 = _mm256_loadu_si256((__m256i *)state3->key_buf); 
   __m256i key31 = _mm256_loadu_si256((__m256i *)&state3->key_buf[32]); 

   key_mask = (1LL << state2->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key20,data20)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key21,data21));
   if ((res_mask & key_mask) == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)output->pos,key20), _mm256_storeu_si256((__m256i *)(output->pos+32),key21);
      output->pos[state2->key_size] = '\n';
      output->pos += state2->key_size + 1;
      ht->pcnt--;
      rv++;
      LOG_RECORD("%d found in table",state2->num);
      }
   else if (*(++state2->data_ref))
      {
      _mm_prefetch(*state2->data_ref,_MM_HINT_T2);
      state2->tick = ht->tick += 5;
      add_state((*state2->data_ref == state2->chain_ref) ? set_headers : set_data,state2);
      LOG_RECORD("%d false positive, goto %s",state2->num,(*state2->data_ref == state2->chain_ref)?"lh":"ld");
      }
   else
      {
      ht->pcnt--, rv++;
      LOG_RECORD("%d false positive, not found in table",state2->num);
      }

   key_mask = (1LL << state3->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key30,data30)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key31,data31));
   if ((res_mask & key_mask) == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)output->pos,key30), _mm256_storeu_si256((__m256i *)(output->pos+32),key31);
      output->pos[state3->key_size] = '\n';
      output->pos += state3->key_size + 1;
      ht->pcnt--;
      rv++;
      LOG_RECORD("%d found in table",state3->num);
      }
   else if (*(++state3->data_ref))
      {
      _mm_prefetch(*state3->data_ref,_MM_HINT_T2);
      state3->tick = ht->tick += 5;
      add_state((*state3->data_ref == state3->chain_ref) ? set_headers : set_data,state3);
      LOG_RECORD("%d false positive, goto %s",state3->num,(*state3->data_ref == state3->chain_ref)?"lh":"ld");
      }
   else
      {
      ht->pcnt--, rv++;
      LOG_RECORD("%d false positive, not found in table",state3->num);
      }
   return rv;
   }

int look_in_data_add(FHashTable *ht,FProcessState *state,FProcessStateBigSet *set_data,FProcessStateBigSet *set_headers)
   {
   uint64_t res_mask,key_mask;

   ht->tick += 15;
   __m256i data0 = _mm256_loadu_si256((__m256i *)(*state->data_ref + 4));
   __m256i data1 = _mm256_loadu_si256((__m256i *)(*state->data_ref + 36));
   __m256i key0 = _mm256_loadu_si256((__m256i *)state->key_buf); 
   __m256i key1 = _mm256_loadu_si256((__m256i *)&state->key_buf[32]); 

   key_mask = (1LL << state->key_size) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key1,data1)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key0,data0));
   if ((res_mask & key_mask) == key_mask)
      { // Unlikely
      LOG_RECORD("%d found in table, not added",state->num);
      return 1;
      }
   if (*(++state->data_ref))
      {
      _mm_prefetch(*state->data_ref,_MM_HINT_T2);
      state->tick = ht->tick += 5;
      if (*state->data_ref == state->chain_ref)
         {
         LOG_RECORD("%d false positive, looking next line",state->num);
         add_state(set_headers,state);
         }
      else
         {
         LOG_RECORD("%d false positive, looking next link",state->num);
         add_state(set_data,state);
         }
      return 0;
      }
   LOG_RECORD("%d not found in table and has no links",state->num);
   return 2;
   }

void reset_state(FProcessState *state)
   {
   state->data_refs[0] = state->chain_ref = NULL;
   state->data_ref = &state->data_refs[0];
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

void process_add_8(FHashTable *ht,FSource *source, FProcessState *states)
   {
   FProcessState *state;
   int i,j;

   ht->pcnt = 0;
   ht->tick = 0;

   FProcessStateSmallSet set_by_size[8] = {0};
   FProcessStateSmallSet unpref2={0};
   FProcessStateBigSet unpref={0},lh={0}, ld={0}, empty = {0};

   for (i = 0; i < BIG_SET_SIZE; i++)
      add_state(&empty,&states[i]);

   while (source->inputpos < source->inputlen)
      {
      while (empty.count)
         {
         state = get_state(&empty);
         reset_state(state);
         process_row(source,state);

         int ks = state->key_size / 8;
         set_by_size[ks].states[set_by_size[ks].count++] = state;

         if (set_by_size[ks].count == 16)
            {
            make_hashes16(ht,(int *)states,&set_by_size[ks],ks+1);
            for(j = 0; j < 16; j++)
               add_state(&unpref,set_by_size[ks].states[j]);
            set_by_size[ks].count = 0;
            }
         if (source->inputpos >= source->inputlen) break;
         }
      while (ht->pcnt < MAX_PREFETCH && unpref.count)
         {
         state = get_state(&unpref);
         _mm_prefetch(*state->data_ref,_MM_HINT_T2);
         ht->pcnt++;
         state->tick = ht->tick + MEM_DELAY;
         add_state((*state->data_ref == state->chain_ref) ? &lh : &ld,state);
         }
      while(lh.count && get_state_link(&lh,0)->tick <= ht->tick)
         look_ht_8_add(ht,get_state(&lh),&unpref2,&empty);
      for(j = 0; j < unpref2.count; j++)
         {
         state = unpref2.states[j];
         _mm_prefetch(*state->data_ref,_MM_HINT_T2);
         ht->pcnt++;
         state->tick = ht->tick + MEM_DELAY;
         add_state((*state->data_ref == state->chain_ref) ? &lh : &ld,state);
         }
      unpref2.count = 0;
      while(ld.count && get_state_link(&ld,0)->tick <= ht->tick)
         {
         state = get_state(&ld);
         switch (look_in_data_add(ht,state,&ld,&lh))
            {
            case 2: ht_add_8(ht,state);
            case 1: ht->pcnt--; add_state(&empty,state);
            }
         }
      ht->tick += 20;
      }
   }

void process_add_12(FHashTable *ht,FSource *source, FProcessState *states)
   {
   FProcessState *state;
   int i,j;

   ht->pcnt = 0;
   ht->tick = 0;

   FProcessStateSmallSet set_by_size[8] = {0};
   FProcessStateSmallSet unpref2={0};
   FProcessStateBigSet unpref={0},lh={0}, ld={0}, empty = {0};

   for (i = 0; i < BIG_SET_SIZE; i++)
      add_state(&empty,&states[i]);

   while (source->inputpos < source->inputlen)
      {
      while (empty.count)
         {
         state = get_state(&empty);
         reset_state(state);
         process_row(source,state);

         int ks = state->key_size / 8;
         set_by_size[ks].states[set_by_size[ks].count++] = state;

         if (set_by_size[ks].count == 16)
            {
            make_hashes16(ht,(int *)states,&set_by_size[ks],ks+1);
            for(j = 0; j < 16; j++)
               add_state(&unpref,set_by_size[ks].states[j]);
            set_by_size[ks].count = 0;
            }
         if (source->inputpos >= source->inputlen) break;
         }
      while (ht->pcnt < MAX_PREFETCH && unpref.count)
         {
         state = get_state(&unpref);
         _mm_prefetch(*state->data_ref,_MM_HINT_T2);
         ht->pcnt++;
         state->tick = ht->tick + MEM_DELAY;
         add_state((*state->data_ref == state->chain_ref) ? &lh : &ld,state);
         }
      while(lh.count && get_state_link(&lh,0)->tick <= ht->tick)
         look_ht_12_add(ht,get_state(&lh),&unpref2,&empty);
      for(j = 0; j < unpref2.count; j++)
         {
         state = unpref2.states[j];
         _mm_prefetch(*state->data_ref,_MM_HINT_T2);
         ht->pcnt++;
         state->tick = ht->tick + MEM_DELAY;
         add_state((*state->data_ref == state->chain_ref) ? &lh : &ld,state);
         }
      unpref2.count = 0;
      while(ld.count && get_state_link(&ld,0)->tick <= ht->tick)
         {
         state = get_state(&ld);
         switch (look_in_data_add(ht,state,&ld,&lh))
            {
            case 2: ht_add_12(ht,state);
            case 1: ht->pcnt--; add_state(&empty,state);
            }
         }
      ht->tick += 20;
      }
   }

double process_search_8(FHashTable *ht,FSource *source, FProcessState *states,FOutput *output)
   {
   FProcessState *state;
   int64_t tm = 0;
   int i,j,k;
   FProcessStateSmallSet set_by_size[8] = {0};
   FProcessStateSmallSet unpref2={0};
   FProcessStateBigSet unpref={0},lh={0}, ld={0};
   int processed = 0;

   ht->pcnt = 0;
   ht->tick = 0;

   for (k = 0; k < ITER_COUNT; k++)
      {
      for (i = 0; i < STATES_STEP; i++)
         {
         state = &states[k * STATES_STEP + i];
         process_row(source,state);
         }
      ht->tick += 5000;

      int64_t t1 = get_nanotime();
      for (i = 0; i < STATES_STEP; i++)
         {
         state = &states[k * STATES_STEP + i];
         int ks = state->key_size / 8;
         set_by_size[ks].states[set_by_size[ks].count++] = state;

         if (set_by_size[ks].count == 16)
            {
            make_hashes16(ht,(int *)states,&set_by_size[ks],ks+1);
            for(j = 0; j < 16; j++)
               {
               LOG_RECORD("%d added to unpref",set_by_size[ks].states[j]->num);
               add_state(&unpref,set_by_size[ks].states[j]);
               }
            set_by_size[ks].count = 0;
            }

         while (ht->pcnt < MAX_PREFETCH && unpref.count)
            {
            state = get_state(&unpref);
            _mm_prefetch(*state->data_ref,_MM_HINT_T2);
            ht->pcnt++;
            state->tick = ht->tick + MEM_DELAY;
            add_state((*state->data_ref == state->chain_ref) ? &lh : &ld,state);
            LOG_RECORD("%d prefetched and pushed in %s",state->num,(*state->data_ref == state->chain_ref) ? "lh":"ld");
            }
         while(lh.count && get_state_link(&lh,0)->tick <= ht->tick)
            processed += look_ht_8(ht,get_state(&lh),&unpref2);
         for(j = 0; j < unpref2.count; j++)
            {
            state = unpref2.states[j];
            _mm_prefetch(*state->data_ref,_MM_HINT_T2);
            ht->pcnt++;
            state->tick = ht->tick + MEM_DELAY;
            add_state((*state->data_ref == state->chain_ref) ? &lh : &ld,state);
            LOG_RECORD("%d prefetched 2 and pushed in %s",state->num,(*state->data_ref == state->chain_ref) ? "lh":"ld");
            }
         unpref2.count = 0;
         while(ld.count >= 4 && get_state_link(&ld,3)->tick <= ht->tick)
            processed += look_in_data_4_states(ht,&ld,&lh,output);
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
   int i,j,k;
   FProcessStateSmallSet set_by_size[8] = {0};
   FProcessStateSmallSet unpref2={0};
   FProcessStateBigSet unpref={0},lh={0}, ld={0};
   int processed = 0;

   ht->pcnt = 0;
   ht->tick = 0;

   for (k = 0; k < ITER_COUNT; k++)
      {
      for (i = 0; i < STATES_STEP; i++)
         {
         state = &states[k * STATES_STEP + i];
         process_row(source,state);
         }
      ht->tick += 5000;

      int64_t t1 = get_nanotime();
      for (i = 0; i < STATES_STEP; i++)
         {
         state = &states[k * STATES_STEP + i];
         int ks = state->key_size / 8;
         set_by_size[ks].states[set_by_size[ks].count++] = state;

         if (set_by_size[ks].count == 16)
            {
            make_hashes16(ht,(int *)states,&set_by_size[ks],ks+1);
            for(j = 0; j < 16; j++)
               add_state(&unpref,set_by_size[ks].states[j]);
            set_by_size[ks].count = 0;
            }

         while (ht->pcnt < MAX_PREFETCH && unpref.count)
            {
            state = get_state(&unpref);
            _mm_prefetch(*state->data_ref,_MM_HINT_T2);
            ht->pcnt++;
            state->tick = ht->tick + MEM_DELAY;
            add_state((*state->data_ref == state->chain_ref) ? &lh : &ld,state);
            }
         while(lh.count && get_state_link(&lh,0)->tick <= ht->tick)
            processed += look_ht_12(ht,get_state(&lh),&unpref2);
         for(j = 0; j < unpref2.count; j++)
            {
            state = unpref2.states[j];
            _mm_prefetch(*state->data_ref,_MM_HINT_T2);
            ht->pcnt++;
            state->tick = ht->tick + MEM_DELAY;
            add_state((*state->data_ref == state->chain_ref) ? &lh : &ld,state);
            }
         unpref2.count = 0;
         while(ld.count >= 4 && get_state_link(&ld,3)->tick <= ht->tick)
            processed += look_in_data_4_states(ht,&ld,&lh,output);
         ht->tick += 20;
         }
      tm += get_nanotime() - t1;
      }
   return (double)tm/processed;
   }

int main(int argc, char *argv[ ])
   {
   FHashTable ht;
   ht.table_size = ht.unlocated = TABLE_SIZE;
   ht.table = aligned_alloc(CACHE_LINE_SIZE,ht.table_size * 2 * CACHE_LINE_SIZE);
   memset(ht.table,0,ht.table_size * 2 * CACHE_LINE_SIZE);
   ht.data = aligned_alloc(CACHE_LINE_SIZE,ITEMS_COUNT * 128);
   memset(ht.data,0,ITEMS_COUNT * 128);
   ht.data_pos = 1;

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

   FSource source,source2;
   init_source(&source,filedata,sz1);

   FParseParams parse_params = {'\t','\t',0,3,__builtin_popcount(3)};

   FProcessState *states = aligned_alloc(CACHE_LINE_SIZE,sizeof(FProcessState) * STATES_COUNT);
   int i;
   FOutput output;
   output.start = output.pos = (char *)malloc(sz1 + 32);

   for (i = 0; i < STATES_COUNT; i++)
      states[i].pp = &parse_params, states[i].num = i;
   
   for (i = 0; i < STATES_COUNT; i++)
      reset_state(&states[i]);

   process_add_8(&ht,&source,states);

   init_source(&source,filedata,sz1);
   for (i = 0; i < STATES_COUNT; i++)
      reset_state(&states[i]);

   double tm8_1 = process_search_8(&ht,&source,states,&output);

   init_source(&source2,filedata2,sz2);
   for (i = 0; i < STATES_COUNT; i++)
      reset_state(&states[i]);
   output.pos = output.start;

   double tm8_2 = process_search_8(&ht,&source2,states,&output);

   ht.unlocated = TABLE_SIZE;
   memset(ht.table,0,ht.table_size * 2 * CACHE_LINE_SIZE);
   memset(ht.data,0,ITEMS_COUNT * 128);
   ht.data_pos = 1;

   init_source(&source,filedata,sz1);
   for (i = 0; i < STATES_COUNT; i++)
      reset_state(&states[i]);

   process_add_12(&ht,&source,states);

   init_source(&source,filedata,sz1);
   for (i = 0; i < STATES_COUNT; i++)
      reset_state(&states[i]);
   output.pos = output.start;

   double tm12_1 = process_search_12(&ht,&source,states,&output);

   init_source(&source2,filedata2,sz2);
   for (i = 0; i < STATES_COUNT; i++)
      reset_state(&states[i]);
   output.pos = output.start;

   double tm12_2 = process_search_12(&ht,&source2,states,&output);

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
   CLOSE_LOG;
   return 0;
   }
