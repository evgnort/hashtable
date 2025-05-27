#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>

#include <setjmp.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

#include "hashtable.h"
#include "csvparse.h"

#define HASH_PRIME 591798841
#define HASH_INITIAL 2166136261

#define MEM_DELAY 200

#define ITEMS_COUNT 1000000
#define TABLE_SIZE (ITEMS_COUNT / 8)

void make_hashes(FHashTable *ht,int *base,FProcessStateSet *reqs,int size)
   {
   int rsize = sizeof(FProcessState);
   int iindexes[8] = {  ((int *)reqs->states[0]) - base, ((int *)reqs->states[1]) - base, ((int *)reqs->states[2]) - base, ((int *)reqs->states[3]) - base, 
                        ((int *)reqs->states[4]) - base, ((int *)reqs->states[5]) - base, ((int *)reqs->states[6]) - base, ((int *)reqs->states[7]) - base };
   __m256i indexes = _mm256_loadu_si256((__m256i *)iindexes);

   __m256i incs = _mm256_set1_epi32(1);
   __m256i leftshifts = _mm256_set1_epi32(5);
   __m256i rightshifts = _mm256_set1_epi32(27);
   __m256i primes = _mm256_set1_epi32(HASH_PRIME);
   __m256i hashes = _mm256_set1_epi32(HASH_INITIAL);
   do
      {
      __m256i a = _mm256_i32gather_epi32(base,indexes,4);
      indexes = _mm256_add_epi32(indexes,incs);
      __m256i hashes1 = _mm256_sllv_epi32(hashes,leftshifts);
      __m256i hashes2 = _mm256_srlv_epi32(hashes,rightshifts);
      hashes = _mm256_or_si256(hashes1,hashes2);
      hashes = _mm256_xor_si256(hashes,a);
      hashes = _mm256_mul_epi32(hashes,primes);
      size--;
      }
   while (size > 0);
   uint32_t i,res[8];
   _mm256_storeu_si256((__m256i *)res,hashes);
   for(i = 0; i < 8; i++)
      reqs->states[i]->data_ref[0] = reqs->states[i]->chain_ref = &ht->table[(res[i] % TABLE_SIZE) * CACHE_LINE_SIZE];
   }

static off_t file_size(const char *filename)
	{
	struct stat st;
	if (!filename || !filename[0] || stat(filename, &st) != 0 || !S_ISREG(st.st_mode))
		return -1;
	return st.st_size;
	}

static inline FProcessState *get_state(FProcessStateSet *set)
   { 
   set->count--; 
   FProcessState *rv = set->states[set->first];
   set->first = (set->first + 1) % SET_SIZE;
   return rv;
   }

static inline void add_state(FProcessStateSet *set,FProcessState *state)
   {
   set->count++;
   set->states[set->last] = state;
   set->last = (set->last + 1) % SET_SIZE;
   }

static inline void set_inc_by(FProcessStateSet *set,int count)
   {
   set->count += count;
   set->last = (set->last + count) % SET_SIZE;
   }

static inline void set_dec_by(FProcessStateSet *set,int count)
   {
   set->count += count;
   set->first = (set->first + count) % SET_SIZE;
   }

int lut8[128][8] = {0};

void make_lut8(void)
   {
   int i,j;
   for (i = 0; i < 128; i++)
      {
      int v = i, bn, pos = 0;
      while (bn = __builtin_ffsl(v))
         {
         lut8[i][pos++] = bn;
         v >>= bn;
         }
      }
   }

// 8 keys in line
void look_ht_8(FHashTable *ht,FProcessState *state,FProcessStateSet *set_data)
   {
   int rv = 0, rn, i;
   __m256i zeropad = _mm256_setzero_si256();
   __m256i base = _mm256_set_epi64x((uint64_t)ht->data,(uint64_t)ht->table,(uint64_t)ht->table,(uint64_t)ht->table);
   
   __m256i headers = _mm256_loadu_si256((__m256i *)state->chain_ref);
   __m256i search = _mm256_set1_epi32(*((int *)state->key_buf));
   __m256i cmpres = _mm256_cmpeq_epi32(headers,search);
   __m256i refs = _mm256_loadu_si256((__m256i *)&state->chain_ref[8]);
   int res = _mm256_movemask_ps(_mm256_castsi256_ps(cmpres));
   refs = _mm256_permutevar8x32_epi32(refs,_mm256_loadu_si256((__m256i *)&lut8[res][0]));

   __m256i links1 = _mm256_unpacklo_epi32(refs,zeropad);
   __m256i links2 = _mm256_unpackhi_epi32(refs,zeropad);

   links1 = _mm256_add_epi64(links1,base);
   links2 = _mm256_add_epi64(links2,base);

   _mm256_storeu_si256((__m256i *)&state->data_refs[0],links1);
   _mm256_storeu_si256((__m256i *)&state->data_refs[3],links2);
   int bcnt = __builtin_popcount(res); // 1 for chain link
   int nidx = *((int *)&state->chain_ref[60]);
   state->chain_ref = state->data_refs[bcnt] = nidx ? &ht->table[(TABLE_SIZE + nidx) * CACHE_LINE_SIZE] : NULL;
   state->data_refs[bcnt + 1] = NULL;
   ht->tick += 25;
   ht->pcnt--;
   if (state->data_ref[0])
      { // Unpredictable branch
      state->tick = ht->tick + 10; // Delaying prefetch to avoid bandwidth pollution
      state->data_ref = &(state->data_ref[0]);
      add_state(set_data,state);
      }
   }

// 4 by step
char *look_in_data_4_states(FHashTable *ht,FProcessStateSet *set_data,FProcessStateSet *set_headers,char *buf)
   {
   uint64_t res_mask,key_mask;
   int i;

   ht->tick += 15;
   FProcessState *state0 = get_state(set_data);
   __m256i data00 = _mm256_loadu_si256((__m256i *)*state0->data_ref);
   __m256i data01 = _mm256_loadu_si256((__m256i *)(*state0->data_ref + 32));
   __m256i key00 = _mm256_loadu_si256((__m256i *)state0->key_buf); 
   __m256i key01 = _mm256_loadu_si256((__m256i *)&state0->key_buf[32]); 

   FProcessState *state1 = get_state(set_data);
   __m256i data10 = _mm256_loadu_si256((__m256i *)*state1->data_ref);
   __m256i data11 = _mm256_loadu_si256((__m256i *)(*state1->data_ref + 32));
   __m256i key10 = _mm256_loadu_si256((__m256i *)state1->key_buf); 
   __m256i key11 = _mm256_loadu_si256((__m256i *)&state1->key_buf[32]); 

   key_mask = (1LL << 64 - __builtin_clz(state0->key_size)) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key00,data00)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key01,data01));
   if (res_mask & key_mask == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)buf,key00), _mm256_storeu_si256((__m256i *)(buf+32),key01);
      buf[state0->key_size] = '\n';
      buf += state0->key_size + 1;
      ht->pcnt--;
      }
   else if (*(++state0->data_ref))
      {
      _mm_prefetch(*state0->data_ref,_MM_HINT_T2);
      state0->tick = ht->tick += 5;
      add_state((*state0->data_ref == state0->chain_ref) ? set_headers : set_data,state0);
      }

   FProcessState *state2 = get_state(set_data);
   __m256i data20 = _mm256_loadu_si256((__m256i *)*state2->data_ref);
   __m256i data21 = _mm256_loadu_si256((__m256i *)(*state2->data_ref + 32));
   __m256i key20 = _mm256_loadu_si256((__m256i *)state2->key_buf); 
   __m256i key21 = _mm256_loadu_si256((__m256i *)&state2->key_buf[32]);  

   key_mask = (1LL << 64 - __builtin_clz(state1->key_size)) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key10,data10)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key11,data11));
   if (res_mask & key_mask == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)buf,key10), _mm256_storeu_si256((__m256i *)(buf+32),key11);
      buf[state1->key_size] = '\n';
      buf += state1->key_size + 1;
      ht->pcnt--;
      }
   else if (*(++state1->data_ref))
      {
      _mm_prefetch(*state1->data_ref,_MM_HINT_T2);
      state1->tick = ht->tick += 5;
      add_state((*state1->data_ref == state1->chain_ref) ? set_headers : set_data,state1);
      }

   FProcessState *state3 = get_state(set_data);
   __m256i data30 = _mm256_loadu_si256((__m256i *)*state3->data_ref);
   __m256i data31 = _mm256_loadu_si256((__m256i *)(*state3->data_ref + 32));
   __m256i key30 = _mm256_loadu_si256((__m256i *)state3->key_buf); 
   __m256i key31 = _mm256_loadu_si256((__m256i *)&state3->key_buf[32]); 

   key_mask = (1LL << 64 - __builtin_clz(state2->key_size)) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key10,data10)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key11,data11));
   if (res_mask & key_mask == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)buf,key10), _mm256_storeu_si256((__m256i *)(buf+32),key11);
      buf[state2->key_size] = '\n';
      buf += state2->key_size + 1;
      ht->pcnt--;
      }
   else if (*(++state2->data_ref))
      {
      _mm_prefetch(*state2->data_ref,_MM_HINT_T2);
      state2->tick = ht->tick += 5;
      add_state((*state2->data_ref == state2->chain_ref) ? set_headers : set_data,state2);
      }

   key_mask = (1LL << 64 - __builtin_clz(state3->key_size)) - 1;
   res_mask = ((uint64_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(key10,data10)) << 32) | _mm256_movemask_epi8(_mm256_cmpeq_epi8(key11,data11));
   if (res_mask & key_mask == key_mask)
      { // Likely
      _mm256_storeu_si256((__m256i *)buf,key10), _mm256_storeu_si256((__m256i *)(buf+32),key11);
      buf[state3->key_size] = '\n';
      buf += state3->key_size + 1;
      ht->pcnt--;
      }
   else if (*(++state3->data_ref))
      {
      _mm_prefetch(*state3->data_ref,_MM_HINT_T2);
      state3->tick = ht->tick += 5;
      add_state((*state3->data_ref == state3->chain_ref) ? set_headers : set_data,state3);
      }
   return buf;
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

int main(void)
   {
   FHashTable ht;
   ht.table = aligned_alloc(CACHE_LINE_SIZE,TABLE_SIZE * 2 * CACHE_LINE_SIZE);
   memset(ht.table,0,TABLE_SIZE * 2 * CACHE_LINE_SIZE);
   ht.data = aligned_alloc(CACHE_LINE_SIZE,ITEMS_COUNT * 128);
   memset(ht.data,0,ITEMS_COUNT * 128);
   ht.data_pos = 0;

   const char *filename1 = "input1.csv";
   const char *filename2 = "input2.csv";

   off_t sz1 = file_size(filename1);
   off_t sz2 = file_size(filename2);

   char *filedata = (char *)malloc(sz1 + 10);

   char *output_buf,*output;
   output = output_buf = (char *)malloc(sz1 + 10);

   off_t rdd = 0;
   int fd = open(filename1,O_RDONLY);
   while (rdd < sz1)
      rdd += read(fd,filedata + rdd,sz1 - rdd);
   close(fd);
   filedata[sz1] = 0;

   make_lut8();
   FSource source = {
      .input = filedata,
      .inputlen = sz1,
      .inputpos = 0,
      .lastpart = {0},
      .lp_size = 0,
      .eod_exit = {0}
      };

   FParseParams parse_params = {'\t','\t',0,3,__builtin_popcount(3)};

   FProcessState *states = aligned_alloc(CACHE_LINE_SIZE,sizeof(FProcessState) * 512);
   int states_count = 0;
   int i,j,k;
   FProcessState *state;

   for (i = 0; i < 512; i++)
      {
      states[i].pp = &parse_params;
      }   
   
   FProcessStateSet set_by_size[16] = {0};
   FProcessStateSet unpref={0}, unpref2={0}, lh={0}, ld={0};
   int prefcnt = 0;

   ht.tick=0;
   int64_t tm = 0;

   for (k = 0; k < 20; k++)
      {
      for (i = 0; i < 512; i++)
         {
         reset_state(&states[i]);
         process_row(&source,&states[i]);
         }

      int64_t t1 = get_nanotime();

      for (i = 0; i < 512; i++)
         {
         int ks = states[i].key_size / 4;
         add_state(&set_by_size[ks],&states[i]);
         if (set_by_size[ks].count == 8)
            {
            make_hashes(&ht,(int *)states,&set_by_size[ks],ks+1);
            for(j = 0; j < 8; j++)
               add_state(&unpref,set_by_size[ks].states[j]);
            set_by_size[ks].count = set_by_size[ks].first = set_by_size[ks].last = 0;
            }

         while (ht.pcnt < 16 && unpref.count)
            {
            state = get_state(&unpref);
            _mm_prefetch(*state->data_ref,_MM_HINT_T2);
            ht.pcnt++;
            state->tick = ht.tick + MEM_DELAY;
            add_state((*state->data_ref == state->chain_ref) ? &lh : &ld,state);
            }
         while (ht.pcnt < 16 && unpref2.count && unpref2.states[0]->tick <= ht.tick)
            {
            state = get_state(&unpref2);
            _mm_prefetch(*state->data_ref,_MM_HINT_T2);
            ht.pcnt++;
            state->tick = ht.tick + MEM_DELAY;
            add_state((*state->data_ref == state->chain_ref) ? &lh : &ld,state);
            }
         while(lh.count && lh.states[0]->tick <= ht.tick)
            look_ht_8(&ht,get_state(&lh),&unpref2);
         while(ld.count >= 4 && ld.states[3]->tick <= ht.tick)
            output = look_in_data_4_states(&ht,&ld,&lh,output);
         ht.tick += 4;
         }
      int64_t t2 = get_nanotime();
      tm += t2 - t1;
      }

   free(ht.table);
   free(ht.data);
   free(filedata);
   free(states);
   free(output_buf);
   printf("%ld\n",tm/10240);
   return 0;
   }
