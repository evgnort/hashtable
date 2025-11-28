# Copyright (C) Evgeniy Buevich

#include "port.h"

#include <immintrin.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "hashtable.h"
#include "csvparse.h"

#define HASH_PRIME 591798841
#define HASH_INITIAL 2166136261

static void normalize_set(FNormalRqSet *set)
   {
   if (set->last < STATES_COUNT)
      return;
   int count = set->last - set->first;
   int pos = 0;
   __m256i zeropad = _mm256_setzero_si256();
   while (pos < count)
      {
      __m256i data = _mm256_loadu_si256((__m256i *)&set->indexes[set->first + pos]);
      _mm256_storeu_si256((__m256i *)&set->indexes[set->first + pos], zeropad);
      _mm256_storeu_si256((__m256i *)&set->indexes[pos], data);
      pos += 8;
      }
   set->first = 0;
   set->last = count;
   }

static void normalize_hashed_set(FHashedRqSet *set)
   {
   if (set->last < STATES_COUNT)
      return;
   int count = set->last - set->first;
   int pos = 0;
   while (pos < count)
      {
      __m256i data = _mm256_loadu_si256((__m256i *)&set->indexes[set->first + pos]);
      __m256i hashes = _mm256_loadu_si256((__m256i *)&set->hashes[set->first + pos]);
      _mm256_storeu_si256((__m256i *)&set->indexes[pos], data);
      _mm256_storeu_si256((__m256i *)&set->hashes[pos], hashes);
      pos += 8;
      }
   set->first = 0;
   set->last = count;
   }

//#define READABLE
//#define NO_MEM_WORK

#ifndef READABLE

#define HEADERS_CHANELS 8
#define DATA_CHANELS 8

double pipeline(FHashTable *ht, FSource *source, char *output, double *acycle, int stat)
   {
   FProcessState *states = aligned_alloc(CACHE_LINE_SIZE,sizeof(FProcessState) * (STATES_COUNT+1));

   int i;
   uint32_t *dummy_ref = (uint32_t *)ht->table;

   memset(&states[0],0,sizeof(FProcessState));
   for (i = 0; i <= STATES_COUNT; i++)
      {
      states[i].sp.offset = i * sizeof(FProcessState);
      states[i].sp.inc = 1;
      states[i].value_start = &ht->value_store[i * 128];
      }

   states[0].key_buf[0] = 1;
   states[0].sp.inc = 0;
   states[0].key_size = 1;

   states[0].next_chain_ref = states[0].cur_chain_ref = dummy_ref;
   states[0].cur_data_ref = (char *)&ht->data[0];

   __m256i zeropad = _mm256_setzero_si256();
//   __m128i zeropad128 = _mm_setzero_si128();
   _mm256_storeu_si256((__m256i *)&ht->table[0], zeropad);
   _mm256_storeu_si256((__m256i *)&ht->table[32], zeropad);

   __m256i empty_mask1 = _mm256_set1_epi32(-1),indexes1 = _mm256_setzero_si256(), hashes1 = _mm256_setzero_si256(), nhashes1;
   __m256i empty_mask2 = _mm256_set1_epi32(-1),indexes2 = _mm256_setzero_si256(), hashes2 = _mm256_setzero_si256(), nhashes2;

   __m256i hashes_initial = _mm256_set1_epi32(HASH_INITIAL);

   __m256i incs1 = _mm256_set1_epi64x(1); // 
   __m256i incs4 = _mm256_set1_epi32(4); // Step of key_datas increase
   __m256i index_mask = _mm256_set1_epi32(~0x7F); 

   uint64_t dr = 0xFFFFFFFFFFFFFFFFLL / (uint64_t)ht->table_size + 1;
   __m256i divr = _mm256_set1_epi64x(dr);
   __m256i divr_hi = _mm256_srli_epi64(divr,32);
   __m256i div = _mm256_set1_epi64x((uint64_t)ht->table_size);

   __m256i primes = _mm256_set1_epi32(HASH_PRIME);
   __m256i table_base = _mm256_set1_epi64x((uint64_t)ht->table);
   __m256i states_base = _mm256_set1_epi64x((uint64_t)states);

   int scalar_empty_mask1 = 0xFF, empty_cnt1 = 8; // All slots are empty at start
   int scalar_empty_mask2 = 0xFF, empty_cnt2 = 8;

   int processed = 0;

   FNormalRqSet empty = {0};
   FNormalRqSet loaded = {0};
   FHashedRqSet hashed = {0};
   FNormalRqSet unpref = {0};
   FNormalRqSet unpref_datas = {0};

   FNormalRqSet *worksets[4] = {&empty,&unpref,&unpref_datas,&unpref_datas};
   const int64_t cnt2mask64[16] = {-1LL,-1LL,-1LL,-1LL,-1LL,-1LL,-1LL,-1LL,0,0,0,0,0,0,0,0};
   const int32_t *cnt2mask32 = (const int32_t *)&cnt2mask64[4];
   const uint64_t numbers = 0x0706050403020100;

   FProcessState *hstates_w[8] = {states,states,states,states,states,states,states,states};
   FProcessState *dstates_w[8] = {states,states,states,states,states,states,states,states};

   int64_t tm = 0;

   for (i = 0; i < STATES_COUNT ; i++)
      empty.indexes[i] = (i + 1) * sizeof(FProcessState);
   empty.last = STATES_COUNT;

   int cnum = 0;
   int hcnt = 0,hdcnt = 0,dcnt = 0;
   int loads = 0;

   int mode = 2;
   int modecnt[3] = {0,0,0};
   int modeswitches = 0;

   __m256i unpack_scheme1 = _mm256_set_epi32(7,6,5,4,3,2,1,0);
   __m256i unpack_scheme2 = unpack_scheme1;

   while (source->input_pos < source->input_end)
      {
      process_csv_plain(source,states,&empty,&loaded);
      loads++;
      normalize_hashed_set(&hashed);
      normalize_set(&empty);
      normalize_set(&loaded);
      normalize_set(&unpref);
      normalize_set(&unpref_datas);

      int pel = empty.last;

      static int mode_translation[3][9] = {{2,2,1,0,0,0,0,0,0},{2,2,1,1,1,1,1,0,0},{2,2,2,2,2,2,1,0,0}};

      __m256i unpref_indexes;

      int64_t t1 = get_nanotime();

      while (loaded.count >= empty_cnt1 + empty_cnt2)
         {
         int rpcnt = STATES_COUNT - (empty.last - empty.first) - loaded.count; // Count of processing request.
         int nmode = mode_translation[mode][rpcnt / 32]; 
         cnum++;
         modecnt[mode]++;
         modeswitches += (mode != nmode);
         mode = nmode;

//         printf("mode %d, rpcnt %d, loaded %d, hashed %d, unpref %d, unpref_data %d, empty %d\n",mode,rpcnt,loaded.count,hashed.last - hashed.first,unpref.last - unpref.first,unpref_datas.last - unpref_datas.first,empty.last - empty.first);
         int step1 = mode;
         int step2 = mode != 2 || (cnum & 1);

   //         printf("-------\n");

         if (step1) 
            {
/*1*/       __m256i new_indexes1 = _mm256_loadu_si256((__m256i *)&loaded.indexes[loaded.first]); // Load next empty_cnt1 loaded requests indexes
/*1*/       loaded.first += empty_cnt1;

/*1*/       __m256i new_indexes2 = _mm256_loadu_si256((__m256i *)&loaded.indexes[loaded.first]); // Load next empty_cnt2 loaded requests indexes
/*1*/       loaded.first += empty_cnt2;

/*1*/       loaded.count -= empty_cnt1;
/*1*/       loaded.count -= empty_cnt2;

            // Redistribute loaded indexes to empty slots in vector
/*1*/       new_indexes1 = _mm256_permutevar8x32_epi32(new_indexes1,unpack_scheme1);
/*1*/       new_indexes2 = _mm256_permutevar8x32_epi32(new_indexes2,unpack_scheme2);

/*1*/       indexes1 = _mm256_blendv_epi8(indexes1, new_indexes1, empty_mask1); // Merging new indexes with present indexes
/*1*/       indexes2 = _mm256_blendv_epi8(indexes2, new_indexes2, empty_mask2); 

   /*2*/    uint32_t hashed_valid_cnt = hashed.last - hashed.first;
   /*2*/    hashed_valid_cnt = (hashed_valid_cnt > 8) ? 8 : hashed_valid_cnt;

         // Loading indexes from hashed chain
   /*2*/    __m256i hashed_indexes = _mm256_loadu_si256((__m256i *)&hashed.indexes[hashed.first]); 
   /*2*/    __m128i h1i32 = _mm_loadu_si128((__m128i *)&hashed.hashes[hashed.first]);
   /*2*/    __m128i h2i32 = _mm_loadu_si128((__m128i *)&hashed.hashes[hashed.first + 4]);

/*1*/       __m256i key_datas1 = _mm256_i32gather_epi32((int *)states,indexes1,1); // (4)->(18)(p0 + p015 + 8*p23 + p5) Loading new portion of data - latency

   /*2*/    __m256i h1i64 = _mm256_cvtepu32_epi64(h1i32); // (3)p5 (VPMOVZXDQ)
   /*2*/    __m256i h2i64 = _mm256_cvtepu32_epi64(h2i32); // (3)p5 (VPMOVZXDQ)
   /*2*/    _mm256_storeu_si256((__m256i*)&hashed.indexes[hashed.first],zeropad); // Replacing them by zero indexes
   /*2*/    hashed.first += hashed_valid_cnt;

   /*2*/    __m256i m1 = _mm256_loadu_si256((__m256i*)&cnt2mask64[8-hashed_valid_cnt]);
   /*2*/    __m256i m2 = _mm256_loadu_si256((__m256i*)&cnt2mask64[12-hashed_valid_cnt]);

   /*2*/    __m256i lb1 = _mm256_add_epi64( _mm256_mul_epu32(divr,h1i64), _mm256_slli_epi64(_mm256_mul_epu32(divr_hi,h1i64),32));
   /*2*/    __m256i lb2 = _mm256_add_epi64( _mm256_mul_epu32(divr,h2i64), _mm256_slli_epi64(_mm256_mul_epu32(divr_hi,h2i64),32));

/*1*/       __m256i key_datas2 = _mm256_i32gather_epi32((int *)states,indexes2,1); 

/*1*/       indexes1 = _mm256_add_epi32(indexes1,incs4); // Increasing indexes to the next portion of data
/*1*/       indexes2 = _mm256_add_epi32(indexes2,incs4);

   /*2*/    __m256i rem1 = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(lb1,32),div),_mm256_srli_epi64(_mm256_mul_epu32(lb1,div),32));
   /*2*/    __m256i rem2 = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(lb2,32),div),_mm256_srli_epi64(_mm256_mul_epu32(lb2,div),32));

   /*2*/    __m256i hr1 = _mm256_srli_epi64(rem1,32);

   /*2*/    hr1 = _mm256_add_epi32(hr1,incs1); // Adding ones to all
   /*2*/    hr1 = _mm256_and_si256(hr1,m1); // Applying mask of validness - resetting invalid to zero
   /*2*/    hr1 = _mm256_slli_epi32(hr1,6);

   /*2*/    __m256i hr2 = _mm256_srli_epi64(rem2,32);

   /*2*/    hr2 = _mm256_add_epi32(hr2,incs1); // Adding ones to all
   /*2*/    hr2 = _mm256_and_si256(hr2,m2); // Applying mask of validness - resetting invalid to zero
   /*2*/    hr2 = _mm256_slli_epi32(hr2,6);

   /*2*/    __m256i hashed_refs1 = _mm256_add_epi64(hr1,table_base); // (1)p015 (VPADDQ)

/*1*/       __m256i key_datas12 = _mm256_i32gather_epi32((int *)states,indexes1,1); // Loading second portion of data - latency
            
            // Releasing p5 by scalar processing two 32 bits indexes as 64 bit value
   /*2*/    uint64_t idx2 = PART_64_256(hashed_indexes,0); // (2)p0 (MOVD), (2)p0 (MOVQ)
   /*2*/    ((FProcessState *)((char *)states + (uint32_t)idx2))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs1,0); // (2)p0 (MOVQ)
   /*2*/    ((FProcessState *)((char *)states + (idx2 >> 32LL)))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs1,1); // vpextrq ymm->xmm (p0 + p5) + vpextrq ymm->mem (p237 + p4 + p5)

   /*2*/    idx2 = PART_64_256(hashed_indexes,1); // (2)p0 (MOVD), (2)p0 (MOVQ)
   /*2*/    ((FProcessState *)((char *)states + (uint32_t)idx2))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs1,2);
   /*2*/    ((FProcessState *)((char *)states + (idx2 >> 32LL)))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs1,3);

   /*2*/    __m256i hashed_refs2 = _mm256_add_epi64(hr2,table_base); // (1)p015 (VPADDQ)

/*1*/       __m256i key_datas22 = _mm256_i32gather_epi32((int *)states,indexes2,1);

            // Increasing indexes to the next portion of data
/*1*/       indexes1 = _mm256_add_epi32(indexes1,incs4); // (1)p015 (VPADDD)
/*1*/       indexes2 = _mm256_add_epi32(indexes2,incs4); // (1)p015 (VPADDD)

   /*2*/    idx2 = PART_64_256(hashed_indexes,2); // (2)p0 (MOVD), (2)p0 (MOVQ)
   /*2*/    ((FProcessState *)((char *)states + (uint32_t)idx2))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs2,0); 
   /*2*/    ((FProcessState *)((char *)states + (idx2 >> 32LL)))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs2,1); 
            
   /*2*/    idx2 = PART_64_256(hashed_indexes,3); // (2)p0 (MOVQ), (2)p0 (MOVD)
   /*2*/    ((FProcessState *)((char *)states + (uint32_t)idx2))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs2,2); 
   /*2*/    ((FProcessState *)((char *)states + (idx2 >> 32LL)))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs2,3); 

            if (step2) 
               {   
               unpref_indexes = _mm256_loadu_si256((__m256i *)&unpref.indexes[unpref.first]); // Loading indexes from unpref chain
               uint32_t unpref_valid_cnt = unpref.last - unpref.first;
               unpref_valid_cnt = (unpref_valid_cnt > 8) ? 8 : unpref_valid_cnt;
               _mm256_storeu_si256((__m256i *)&unpref.indexes[unpref.first],zeropad); // Replacing them by zero indexes
               unpref.first += unpref_valid_cnt;
               hdcnt += unpref_valid_cnt;
               }

#ifdef NO_MEM_WORK
   /*2*/    _mm256_storeu_si256((__m256i*)&empty.indexes[empty.last],hashed_indexes); // storing them in unpref queue as is
   /*2*/    empty.last += hashed_valid_cnt;
#else 
   /*2*/    _mm256_storeu_si256((__m256i*)&unpref.indexes[unpref.last],hashed_indexes); // storing them in unpref queue as is
   /*2*/    unpref.last += hashed_valid_cnt;
#endif
   /*2*/    hcnt += hashed_valid_cnt;
   
            // Shifting hashes
/*1*/       nhashes1 = _mm256_or_si256(_mm256_slli_epi32(hashes1,5), _mm256_srli_epi32(hashes1,27)); // ((1)p01,(1)p01)-> (1)p015
/*1*/       nhashes2 = _mm256_or_si256(_mm256_slli_epi32(hashes2,5), _mm256_srli_epi32(hashes2,27)); // ((1)p01,(1)p01)-> (1)p015

            // Merging hashes with initial values
/*1*/       nhashes1 = _mm256_blendv_epi8(nhashes1, hashes_initial, empty_mask1); // (4)p23 (VPBLENDVB) -> (2)2*p015 (VPBLENDVB)
/*1*/       nhashes2 = _mm256_blendv_epi8(nhashes2, hashes_initial, empty_mask2); // (4)p23 (VPBLENDVB) -> (2)2*p015 (VPBLENDVB)

            // xoring hashes with new portion of data
/*1*/       nhashes1 = _mm256_xor_si256(nhashes1,key_datas1); // (1)p015 (PXOR)
/*1*/       nhashes2 = _mm256_xor_si256(nhashes2,key_datas2); // (1)p015 (PXOR)

            // Multiplying with primes - latency
/*1*/       nhashes1 = _mm256_mullo_epi32(nhashes1,primes); // (10)2*p01 
/*1*/       nhashes2 = _mm256_mullo_epi32(nhashes2,primes); // (10)2*p01

            // Obtaing mask of elements with zero rest size
            // (1)p015 (VPXOR)
/*1*/       empty_mask1 = _mm256_cmpeq_epi32(key_datas1,_mm256_setzero_si256()); // (1)p01 (VPCMPEQDD)
/*1*/       empty_mask2 = _mm256_cmpeq_epi32(key_datas2,_mm256_setzero_si256()); // (1)p01 (VPCMPEQDD)

//          Second iteration

            // Restoring old hash values for finished on first step
/*1*/       nhashes1 = _mm256_blendv_epi8(nhashes1, hashes1, empty_mask1); // (2)2*p015 (VPBLENDVB)
/*1*/       nhashes2 = _mm256_blendv_epi8(nhashes2, hashes2, empty_mask2); // (2)2*p015 (VPBLENDVB)

/*1*/       empty_mask1 = _mm256_cmpeq_epi32(key_datas12,_mm256_setzero_si256()); // (1)p01 (VPCMPEQDD)
/*1*/       empty_mask2 = _mm256_cmpeq_epi32(key_datas22,_mm256_setzero_si256()); // (1)p01 (VPCMPEQDD)

            // Obtain scalar mask of empty slots
/*1*/       scalar_empty_mask1 = _mm256_movemask_ps(_mm256_castsi256_ps(empty_mask1)); // (3)p0
/*1*/       scalar_empty_mask2 = _mm256_movemask_ps(_mm256_castsi256_ps(empty_mask2)); // (3)p0

/*1*/       uint64_t byte_mask1 = _pdep_u64(scalar_empty_mask1, 0x0101010101010101) * 0xFF; // (3)p1 (PDEP) -> [(3)p1 (IMUL) or (1)p06 (SAR) -> (1)p0156 (SUB)]
/*1*/       uint64_t byte_mask2 = _pdep_u64(scalar_empty_mask2, 0x0101010101010101) * 0xFF; // (3)p1 (PDEP) -> [(3)p1 (IMUL) or (1)p06 (SAR) -> (1)p0156 (SUB)]
         
/*1*/       empty_cnt1 = __builtin_popcount(scalar_empty_mask1); // (3)p1 (POPCNT)
/*1*/       empty_cnt2 = __builtin_popcount(scalar_empty_mask2); // (3)p1 (POPCNT)

/*1*/       hashes1 = _mm256_or_si256(_mm256_slli_epi32(nhashes1,5), _mm256_srli_epi32(nhashes1,27)); // ((1)p01,(1)p01)-> (1)p015
/*1*/       hashes2 = _mm256_or_si256(_mm256_slli_epi32(nhashes2,5), _mm256_srli_epi32(nhashes2,27)); // ((1)p01,(1)p01)-> (1)p015

            // Packing data from empty slots 1
/*1*/       __m256i pack_scheme1 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(_pext_u64(numbers, byte_mask1))); // (3)p1 (PEXT) -> (2)p5 (MOVQ) -> (3)p5 (VPMOVZXBD)
/*1*/       __m256i pack_scheme2 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(_pext_u64(numbers, byte_mask2))); // (3)p1 (PEXT) -> (2)p5 (MOVQ) -> (3)p5 (VPMOVZXBD)

            // xoring hashes with new portion of data
/*1*/       hashes1 = _mm256_xor_si256(hashes1,key_datas12); // (1)p015 (PXOR)
/*1*/       hashes2 = _mm256_xor_si256(hashes2,key_datas22); // (1)p015 (PXOR)

/*1*/       __m256i packed_indexes1 = _mm256_permutevar8x32_epi32(indexes1,pack_scheme1); // (3)p5 (VPERMD)
/*1*/       __m256i packed_indexes2 = _mm256_permutevar8x32_epi32(indexes2,pack_scheme2); // (3)p5 (VPERMD)

            // Multiplying with primes - latency
/*1*/       hashes1 = _mm256_mullo_epi32(hashes1,primes); // (10)2*p01
/*1*/       hashes2 = _mm256_mullo_epi32(hashes2,primes); // (10)2*p01

/*1*/       __m256i packed_hashes1 = _mm256_permutevar8x32_epi32(nhashes1,pack_scheme1); // (3)p5 (VPERMD)
/*1*/       __m256i packed_hashes2 = _mm256_permutevar8x32_epi32(nhashes2,pack_scheme2);  // (3)p5 (VPERMD)

/*1*/       packed_indexes1 = _mm256_and_si256(packed_indexes1,index_mask); // (4)p23 (VPAND) -> (1)p015 (VPAND)
/*1*/       packed_indexes2 = _mm256_and_si256(packed_indexes2,index_mask); // (4)p23 (VPAND) -> (1)p015 (VPAND)

            // Storing those elements to hashed queue
/*1*/       _mm256_storeu_si256((__m256i *)&hashed.indexes[hashed.last],packed_indexes1);
/*1*/       _mm256_storeu_si256((__m256i *)&hashed.hashes[hashed.last],packed_hashes1);
/*1*/       hashed.last += empty_cnt1;

/*1*/       unpack_scheme1 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(_pdep_u64(numbers, byte_mask1))); // (3)p1 (PDEP) -> (2)p5 (MOVQ) -> (3)p5 (VPMOVZXBD)
/*1*/       unpack_scheme2 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(_pdep_u64(numbers, byte_mask2))); // (3)p1 (PDEP) -> (2)p5 (MOVQ) -> (3)p5 (VPMOVZXBD)

/*1*/       _mm256_storeu_si256((__m256i *)&hashed.indexes[hashed.last],packed_indexes2);
/*1*/       _mm256_storeu_si256((__m256i *)&hashed.hashes[hashed.last],packed_hashes2);
/*1*/       hashed.last += empty_cnt2;

/*1*/       _mm256_storeu_si256((__m256i *)&hashed.indexes[hashed.last],_mm256_setzero_si256());
            }
         else
            {
   /*2*/    uint32_t hashed_valid_cnt = hashed.last - hashed.first;
   /*2*/    hashed_valid_cnt = (hashed_valid_cnt > 8) ? 8 : hashed_valid_cnt;

         // Loading indexes from hashed chain
   /*2*/    __m256i hashed_indexes = _mm256_loadu_si256((__m256i *)&hashed.indexes[hashed.first]); 
   /*2*/    __m128i h1i32 = _mm_loadu_si128((__m128i *)&hashed.hashes[hashed.first]);
   /*2*/    __m128i h2i32 = _mm_loadu_si128((__m128i *)&hashed.hashes[hashed.first + 4]);
   /*2*/    __m256i h1i64 = _mm256_cvtepu32_epi64(h1i32); // (3)p5 (VPMOVZXDQ)
   /*2*/    __m256i h2i64 = _mm256_cvtepu32_epi64(h2i32); // (3)p5 (VPMOVZXDQ)
   /*2*/    _mm256_storeu_si256((__m256i*)&hashed.indexes[hashed.first],zeropad); // Replacing them by zero indexes
   /*2*/    hashed.first += hashed_valid_cnt;

   /*2*/    __m256i m1 = _mm256_loadu_si256((__m256i*)&cnt2mask64[8-hashed_valid_cnt]);
   /*2*/    __m256i m2 = _mm256_loadu_si256((__m256i*)&cnt2mask64[12-hashed_valid_cnt]);

   /*2*/    __m256i lb1 = _mm256_add_epi64( _mm256_mul_epu32(divr,h1i64), _mm256_slli_epi64(_mm256_mul_epu32(divr_hi,h1i64),32));
   /*2*/    __m256i lb2 = _mm256_add_epi64( _mm256_mul_epu32(divr,h2i64), _mm256_slli_epi64(_mm256_mul_epu32(divr_hi,h2i64),32));
   /*2*/    __m256i rem1 = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(lb1,32),div),_mm256_srli_epi64(_mm256_mul_epu32(lb1,div),32));
   /*2*/    __m256i rem2 = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(lb2,32),div),_mm256_srli_epi64(_mm256_mul_epu32(lb2,div),32));

   /*2*/    __m256i hr1 = _mm256_srli_epi64(rem1,32);

   /*2*/    hr1 = _mm256_add_epi32(hr1,incs1); // Adding ones to all
   /*2*/    hr1 = _mm256_and_si256(hr1,m1); // Applying mask of validness - resetting invalid to zero
   /*2*/    hr1 = _mm256_slli_epi32(hr1,6);

   /*2*/    __m256i hr2 = _mm256_srli_epi64(rem2,32);

   /*2*/    hr2 = _mm256_add_epi32(hr2,incs1); // Adding ones to all
   /*2*/    hr2 = _mm256_and_si256(hr2,m2); // Applying mask of validness - resetting invalid to zero
   /*2*/    hr2 = _mm256_slli_epi32(hr2,6);

   /*2*/    __m256i hashed_refs1 = _mm256_add_epi64(hr1,table_base); // (1)p015 (VPADDQ)
   /*2*/    uint64_t idx2 = PART_64_256(hashed_indexes,0); // (2)p0 (MOVD), (2)p0 (MOVQ)
   /*2*/    ((FProcessState *)((char *)states + (uint32_t)idx2))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs1,0); // (2)p0 (MOVQ)
   /*2*/    ((FProcessState *)((char *)states + (idx2 >> 32LL)))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs1,1); // vpextrq ymm->xmm (p0 + p5) + vpextrq ymm->mem (p237 + p4 + p5)

   /*2*/    idx2 = PART_64_256(hashed_indexes,1); // (2)p0 (MOVD), (2)p0 (MOVQ)
   /*2*/    ((FProcessState *)((char *)states + (uint32_t)idx2))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs1,2);
   /*2*/    ((FProcessState *)((char *)states + (idx2 >> 32LL)))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs1,3);

   /*2*/    __m256i hashed_refs2 = _mm256_add_epi64(hr2,table_base); // (1)p015 (VPADDQ)

   /*2*/    idx2 = PART_64_256(hashed_indexes,2); // (2)p0 (MOVD), (2)p0 (MOVQ)
   /*2*/    ((FProcessState *)((char *)states + (uint32_t)idx2))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs2,0); 
   /*2*/    ((FProcessState *)((char *)states + (idx2 >> 32LL)))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs2,1);
            
   /*2*/    idx2 = PART_64_256(hashed_indexes,3); // (2)p0 (MOVQ), (2)p0 (MOVD)
   /*2*/    ((FProcessState *)((char *)states + (uint32_t)idx2))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs2,2);
   /*2*/    ((FProcessState *)((char *)states + (idx2 >> 32LL)))->next_chain_ref = (uint32_t *)PART_64_256(hashed_refs2,3);

            unpref_indexes = _mm256_loadu_si256((__m256i *)&unpref.indexes[unpref.first]); // Loading indexes from unpref chain
            uint32_t unpref_valid_cnt = unpref.last - unpref.first;
            unpref_valid_cnt = (unpref_valid_cnt > 8) ? 8 : unpref_valid_cnt;
            _mm256_storeu_si256((__m256i *)&unpref.indexes[unpref.first],zeropad); // Replacing them by zero indexes
            unpref.first += unpref_valid_cnt;
            hdcnt += unpref_valid_cnt;

#ifdef NO_MEM_WORK
   /*2*/    _mm256_storeu_si256((__m256i*)&empty.indexes[empty.last],hashed_indexes); // storing them in unpref queue as is
   /*2*/    empty.last += hashed_valid_cnt;
#else 
   /*2*/    _mm256_storeu_si256((__m256i*)&unpref.indexes[unpref.last],hashed_indexes); // storing them in unpref queue as is
   /*2*/    unpref.last += hashed_valid_cnt;
#endif
   /*2*/    hcnt += hashed_valid_cnt;
            }

         if (!step2) continue;

         __m256i unpref_datas_indexes = _mm256_loadu_si256((__m256i *)&unpref_datas.indexes[unpref_datas.first]); // Loading indexes from unpref chain
         uint32_t unpref_datas_valid_cnt = unpref_datas.last - unpref_datas.first;
         unpref_datas_valid_cnt = (unpref_datas_valid_cnt > 8) ? 8 : unpref_datas_valid_cnt;
         __m256i unpref_mask = _mm256_loadu_si256((__m256i*)&cnt2mask32[8-unpref_datas_valid_cnt]);
         unpref_datas_indexes = _mm256_and_si256(unpref_datas_indexes,unpref_mask);

//         _mm256_storeu_si256((__m256i *)&unpref_datas.indexes[unpref_datas.first],zeropad); // Replacing them by zero indexes
         unpref_datas.first += unpref_datas_valid_cnt;
         dcnt += unpref_datas_valid_cnt;

         // (4)p23 (VMOVDQA) - loading states_base
         __m256i hstates_lo = _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_castsi256_si128(unpref_indexes)),states_base); // (3)p5 (VPMOVZXDQ) -> (1)p015 (VPADDQ)
         __m256i dstates_lo = _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_castsi256_si128(unpref_datas_indexes)),states_base); // (3)p5 (VPMOVZXDQ) -> (1)p015 (VPADDQ)

         for(i = 0; i < 4; i++)
            {
/*1*/       FProcessState * restrict hstate_w = hstates_w[i]; // (4)p23 (MOV)
   /*2*/    FProcessState * restrict dstate_w = dstates_w[i]; // (4)p23 (MOV)

/*1*/       FProcessState * restrict hstate_p = (FProcessState *)PART_64_256(hstates_lo,0); // (3)p0 (MOVQ)
/*1*/       hstates_lo = _mm256_permute4x64_epi64(hstates_lo,0b00111001); // (3)p5 (VPERMQ)

   /*2*/    FProcessState * restrict dstate_p = (FProcessState *)PART_64_256(dstates_lo,0); // (3)p0 (MOVQ)
   /*2*/    dstates_lo = _mm256_permute4x64_epi64(dstates_lo,0b00111001); // (3)p5 (VPERMQ)

/*1*/       uint32_t tmp = *((uint32_t *)hstate_w->key_buf); // (4)p23 (MOV)  Trick to force compiler to one data load 
/*1*/       uint32_t * restrict chain_ref = hstate_w->cur_chain_ref = hstate_w->next_chain_ref; // (4)p23 (MOV) -> (2)p237->(1)p4

/*1*/       char *ncr = (char *)hstate_p->next_chain_ref; // (4)p23 (MOV)

/*1*/       uint32_t src_bits = 0x7FF * (((tmp >> 8) & 0x1) + ((tmp & 0x200) << 7)); // (1)p06 (SHR) -> (1)p0156 (AND) + (1)p0156 (AND) -> (1)p06 (SHL)

   /*2*/    int dnum = 0xF & _tzcnt_u32(dstate_p->items_mask); // (3)p1 (TZCNT) -> (1)p0156 (AND)

   /*2*/    __m256i *kb = (__m256i *)dstate_w->key_buf,*dr = (__m256i *)dstate_w->cur_data_ref,*ot = (__m256i *)output; // No op
   /*2*/    __m256i key0 = _mm256_loadu_si256(kb);  // (4)p23 (VMOVDQU)
   /*2*/    __m256i data0 = _mm256_loadu_si256(dr); // (4)p23 (VMOVDQU)
   /*2*/    int ks = dstate_w->key_size;

            uint32_t ccr = dstate_p->cur_chain_ref[dnum]; // (4)p23 (MOV) -> (4)p23 (MOV))
/*1*/       __m128i search = _mm_cvtsi64_si128((uint64_t)tmp); // (2)p5 (MOVQ)
/*1*/       search = _mm_shuffle_epi8(search,_mm_setzero_si128()); // (1)p5 (VPSHUFB)

   /*2*/    dstate_p->cur_data_ref = (char *)&ht->data[ccr]; // ((4)p23 (MOV) -> (1)p15 (LEA)

/*1*/       _mm_prefetch(ncr,_MM_HINT_T2); // (1)p23 (PREFETCHT2)
   /*2*/    _mm_prefetch(dstate_p->cur_data_ref,_MM_HINT_T2); // (1)p23 (PREFETCHT2)

/*1*/       __m128i headers_set = _mm_loadu_si128((__m128i *)chain_ref);

            // Low half for bit8, up half for bit 9
/*1*/       uint32_t bit_res = ~(chain_ref[3] ^ src_bits); 

   /*2*/    uint32_t cmpmask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(key0,data0)); // (1)p01 (VCMPPS) -> (3)p0 (VPMOVMSKB) 

/*1*/       __m128i cmpres = _mm_cmpeq_epi8(headers_set,search); // (1)p01 (VCMPPS)

   /*2*/    int bnum = _tzcnt_u32(~cmpmask); // (1)p0156 -> (3)p1 (TZCNT)
   /*2*/    _mm256_storeu_si256((__m256i *)ot,key0);

/*1*/       uint32_t items_mask = _mm_movemask_epi8(cmpres) & bit_res & (bit_res >> 16) & 0x7FF;

/*1*/       hstate_w->next_chain_ref = (uint32_t *)&ht->table[(int64_t)chain_ref[15] * CACHE_LINE_SIZE];
            hstate_w->chain_cont = chain_ref[15] ? 1 : 0;
/*1*/       hstate_w->items_mask = items_mask << 4; // Shifting by 4 to start of refs in table cacheline to remove 3 component addressing

   /*2*/    while (ks > 32)
   /*2*/       {
   /*2*/       key0 = _mm256_loadu_si256(++kb),data0 = _mm256_loadu_si256(++dr);
   /*2*/       bnum = _tzcnt_u32(~_mm256_movemask_epi8(_mm256_cmpeq_epi8(key0,data0)));
   /*2*/       _mm256_storeu_si256(++ot,key0);
   /*2*/       ks -= 32;
   /*2*/       }

/*1*/       int wsnum = (items_mask ? 2:0) + hstate_w->chain_cont;
            FPStateParams hps = hstate_w->sp;
/*1*/       worksets[wsnum]->indexes[worksets[wsnum]->last] = hps.offset;
/*1*/       worksets[wsnum]->last += hps.inc;

   /*2*/    dstate_w->items_mask = _blsr_u32(dstate_w->items_mask);
   /*2*/    int found = (bnum >= ks);

   /*2*/    output[dstate_w->key_size] = '\n';
   /*2*/    output += found * (dstate_w->key_size + 1);

   /*2*/    int wsnumd = (1 - found) * ((dstate_w->items_mask ? 2:0) + dstate_w->chain_cont);
            FPStateParams dps = dstate_w->sp;
   /*2*/    worksets[wsnumd]->indexes[worksets[wsnumd]->last] = dps.offset;
   /*2*/    worksets[wsnumd]->last += dps.inc;

            }

         __m256i hstates_hi = _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_extracti128_si256(unpref_indexes,1)),states_base); // (3)p5 (VEXTRACTI128) -> (3)p5 (VPMOVZXDQ) -> (1)p015 (VPADDQ)
         __m256i dstates_hi = _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_extracti128_si256(unpref_datas_indexes,1)),states_base); // (3)p5 (VEXTRACTI128) -> (3)p5 (VPMOVZXDQ) -> (1)p015 (VPADDQ)

         _mm256_storeu_si256((__m256i *)&hstates_w[0],hstates_lo); 
         _mm256_storeu_si256((__m256i *)&dstates_w[0],dstates_lo);

         for(i = 4; i < 8; i++)
            {
/*1*/       FProcessState * restrict hstate_w = hstates_w[i]; // (4)p23 (MOV)
   /*2*/    FProcessState * restrict dstate_w = dstates_w[i]; // (4)p23 (MOV)

/*1*/       FProcessState * restrict hstate_p = (FProcessState *)PART_64_256(hstates_hi,0); // (3)p0 (MOVQ)
/*1*/       hstates_hi = _mm256_permute4x64_epi64(hstates_hi,0b00111001); // (3)p5 (VPERMQ)

   /*2*/    FProcessState * restrict dstate_p = (FProcessState *)PART_64_256(dstates_hi,0); // (3)p0 (MOVQ)
   /*2*/    dstates_hi = _mm256_permute4x64_epi64(dstates_hi,0b00111001); // (3)p5 (VPERMQ)

/*1*/       uint32_t tmp = *((uint32_t *)hstate_w->key_buf); // Trick to force compiler to one data load
/*1*/       uint32_t * restrict chain_ref = hstate_w->cur_chain_ref = hstate_w->next_chain_ref;

/*1*/       char *ncr = (char *)hstate_p->next_chain_ref;

   /*2*/    int dnum = 0xF & _tzcnt_u32(dstate_p->items_mask);

   /*2*/    __m256i *kb = (__m256i *)dstate_w->key_buf,*dr = (__m256i *)dstate_w->cur_data_ref,*ot = (__m256i *)output;
   /*2*/    __m256i key0 = _mm256_loadu_si256(kb);
   /*2*/    __m256i data0 = _mm256_loadu_si256(dr);
   /*2*/    int ks = dstate_w->key_size;

            __m128i search = _mm_cvtsi64_si128((uint64_t)tmp);
            search = _mm_shuffle_epi8(search,_mm_setzero_si128());

   /*2*/    dstate_p->cur_data_ref = (char *)&ht->data[dstate_p->cur_chain_ref[dnum]];

/*1*/       _mm_prefetch(ncr,_MM_HINT_T2);
   /*2*/    _mm_prefetch(dstate_p->cur_data_ref,_MM_HINT_T2);

/*1*/       __m128i headers_set = _mm_loadu_si128((__m128i *)chain_ref);

/*1*/       uint32_t src_bits = 0x7FF * (((tmp & 0x100) >> 8) + ((tmp & 0x200) << 7));

/*1*/       uint32_t bit_res = ~(chain_ref[3] ^ src_bits); // Low half for bit8, up half for bit 9

   /*2*/    uint32_t cmpmask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(key0,data0));

/*1*/       __m128i cmpres = _mm_cmpeq_epi8(headers_set,search);
   /*2*/    int bnum = _tzcnt_u32(~cmpmask); // 
   /*2*/    _mm256_storeu_si256((__m256i *)ot,key0);

/*1*/       uint32_t items_mask = _mm_movemask_epi8(cmpres) & bit_res & (bit_res >> 16) & 0x7FF;

/*1*/       hstate_w->next_chain_ref = (uint32_t *)&ht->table[(int64_t)chain_ref[15] * CACHE_LINE_SIZE];
            hstate_w->chain_cont = chain_ref[15] ? 1 : 0;
/*1*/       hstate_w->items_mask = items_mask << 4; // Shifting by 4 to start of refs in table cacheline to remove 3 component addressing
         
   /*2*/    while (ks > 32)
   /*2*/       {
   /*2*/       key0 = _mm256_loadu_si256(++kb),data0 = _mm256_loadu_si256(++dr);
   /*2*/       bnum = _tzcnt_u32(~_mm256_movemask_epi8(_mm256_cmpeq_epi8(key0,data0)));
   /*2*/       _mm256_storeu_si256(++ot,key0);
   /*2*/       ks -= 32;
   /*2*/       }

/*1*/       int wsnum = (items_mask ? 2:0) + hstate_w->chain_cont;
            FPStateParams hps = hstate_w->sp;
/*1*/       worksets[wsnum]->indexes[worksets[wsnum]->last] = hps.offset;
/*1*/       worksets[wsnum]->last += hps.inc;

   /*2*/    dstate_w->items_mask = _blsr_u32(dstate_w->items_mask);
   /*2*/    int found = (bnum >= ks);

   /*2*/    output[dstate_w->key_size] = '\n';
   /*2*/    output += found * (dstate_w->key_size + 1);

   /*2*/    int wsnumd = (1 - found) * ((dstate_w->items_mask ? 2:0) + dstate_w->chain_cont);
            FPStateParams dps = dstate_w->sp;
   /*2*/    worksets[wsnumd]->indexes[worksets[wsnumd]->last] = dps.offset;
   /*2*/    worksets[wsnumd]->last += dps.inc;
            }
         _mm256_storeu_si256((__m256i *)&hstates_w[4],hstates_hi); 
         _mm256_storeu_si256((__m256i *)&dstates_w[4],dstates_hi);
         }

      tm += get_nanotime() - t1;
      processed += empty.last - pel;
      }
   aligned_free(states);
 
   if (stat)
      {
      printf("Hashed : %.2f per cycle\n",(double)hcnt / cnum);
      printf("Headers: %.2f per cycle\n",(double)hdcnt / cnum);
      printf("Datas  : %.2f per cycle\n",(double)dcnt / cnum);
      printf("Cycles : %d\n",cnum);
//      printf("Modes  : %d %d %d, switches %d, loads %d\n",modecnt[0],modecnt[1],modecnt[2],modeswitches,loads);
      }

   *acycle = (double)tm / cnum;
   return (double)tm/processed;
   }

#else

#define HEADERS_CHANELS 8
#define DATA_CHANELS 8

double pipeline(FHashTable *ht, FSource *source, char *output, double *acycle, int stat)
   {
   FProcessState *states = aligned_alloc(CACHE_LINE_SIZE,sizeof(FProcessState) * (STATES_COUNT+1));

   int i;
   uint32_t *dummy_ref = (uint32_t *)ht->table;

   memset(&states[0],0,sizeof(FProcessState));
   for (i = 0; i <= STATES_COUNT; i++)
      {
      states[i].sp.offset = i * sizeof(FProcessState);
      states[i].sp.inc = 1;
      states[i].value_start = &ht->value_store[i * 128];
      }

   states[0].key_buf[0] = 1;
   states[0].sp.inc = 0;
   states[0].key_size = 1;

   states[0].next_chain_ref = states[0].cur_chain_ref = dummy_ref;
   states[0].cur_data_ref = (char *)&ht->data[0];

   __m256i zeropad = _mm256_setzero_si256();
//   __m128i zeropad128 = _mm_setzero_si128();
   _mm256_storeu_si256((__m256i *)&ht->table[0], zeropad);
   _mm256_storeu_si256((__m256i *)&ht->table[32], zeropad);

   __m256i empty_mask1 = _mm256_set1_epi32(-1),indexes1 = _mm256_setzero_si256(), hashes1 = _mm256_setzero_si256(), nhashes1;
   __m256i empty_mask2 = _mm256_set1_epi32(-1),indexes2 = _mm256_setzero_si256(), hashes2 = _mm256_setzero_si256(), nhashes2;

   __m256i hashes_initial = _mm256_set1_epi32(HASH_INITIAL);

   __m256i incs1 = _mm256_set1_epi64x(1); // 
   __m256i incs4 = _mm256_set1_epi32(4); // Step of key_datas increase
   __m256i index_mask = _mm256_set1_epi32(~0x7F); 

   uint64_t dr = 0xFFFFFFFFFFFFFFFFLL / (uint64_t)ht->table_size + 1;
   __m256i divr = _mm256_set1_epi64x(dr);
   __m256i divr_hi = _mm256_srli_epi64(divr,32);
   __m256i div = _mm256_set1_epi64x((uint64_t)ht->table_size);

   __m256i primes = _mm256_set1_epi32(HASH_PRIME);
   __m256i table_base = _mm256_set1_epi64x((uint64_t)ht->table);
   __m256i states_base = _mm256_set1_epi64x((uint64_t)states);

   int scalar_empty_mask1 = 0xFF, empty_cnt1 = 8; // All slots are empty at start
   int scalar_empty_mask2 = 0xFF, empty_cnt2 = 8;

   int processed = 0;

   FNormalRqSet empty = {0};
   FNormalRqSet loaded = {0};
   FHashedRqSet hashed = {0};
   FNormalRqSet unpref = {0};
   FNormalRqSet unpref_datas = {0};

   FNormalRqSet *worksets[4] = {&empty,&unpref,&unpref_datas,&unpref_datas};
   const int64_t cnt2mask[16] = {-1LL,-1LL,-1LL,-1LL,-1LL,-1LL,-1LL,-1LL,0,0,0,0,0,0,0,0};
   const uint64_t numbers = 0x0706050403020100;

   FProcessState *hstates_w[8] = {states,states,states,states,states,states,states,states};
   FProcessState *dstates_w[8] = {states,states,states,states,states,states,states,states};

   int64_t tm = 0;

   for (i = 0; i < STATES_COUNT ; i++)
      empty.indexes[i] = (i + 1) * sizeof(FProcessState);
   empty.last = STATES_COUNT;

   int cnum = 0;
   int hcnt = 0,hdcnt = 0,dcnt = 0;
   int loads = 0;

   int mode = 2;
   int modecnt[3] = {0,0,0};
   int modeswitches = 0;

   __m256i unpack_scheme1 = _mm256_set_epi32(7,6,5,4,3,2,1,0);
   __m256i unpack_scheme2 = unpack_scheme1;

   while (source->input_pos < source->input_end)
      {
      process_csv_plain(source,states,&empty,&loaded);
      loads++;
      normalize_hashed_set(&hashed);
      normalize_set(&empty);
      normalize_set(&loaded);
      normalize_set(&unpref);
      normalize_set(&unpref_datas);

      int pel = empty.last;

      static int mode_translation[3][9] = {{2,2,1,0,0,0,0,0,0},{2,2,1,1,1,1,1,0,0},{2,2,2,2,2,2,1,0,0}};

      __m256i unpref_indexes;

      int64_t t1 = get_nanotime();

      while (loaded.count >= empty_cnt1 + empty_cnt2)
         {
         int rpcnt = STATES_COUNT - (empty.last - empty.first) - loaded.count; // Count of processing request.
         int nmode = mode_translation[mode][rpcnt / 32]; 
         cnum++;
         modecnt[mode]++;
         modeswitches += (mode != nmode);
         mode = nmode;

//         printf("mode %d, rpcnt %d, loaded %d, hashed %d, unpref %d, unpref_data %d, empty %d\n",mode,rpcnt,loaded.count,hashed.last - hashed.first,unpref.last - unpref.first,unpref_datas.last - unpref_datas.first,empty.last - empty.first);
         int step1 = mode;
         int step2 = mode != 2 || (cnum & 1);

   //         printf("-------\n");

         if (step1) 
            {
/*1*/       __m256i new_indexes1 = _mm256_loadu_si256((__m256i *)&loaded.indexes[loaded.first]); // Load next empty_cnt1 loaded requests indexes
/*1*/       loaded.first += empty_cnt1;

/*1*/       __m256i new_indexes2 = _mm256_loadu_si256((__m256i *)&loaded.indexes[loaded.first]); // Load next empty_cnt2 loaded requests indexes
/*1*/       loaded.first += empty_cnt2;

/*1*/       loaded.count -= empty_cnt1;
/*1*/       loaded.count -= empty_cnt2;

            // Redistribute loaded indexes to empty slots in vector
/*1*/       new_indexes1 = _mm256_permutevar8x32_epi32(new_indexes1,unpack_scheme1);
/*1*/       new_indexes2 = _mm256_permutevar8x32_epi32(new_indexes2,unpack_scheme2);

/*1*/       indexes1 = _mm256_blendv_epi8(indexes1, new_indexes1, empty_mask1); // Merging new indexes with present indexes
/*1*/       indexes2 = _mm256_blendv_epi8(indexes2, new_indexes2, empty_mask2); 

/*1*/       __m256i key_datas1 = _mm256_i32gather_epi32((int *)states,indexes1,1); // (4)->(18)(p0 + p015 + 8*p23 + p5) Loading new portion of data - latency

/*1*/       __m256i key_datas2 = _mm256_i32gather_epi32((int *)states,indexes2,1); 

/*1*/       indexes1 = _mm256_add_epi32(indexes1,incs4); // Increasing indexes to the next portion of data
/*1*/       indexes2 = _mm256_add_epi32(indexes2,incs4);

/*1*/       __m256i key_datas12 = _mm256_i32gather_epi32((int *)states,indexes1,1); // Loading second portion of data - latency

/*1*/       __m256i key_datas22 = _mm256_i32gather_epi32((int *)states,indexes2,1);

            // Increasing indexes to the next portion of data
/*1*/       indexes1 = _mm256_add_epi32(indexes1,incs4); // (1)p015 (VPADDD)
/*1*/       indexes2 = _mm256_add_epi32(indexes2,incs4); // (1)p015 (VPADDD)

            // Shifting hashes
/*1*/       nhashes1 = _mm256_or_si256(_mm256_slli_epi32(hashes1,5), _mm256_srli_epi32(hashes1,27)); // ((1)p01,(1)p01)-> (1)p015
/*1*/       nhashes2 = _mm256_or_si256(_mm256_slli_epi32(hashes2,5), _mm256_srli_epi32(hashes2,27)); // ((1)p01,(1)p01)-> (1)p015

            // Merging hashes with initial values
/*1*/       nhashes1 = _mm256_blendv_epi8(nhashes1, hashes_initial, empty_mask1); // (4)p23 (VPBLENDVB) -> (2)2*p015 (VPBLENDVB)
/*1*/       nhashes2 = _mm256_blendv_epi8(nhashes2, hashes_initial, empty_mask2); // (4)p23 (VPBLENDVB) -> (2)2*p015 (VPBLENDVB)

            // xoring hashes with new portion of data
/*1*/       nhashes1 = _mm256_xor_si256(nhashes1,key_datas1); // (1)p015 (PXOR)
/*1*/       nhashes2 = _mm256_xor_si256(nhashes2,key_datas2); // (1)p015 (PXOR)

            // Multiplying with primes - latency
/*1*/       nhashes1 = _mm256_mullo_epi32(nhashes1,primes); // (10)2*p01 
/*1*/       nhashes2 = _mm256_mullo_epi32(nhashes2,primes); // (10)2*p01

            // Obtaing mask of elements with zero rest size
            // (1)p015 (VPXOR)
/*1*/       empty_mask1 = _mm256_cmpeq_epi32(key_datas1,_mm256_setzero_si256()); // (1)p01 (VPCMPEQDD)
/*1*/       empty_mask2 = _mm256_cmpeq_epi32(key_datas2,_mm256_setzero_si256()); // (1)p01 (VPCMPEQDD)

//          Second iteration

            // Restoring old hash values for finished on first step
/*1*/       nhashes1 = _mm256_blendv_epi8(nhashes1, hashes1, empty_mask1); // (2)2*p015 (VPBLENDVB)
/*1*/       nhashes2 = _mm256_blendv_epi8(nhashes2, hashes2, empty_mask2); // (2)2*p015 (VPBLENDVB)

/*1*/       empty_mask1 = _mm256_cmpeq_epi32(key_datas12,_mm256_setzero_si256()); // (1)p01 (VPCMPEQDD)
/*1*/       empty_mask2 = _mm256_cmpeq_epi32(key_datas22,_mm256_setzero_si256()); // (1)p01 (VPCMPEQDD)

            // Obtain scalar mask of empty slots
/*1*/       scalar_empty_mask1 = _mm256_movemask_ps(_mm256_castsi256_ps(empty_mask1)); // (3)p0
/*1*/       scalar_empty_mask2 = _mm256_movemask_ps(_mm256_castsi256_ps(empty_mask2)); // (3)p0

/*1*/       uint64_t byte_mask1 = _pdep_u64(scalar_empty_mask1, 0x0101010101010101) * 0xFF; // (3)p1 (PDEP) -> [(3)p1 (IMUL) or (1)p06 (SAR) -> (1)p0156 (SUB)]
/*1*/       uint64_t byte_mask2 = _pdep_u64(scalar_empty_mask2, 0x0101010101010101) * 0xFF; // (3)p1 (PDEP) -> [(3)p1 (IMUL) or (1)p06 (SAR) -> (1)p0156 (SUB)]
         
/*1*/       empty_cnt1 = __builtin_popcount(scalar_empty_mask1); // (3)p1 (POPCNT)
/*1*/       empty_cnt2 = __builtin_popcount(scalar_empty_mask2); // (3)p1 (POPCNT)

/*1*/       hashes1 = _mm256_or_si256(_mm256_slli_epi32(nhashes1,5), _mm256_srli_epi32(nhashes1,27)); // ((1)p01,(1)p01)-> (1)p015
/*1*/       hashes2 = _mm256_or_si256(_mm256_slli_epi32(nhashes2,5), _mm256_srli_epi32(nhashes2,27)); // ((1)p01,(1)p01)-> (1)p015

            // Packing data from empty slots 1
/*1*/       __m256i pack_scheme1 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(_pext_u64(numbers, byte_mask1))); // (3)p1 (PEXT) -> (2)p5 (MOVQ) -> (3)p5 (VPMOVZXBD)
/*1*/       __m256i pack_scheme2 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(_pext_u64(numbers, byte_mask2))); // (3)p1 (PEXT) -> (2)p5 (MOVQ) -> (3)p5 (VPMOVZXBD)

            // xoring hashes with new portion of data
/*1*/       hashes1 = _mm256_xor_si256(hashes1,key_datas12); // (1)p015 (PXOR)
/*1*/       hashes2 = _mm256_xor_si256(hashes2,key_datas22); // (1)p015 (PXOR)

/*1*/       __m256i packed_indexes1 = _mm256_permutevar8x32_epi32(indexes1,pack_scheme1); // (3)p5 (VPERMD)
/*1*/       __m256i packed_indexes2 = _mm256_permutevar8x32_epi32(indexes2,pack_scheme2); // (3)p5 (VPERMD)

            // Multiplying with primes - latency
/*1*/       hashes1 = _mm256_mullo_epi32(hashes1,primes); // (10)2*p01
/*1*/       hashes2 = _mm256_mullo_epi32(hashes2,primes); // (10)2*p01

/*1*/       __m256i packed_hashes1 = _mm256_permutevar8x32_epi32(nhashes1,pack_scheme1); // (3)p5 (VPERMD)
/*1*/       __m256i packed_hashes2 = _mm256_permutevar8x32_epi32(nhashes2,pack_scheme2);  // (3)p5 (VPERMD)

/*1*/       packed_indexes1 = _mm256_and_si256(packed_indexes1,index_mask); // (4)p23 (VPAND) -> (1)p015 (VPAND)
/*1*/       packed_indexes2 = _mm256_and_si256(packed_indexes2,index_mask); // (4)p23 (VPAND) -> (1)p015 (VPAND)

            // Storing those elements to hashed queue
/*1*/       _mm256_storeu_si256((__m256i *)&hashed.indexes[hashed.last],packed_indexes1);
/*1*/       _mm256_storeu_si256((__m256i *)&hashed.hashes[hashed.last],packed_hashes1);
/*1*/       hashed.last += empty_cnt1;

/*1*/       unpack_scheme1 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(_pdep_u64(numbers, byte_mask1))); // (3)p1 (PDEP) -> (2)p5 (MOVQ) -> (3)p5 (VPMOVZXBD)
/*1*/       unpack_scheme2 = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(_pdep_u64(numbers, byte_mask2))); // (3)p1 (PDEP) -> (2)p5 (MOVQ) -> (3)p5 (VPMOVZXBD)

/*1*/       _mm256_storeu_si256((__m256i *)&hashed.indexes[hashed.last],packed_indexes2);
/*1*/       _mm256_storeu_si256((__m256i *)&hashed.hashes[hashed.last],packed_hashes2);
/*1*/       hashed.last += empty_cnt2;

/*1*/       _mm256_storeu_si256((__m256i *)&hashed.indexes[hashed.last],_mm256_setzero_si256());
            }

/*2*/    uint32_t hashed_valid_cnt = hashed.last - hashed.first;
/*2*/    hashed_valid_cnt = (hashed_valid_cnt > 8) ? 8 : hashed_valid_cnt;

      // Loading indexes from hashed chain
/*2*/    __m256i hashed_indexes = _mm256_loadu_si256((__m256i *)&hashed.indexes[hashed.first]); 
/*2*/    __m128i h1i32 = _mm_loadu_si128((__m128i *)&hashed.hashes[hashed.first]);
/*2*/    __m128i h2i32 = _mm_loadu_si128((__m128i *)&hashed.hashes[hashed.first + 4]);

/*2*/    __m256i h1i64 = _mm256_cvtepu32_epi64(h1i32); // (3)p5 (VPMOVZXDQ)
/*2*/    __m256i h2i64 = _mm256_cvtepu32_epi64(h2i32); // (3)p5 (VPMOVZXDQ)
/*2*/    _mm256_storeu_si256((__m256i*)&hashed.indexes[hashed.first],zeropad); // Replacing them by zero indexes
/*2*/    hashed.first += hashed_valid_cnt;

/*2*/    __m256i m1 = _mm256_loadu_si256((__m256i*)&cnt2mask[8-hashed_valid_cnt]);
/*2*/    __m256i m2 = _mm256_loadu_si256((__m256i*)&cnt2mask[12-hashed_valid_cnt]);

/*2*/    __m256i lb1 = _mm256_add_epi64( _mm256_mul_epu32(divr,h1i64), _mm256_slli_epi64(_mm256_mul_epu32(divr_hi,h1i64),32));
/*2*/    __m256i lb2 = _mm256_add_epi64( _mm256_mul_epu32(divr,h2i64), _mm256_slli_epi64(_mm256_mul_epu32(divr_hi,h2i64),32));

/*2*/    __m256i rem1 = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(lb1,32),div),_mm256_srli_epi64(_mm256_mul_epu32(lb1,div),32));
/*2*/    __m256i rem2 = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(lb2,32),div),_mm256_srli_epi64(_mm256_mul_epu32(lb2,div),32));

/*2*/    __m256i hr1 = _mm256_srli_epi64(rem1,32);

/*2*/    hr1 = _mm256_add_epi32(hr1,incs1); // Adding ones to all
/*2*/    hr1 = _mm256_and_si256(hr1,m1); // Applying mask of validness - resetting invalid to zero
/*2*/    hr1 = _mm256_slli_epi32(hr1,6);

/*2*/    __m256i hr2 = _mm256_srli_epi64(rem2,32);

/*2*/    hr2 = _mm256_add_epi32(hr2,incs1); // Adding ones to all
/*2*/    hr2 = _mm256_and_si256(hr2,m2); // Applying mask of validness - resetting invalid to zero
/*2*/    hr2 = _mm256_slli_epi32(hr2,6);

/*2*/    __m256i hashed_refs1 = _mm256_add_epi64(hr1,table_base); // (1)p015 (VPADDQ)
            
         // Releasing p5 by scalar processing two 32 bits indexes as 64 bit value
/*2*/    uint64_t idx2 = hashed_indexes.m256i_u64[0]; // (2)p0 (MOVD), (2)p0 (MOVQ)
/*2*/    ((FProcessState *)((char *)states + (uint32_t)idx2))->next_chain_ref = (uint32_t *)hashed_refs1.m256i_u64[0]; // (2)p0 (MOVQ)
/*2*/    ((FProcessState *)((char *)states + (idx2 >> 32LL)))->next_chain_ref = (uint32_t *)hashed_refs1.m256i_u64[1]; // vpextrq ymm->xmm (p0 + p5) + vpextrq ymm->mem (p237 + p4 + p5)

/*2*/    idx2 = hashed_indexes.m256i_u64[1]; // (2)p0 (MOVD), (2)p0 (MOVQ)
/*2*/    ((FProcessState *)((char *)states + (uint32_t)idx2))->next_chain_ref = (uint32_t *)hashed_refs1.m256i_u64[2];
/*2*/    ((FProcessState *)((char *)states + (idx2 >> 32LL)))->next_chain_ref = (uint32_t *)hashed_refs1.m256i_u64[3];

/*2*/    __m256i hashed_refs2 = _mm256_add_epi64(hr2,table_base); // (1)p015 (VPADDQ)

/*2*/    idx2 = hashed_indexes.m256i_u64[2]; // (2)p0 (MOVD), (2)p0 (MOVQ)
/*2*/    ((FProcessState *)((char *)states + (uint32_t)idx2))->next_chain_ref = (uint32_t *)hashed_refs2.m256i_u64[0]; 
/*2*/    ((FProcessState *)((char *)states + (idx2 >> 32LL)))->next_chain_ref = (uint32_t *)hashed_refs2.m256i_u64[1];
            
/*2*/    idx2 = hashed_indexes.m256i_u64[3]; // (2)p0 (MOVQ), (2)p0 (MOVD)
/*2*/    ((FProcessState *)((char *)states + (uint32_t)idx2))->next_chain_ref = (uint32_t *)hashed_refs2.m256i_u64[2];
/*2*/    ((FProcessState *)((char *)states + (idx2 >> 32LL)))->next_chain_ref = (uint32_t *)hashed_refs2.m256i_u64[3];

         __m256i unpref_indexes;
         if (step2)
            {
            unpref_indexes = _mm256_loadu_si256((__m256i *)&unpref.indexes[unpref.first]); // Loading indexes from unpref chain
            uint32_t unpref_valid_cnt = unpref.last - unpref.first;
            unpref_valid_cnt = (unpref_valid_cnt > 8) ? 8 : unpref_valid_cnt;
            _mm256_storeu_si256((__m256i *)&unpref.indexes[unpref.first],zeropad); // Replacing them by zero indexes
            unpref.first += unpref_valid_cnt;
            hdcnt += unpref_valid_cnt;
            }

#ifdef NO_MEM_WORK
/*2*/    _mm256_storeu_si256((__m256i*)&empty.indexes[empty.last],hashed_indexes); // storing them in unpref queue as is
/*2*/    empty.last += hashed_valid_cnt;
#else 
/*2*/    _mm256_storeu_si256((__m256i*)&unpref.indexes[unpref.last],hashed_indexes); // storing them in unpref queue as is
/*2*/    unpref.last += hashed_valid_cnt;
#endif
/*2*/    hcnt += hashed_valid_cnt;

         if (!step2)
            continue;

         __m256i unpref_datas_indexes = _mm256_loadu_si256((__m256i *)&unpref_datas.indexes[unpref_datas.first]); // Loading indexes from unpref chain
         uint32_t unpref_datas_valid_cnt = unpref_datas.last - unpref_datas.first;
         unpref_datas_valid_cnt = (unpref_datas_valid_cnt > 8) ? 8 : unpref_datas_valid_cnt;
         _mm256_storeu_si256((__m256i *)&unpref_datas.indexes[unpref_datas.first],zeropad); // Replacing them by zero indexes
         unpref_datas.first += unpref_datas_valid_cnt;
         dcnt += unpref_datas_valid_cnt;

         // (4)p23 (VMOVDQA) - loading states_base
         __m256i hstates_lo = _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_castsi256_si128(unpref_indexes)),states_base); // (3)p5 (VPMOVZXDQ) -> (1)p015 (VPADDQ)
         __m256i dstates_lo = _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_castsi256_si128(unpref_datas_indexes)),states_base); // (3)p5 (VPMOVZXDQ) -> (1)p015 (VPADDQ)

         for(i = 0; i < 4; i++)
            {
/*1*/       FProcessState * restrict hstate_p = (FProcessState *)hstates_lo.m256i_u64[0]; // (3)p0 (MOVQ)
/*1*/       hstates_lo = _mm256_permute4x64_epi64(hstates_lo,0b00111001); // (3)p5 (VPERMQ)
/*1*/       char *ncr = (char *)hstate_p->next_chain_ref; // (4)p23 (MOV)
/*1*/       _mm_prefetch(ncr,_MM_HINT_T2); // (1)p23 (PREFETCHT2)

   /*2*/    FProcessState * restrict dstate_p = (FProcessState *)dstates_lo.m256i_u64[0]; // (3)p0 (MOVQ)
   /*2*/    dstates_lo = _mm256_permute4x64_epi64(dstates_lo,0b00111001); // (3)p5 (VPERMQ)
   /*2*/    int dnum = 0xF & _tzcnt_u32(dstate_p->items_mask); // (3)p1 (TZCNT) -> (1)p0156 (AND)
            uint32_t ccr = dstate_p->cur_chain_ref[dnum]; // (4)p23 (MOV) -> (4)p23 (MOV))
   /*2*/    dstate_p->cur_data_ref = (char *)&ht->data[ccr]; // ((4)p23 (MOV) -> (1)p15 (LEA)
   /*2*/    _mm_prefetch(dstate_p->cur_data_ref,_MM_HINT_T2); // (1)p23 (PREFETCHT2)
            }

         __m256i hstates_hi = _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_extracti128_si256(unpref_indexes,1)),states_base); // (3)p5 (VEXTRACTI128) -> (3)p5 (VPMOVZXDQ) -> (1)p015 (VPADDQ)
         __m256i dstates_hi = _mm256_add_epi64(_mm256_cvtepu32_epi64(_mm256_extracti128_si256(unpref_datas_indexes,1)),states_base); // (3)p5 (VEXTRACTI128) -> (3)p5 (VPMOVZXDQ) -> (1)p015 (VPADDQ)

         for(i = 0; i < 4; i++)
            {
/*1*/       FProcessState * restrict hstate_p = (FProcessState *)hstates_hi.m256i_u64[0]; // (3)p0 (MOVQ)
/*1*/       hstates_hi = _mm256_permute4x64_epi64(hstates_hi,0b00111001); // (3)p5 (VPERMQ)
/*1*/       char *ncr = (char *)hstate_p->next_chain_ref; // (4)p23 (MOV)
/*1*/       _mm_prefetch(ncr,_MM_HINT_T2); // (1)p23 (PREFETCHT2)

   /*2*/    FProcessState * restrict dstate_p = (FProcessState *)dstates_hi.m256i_u64[0]; // (3)p0 (MOVQ)
   /*2*/    dstates_hi = _mm256_permute4x64_epi64(dstates_hi,0b00111001); // (3)p5 (VPERMQ)
   /*2*/    int dnum = 0xF & _tzcnt_u32(dstate_p->items_mask); // (3)p1 (TZCNT) -> (1)p0156 (AND)
            uint32_t ccr = dstate_p->cur_chain_ref[dnum]; // (4)p23 (MOV) -> (4)p23 (MOV))
   /*2*/    dstate_p->cur_data_ref = (char *)&ht->data[ccr]; // ((4)p23 (MOV) -> (1)p15 (LEA)
   /*2*/    _mm_prefetch(dstate_p->cur_data_ref,_MM_HINT_T2); // (1)p23 (PREFETCHT2)
            }


         for(i = 0; i < 8; i++)
            {
/*1*/       FProcessState * restrict hstate_w = hstates_w[i]; // (4)p23 (MOV)

/*1*/       uint32_t tmp = *((uint32_t *)hstate_w->key_buf); // (4)p23 (MOV)  Trick to force compiler to one data load 
/*1*/       uint32_t * restrict chain_ref = hstate_w->cur_chain_ref = hstate_w->next_chain_ref; // (4)p23 (MOV) -> (2)p237->(1)p4

/*1*/       uint32_t src_bits = 0x7FF * (((tmp >> 8) & 0x1) + ((tmp & 0x200) << 7)); // (1)p06 (SHR) -> (1)p0156 (AND) + (1)p0156 (AND) -> (1)p06 (SHL)

/*1*/       __m128i search = _mm_cvtsi64_si128((uint64_t)tmp); // (2)p5 (MOVQ)
/*1*/       search = _mm_shuffle_epi8(search,_mm_setzero_si128()); // (1)p5 (VPSHUFB)

/*1*/       __m128i headers_set = _mm_loadu_si128((__m128i *)chain_ref);

            // Low half for bit8, up half for bit 9
/*1*/       uint32_t bit_res = ~(chain_ref[3] ^ src_bits); 

/*1*/       __m128i cmpres = _mm_cmpeq_epi8(headers_set,search); // (1)p01 (VCMPPS)
/*1*/       uint32_t items_mask = _mm_movemask_epi8(cmpres) & bit_res & (bit_res >> 16) & 0x7FF;

/*1*/       hstate_w->next_chain_ref = (uint32_t *)&ht->table[(int64_t)chain_ref[15] * CACHE_LINE_SIZE];
            hstate_w->chain_cont = chain_ref[15] ? 1 : 0;
/*1*/       hstate_w->items_mask = items_mask << 4; // Shifting by 4 to start of refs in table cacheline to remove 3 component addressing

/*1*/       int wsnum = (items_mask ? 2:0) + hstate_w->chain_cont;
            FPStateParams hps = hstate_w->sp;
/*1*/       worksets[wsnum]->indexes[worksets[wsnum]->last] = hps.offset;
/*1*/       worksets[wsnum]->last += hps.inc;

   /*2*/    FProcessState * restrict dstate_w = dstates_w[i]; // (4)p23 (MOV)
   /*2*/    __m256i *kb = (__m256i *)dstate_w->key_buf,*dr = (__m256i *)dstate_w->cur_data_ref,*ot = (__m256i *)output; // No op
   /*2*/    __m256i key0 = _mm256_loadu_si256(kb);  // (4)p23 (VMOVDQU)
   /*2*/    __m256i data0 = _mm256_loadu_si256(dr); // (4)p23 (VMOVDQU)
   /*2*/    int ks = dstate_w->key_size;

   /*2*/    uint32_t cmpmask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(key0,data0)); // (1)p01 (VCMPPS) -> (3)p0 (VPMOVMSKB) 

   /*2*/    int bnum = _tzcnt_u32(~cmpmask); // (1)p0156 -> (3)p1 (TZCNT)
   /*2*/    _mm256_storeu_si256((__m256i *)ot,key0);

   /*2*/    while (ks > 32)
   /*2*/       {
   /*2*/       key0 = _mm256_loadu_si256(++kb),data0 = _mm256_loadu_si256(++dr);
   /*2*/       bnum = _tzcnt_u32(~_mm256_movemask_epi8(_mm256_cmpeq_epi8(key0,data0)));
   /*2*/       _mm256_storeu_si256(++ot,key0);
   /*2*/       ks -= 32;
   /*2*/       }

   /*2*/    dstate_w->items_mask = _blsr_u32(dstate_w->items_mask);
   /*2*/    int found = (bnum >= ks);

   /*2*/    output[dstate_w->key_size] = '\n';
   /*2*/    output += found * (dstate_w->key_size + 1);

   /*2*/    int wsnumd = (1 - found) * ((dstate_w->items_mask ? 2:0) + dstate_w->chain_cont);
            FPStateParams dps = dstate_w->sp;
   /*2*/    worksets[wsnumd]->indexes[worksets[wsnumd]->last] = dps.offset;
   /*2*/    worksets[wsnumd]->last += dps.inc;

            }

         _mm256_storeu_si256((__m256i *)&hstates_w[0],hstates_lo); 
         _mm256_storeu_si256((__m256i *)&dstates_w[0],dstates_lo);

         _mm256_storeu_si256((__m256i *)&hstates_w[4],hstates_hi); 
         _mm256_storeu_si256((__m256i *)&dstates_w[4],dstates_hi);
         }
      tm += get_nanotime() - t1;
      processed += empty.last - pel;
      }
   aligned_free(states);
 
   if (stat)
      {
      printf("Hashed : %.2f per cycle\n",(double)hcnt / cnum);
      printf("Headers: %.2f per cycle\n",(double)hdcnt / cnum);
      printf("Datas  : %.2f per cycle\n",(double)dcnt / cnum);
      printf("Cycles : %d\n",cnum);
//      printf("Modes  : %d %d %d, switches %d, loads %d\n",modecnt[0],modecnt[1],modecnt[2],modeswitches,loads);
      }

   *acycle = (double)tm / cnum;
   return (double)tm/processed;
   }

#endif