# Copyright (C) Evgeniy Buevich
 
#include "port.h"

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>

#include <string.h>
#include <stdio.h>
#include <fcntl.h>

#include "csvparse.h"
#include "hashtable.h"

uint8_t shifts[32] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}; 

static inline void get_next_part(FSource *source)
   {
   int diff = (int)(source->input_end - source->input_pos);
   if (diff <= 0)
      longjmp(source->eod_exit,1);
   source->lastpart = _mm_loadu_si128((void *)source->input_pos);
   source->lp_size = (diff > 16) ? 16 : diff; 
   source->input_pos += source->lp_size;
   }

static inline void skip_eols(FSource *source)
   {
   __m128i characters = _mm_setr_epi8('\r','\n',0,0,0,0,0,0,0,0,0,0,0,0,0,0);
   int idx;

   source->lp_size -= (idx = _mm_cmpistri(characters,source->lastpart,_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_NEGATIVE_POLARITY));
   while (!source->lp_size)
      {
      get_next_part(source);
      source->lp_size -= (idx = _mm_cmpistri(characters,source->lastpart,_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_NEGATIVE_POLARITY));
      }
   source->lastpart = _mm_shuffle_epi8(source->lastpart,_mm_loadu_si128((void *)&shifts[idx]));
   }


void skip_to_next(FSource *source)
   {
   __m128i maskset;
   __m128i characters = _mm_setr_epi8('\r','\n',0,0,0,0,0,0,0,0,0,0,0,0,0,0);
   uint16_t mask;

   if (!source->lp_size)
      get_next_part(source);

skip_to_next_1:

   maskset = _mm_cmpestrm(characters,3,source->lastpart,16,_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_POSITIVE_POLARITY);
   mask = _mm_extract_epi16(maskset,0);
   if (!mask)
      { 
      get_next_part(source);
      goto skip_to_next_1; 
      }
   
   mask &= (1 << source->lp_size) - 1; // Resetting bits of processed part
   int bpos = _tzcnt_u32(mask);
   mask >>= bpos; // Number of trailing not endlines in part;
   bpos += _tzcnt_u32(~mask);
   if (source->lp_size > bpos)
      {
      source->lastpart = _mm_shuffle_epi8(source->lastpart,_mm_loadu_si128((void *)&shifts[bpos]));
      source->lp_size -= bpos;
      }

   do
      {
      get_next_part(source);
      source->lp_size -= _mm_cmpistri(characters,source->lastpart,_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_NEGATIVE_POLARITY);
      }
   while (!source->lp_size);

   source->lastpart = _mm_shuffle_epi8(source->lastpart,_mm_loadu_si128((void *)&shifts[16 - source->lp_size]));
   }

void reset_results(FSource *source,FParseState *state)
   {
   state->col_num = 0;
   *state->key_size_ref = 0;
   state->key_pos = state->key_buf;
   state->value_pos = state->value_start;
   skip_to_next(source);
   }

// returns 1 if state is valid (key column present), 0 otherwise
int finalize_state(FParseState *state, FParseParams *pp)
   {
   if (!*state->key_size_ref)
      return 0;
   state->col_num++;
   while(state->col_num < pp->value_col_count)
      {
      state->col_num++;
      *(state->value_pos) = pp->delimiter;
      state->value_pos++;
      }
   *state->value_size_ref = (int)(state->value_pos - state->value_start);

   return 1;
   }

int finalize_eol_found(FSource *source,FParseState *state)
   {
   skip_eols(source);
   state->col_num++;
   return finalize_state(state,source->pp);
   }

void init_source(FSource *source,char *input,int len)
   {
   source->input_start = input;
   source->input_end = input + len;
   source->input_pos = input;
   source->lastpart = _mm_setzero_si128();
   source->lp_size = 0;
   get_next_part(source);
   }

#define skip_char(source)  if (--source->lp_size > 0) source->lastpart = _mm_shuffle_epi8(source->lastpart,shift_one); \
                           else get_next_part(source);

int process_row(FSource *source,FParseState *state)
   {
   __m128i shift_one = _mm_loadu_si128((void *)&shifts[1]);
   __m128i characters = _mm_setr_epi8('\r','\n',source->pp->src_delimiter,'"',0,0,0,0,0,0,0,0,0,0,0,0);
   __m128i qset = _mm_set1_epi8('\"');

   char **output;
   int idx,qmask,kc,kv;
   char sym;

   if (setjmp(source->eod_exit))
      { // End of data;

      if (state->col_num == source->pp->key_col_num)
         {
         *state->key_size_ref = (int)(state->key_pos - state->key_buf);
         *((uint64_t *)state->key_pos) = 0LL;
         *((uint32_t *)&state->key_pos[8]) = 0;
//         _mm256_storeu_si256((__m256i  *)state->key_pos,zeropad);
         if (source->pp->value_col_mask & (1 << state->col_num))
            {
            memcpy(state->value_pos,state->key_buf,*state->key_size_ref);
            state->value_pos += *state->key_size_ref;
            }
         }

      return finalize_state(state,source->pp);
      }

pr_next_column:

   kc = (state->col_num == source->pp->key_col_num) ? 1 : 0;
   kv = source->pp->value_col_mask & (1 << state->col_num);
   output = kc ? &state->key_pos : &state->value_pos;

   if (_mm_extract_epi8(source->lastpart,0) == '"')
      {
      skip_char(source);
      if (!output)
         {
pr_quoted_skip_next_part:
         qmask = _mm_movemask_epi8(_mm_cmpeq_epi8(source->lastpart,qset)) & ((1 << source->lp_size) - 1);
         if (!qmask)
            {
            get_next_part(source);
            goto pr_quoted_skip_next_part;
            }
         idx = _tzcnt_u32(qmask) + 1;
         if ((source->lp_size -= idx) > 0)
            source->lastpart = _mm_shuffle_epi8(source->lastpart,_mm_loadu_si128((void *)&shifts[idx]));
         else
            get_next_part(source);
         char sym = _mm_extract_epi8(source->lastpart,0);
         switch (sym)
            {
            case '\r': case '\n':
               if (finalize_eol_found(source,state))
                  return 1;
               reset_results(source,state);
               goto pr_next_column;
            case '"':
               skip_char(source);
               goto pr_quoted_skip_next_part;
            default: 
               if (sym != source->pp->src_delimiter)
                  return reset_results(source,state),0;
               state->col_num++;
               skip_char(source);
            }
         goto pr_next_column;
         }

pr_quoted_next_part_mask:
      qmask = _mm_movemask_epi8(_mm_cmpeq_epi8(source->lastpart,qset)) & ((1 << source->lp_size) - 1);
pr_quoted_next_part:
      _mm_storeu_si128((__m128i *)(*output),source->lastpart);

      if (!qmask)
         {
         *output += source->lp_size;
         get_next_part(source);
         goto pr_quoted_next_part_mask;
         }
      idx = _tzcnt_u32(qmask);
      *output += idx;
      idx++;
      if ((source->lp_size -= idx) > 0)
         { // have some data after quote
         qmask >>= idx;
         if (qmask & 1)
            {
            (*output)++; // Leave one stored quote in ouput
            if (--source->lp_size) 
               {
               source->lastpart = _mm_shuffle_epi8(source->lastpart,_mm_loadu_si128((void *)&shifts[idx + 1]));
               qmask >>= 1;
               goto pr_quoted_next_part;
               }
            get_next_part(source);
            goto pr_quoted_next_part_mask;
            }
         source->lastpart = _mm_shuffle_epi8(source->lastpart,_mm_loadu_si128((void *)&shifts[idx]));
         goto switch_sym_after_quote;
         }
      get_next_part(source);
      qmask = _mm_movemask_epi8(_mm_cmpeq_epi8(source->lastpart,qset)) & ((1 << source->lp_size) - 1);
      if (qmask & 1)
         {
         (*output)++; // Leave one stored quote in ouput
         if (!--source->lp_size)   // Incomplete portion loaded, eod
            longjmp(source->eod_exit,1);
         source->lastpart = _mm_shuffle_epi8(source->lastpart,shift_one);
         qmask >>= 1;
         goto pr_quoted_next_part;
         }
switch_sym_after_quote:
      sym = _mm_extract_epi8(source->lastpart,0);
      switch (sym)
         {
         case '\r': case '\n':
            if (kc)
               {
               *state->key_size_ref = (int)(state->key_pos - state->key_buf);
               *((uint64_t *)state->key_pos) = 0LL;
               *((uint32_t *)&state->key_pos[8]) = 0;
//               _mm256_storeu_si256((__m256i *)state->key_pos,zeropad);
               if (kv)
                  {
                  memcpy(state->value_pos,state->key_buf,*state->key_size_ref);
                  state->value_pos += *state->key_size_ref;
                  }
               }
            if (finalize_eol_found(source,state))
               return 1;
            reset_results(source,state);
            goto pr_next_column;
         default: 
            if (sym != source->pp->src_delimiter)
               return reset_results(source,state),0;
            if (kc)
               {
               *state->key_size_ref = (int)(state->key_pos - state->key_buf);
               *((uint64_t *)state->key_pos) = 0LL;
               *((uint32_t *)&state->key_pos[8]) = 0;
//               _mm256_storeu_si256((__m256i *)state->key_pos,zeropad);
               if (kv)
                  {
                  memcpy(state->value_pos,state->key_buf,*state->key_size_ref);
                  state->value_pos += *state->key_size_ref;
                  output = &state->value_pos;
                  }
               }
            state->col_num++;
            *(*output) = source->pp->delimiter;
            (*output)++;
            skip_char(source);
         }
      goto pr_next_column;
      }

   if (!output)
      {
pr_skip_next_part:
      idx = _mm_cmpestri(characters,4,source->lastpart,source->lp_size,_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_POSITIVE_POLARITY);
      if ((source->lp_size -= idx) <= 0) 
         {
         get_next_part(source);
         goto pr_skip_next_part;
         }
      source->lastpart = _mm_shuffle_epi8(source->lastpart,_mm_loadu_si128((void *)&shifts[idx]));

      switch (_mm_extract_epi8(source->lastpart,0))
         {
         case '\n': case '\r':
            if (finalize_eol_found(source,state))
               return 1;
         case '"': 
            reset_results(source,state); 
            goto pr_next_column;
         default: // Only delimiter
            state->col_num++;
            skip_char(source);
         }
      goto pr_next_column;
      }

pr_next_part:
   _mm_storeu_si128((__m128i *)(*output),source->lastpart);
   idx = _mm_cmpestri(characters,4,source->lastpart,source->lp_size,_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_POSITIVE_POLARITY);
   if (source->lp_size <= idx) 
      {
      *output += source->lp_size;
      get_next_part(source);
      goto pr_next_part;
      }
   source->lp_size -= idx;
   *output += idx;
   source->lastpart = _mm_shuffle_epi8(source->lastpart,_mm_loadu_si128((void *)&shifts[idx]));

   switch (_mm_extract_epi8(source->lastpart,0))
      {
      case '\n': case '\r':
         if (kc)
            {
            *state->key_size_ref = (int)(state->key_pos - state->key_buf);
            *((uint64_t *)state->key_pos) = 0LL;
            *((uint32_t *)&state->key_pos[8]) = 0;
//            _mm256_storeu_si256((__m256i *)state->key_pos,zeropad);
            if (kv)
               {
               memcpy(state->value_pos,state->key_buf,*state->key_size_ref);
               state->value_pos += *state->key_size_ref;
               }
            }
         if (finalize_eol_found(source,state))
            return 1;
      case '"': 
         reset_results(source,state); 
         goto pr_next_column;
      default: // Only delimiter
         if (kc)
            {
            *state->key_size_ref = (int)(state->key_pos - state->key_buf);
            *((uint64_t *)state->key_pos) = 0LL;
            *((uint32_t *)&state->key_pos[8]) = 0;
//            _mm256_storeu_si256((__m256i *)state->key_pos,zeropad);
            if (kv)
               {
               memcpy(state->value_pos,state->key_buf,*state->key_size_ref);
               state->value_pos += *state->key_size_ref;
               output = &state->value_pos;
               }
            }
         state->col_num++;
         *(*output) = source->pp->delimiter;
         (*output)++;
         skip_char(source);
      }
   goto pr_next_column;
   }

int process_csv_plain(FSource *source,FProcessState *states,FNormalRqSet *empty_set,FNormalRqSet *loaded_set)
// Без перестановки столбцов, без замены разделителя, столбцы значения подряд, столбец ключа первый
// 
   {
   __m256i quote_pat = _mm256_set1_epi8('"');
   __m256i delim_pat = _mm256_set1_epi8(source->pp->src_delimiter);
   __m256i eol_pat = _mm256_set1_epi8('\n');
   __m256i lf_pat = _mm256_set1_epi8('\r');

   int colnum = 0;
   FProcessState *state;

   int keysize = 0;
   int char_pos = 0;

   if (empty_set->first >= empty_set->last)
         return 0;

   state = nrs_get_first(states,empty_set);
   reset_state_search_data(state);
   char *output_pos = state->value_start;
   keysize = 0;
   char *field_start = output_pos;
      
process_csv_plain_repeat:
   __m256i part = _mm256_loadu_si256((__m256i *)source->input_pos);
   _mm256_storeu_si256((__m256i *)output_pos,part);

   __m256i quotemaskv = _mm256_cmpeq_epi8(part,quote_pat);
   __m256i delimmaskv = _mm256_cmpeq_epi8(part,delim_pat);
   __m256i eolmaskv = _mm256_cmpeq_epi8(part,eol_pat);
   __m256i lfmaskv = _mm256_cmpeq_epi8(part,lf_pat);

   uint32_t quotemask = _mm256_movemask_epi8(quotemaskv);
   uint32_t delimmask = _mm256_movemask_epi8(delimmaskv);

   uint32_t eolmask = _mm256_movemask_epi8(eolmaskv) | _mm256_movemask_epi8(lfmaskv);

   uint32_t cmpmask,combined = quotemask | delimmask | eolmask;

   while (combined)
      {
      char_pos = _tzcnt_u32(combined);
      cmpmask = _blsi_u32(combined);
      combined &= ~cmpmask; 

      if (quotemask & cmpmask)
         { // Processing quoted field
         if (output_pos + char_pos != field_start)
            {
            keysize = 0;
            goto process_csv_skip_to_eol; // Quote in unquoted field - skipping string
            }

         output_pos += char_pos;
         // Looking for next quote
         source->input_pos += char_pos + 1;

         do
            {
            do
               {
               part = _mm256_loadu_si256((__m256i *)source->input_pos);
               _mm256_storeu_si256((__m256i *)output_pos,part);

               quotemaskv = _mm256_cmpeq_epi8(part,quote_pat);
               quotemask = _mm256_movemask_epi8(quotemaskv);
               }
            while(!quotemask);
            char_pos = _tzcnt_u32(quotemask);
            output_pos += char_pos;
            source->input_pos += char_pos + 1;
            }
         while (*source->input_pos == '"');
         if (*source->input_pos == '\r')
            goto process_csv_next_eol_found;

         if (*source->input_pos == source->pp->src_delimiter)
            goto process_csv_next_column;

         keysize = 0;
         goto process_csv_skip_to_eol; // single quote is not at the column end
         }

      keysize += (!colnum) * (int)(output_pos + char_pos - field_start);
      if (eolmask & cmpmask)
         {
process_csv_next_eol_found:
         output_pos += char_pos;
         source->input_pos += char_pos;
         while (++colnum < source->pp->value_col_count)
            *output_pos++ = source->pp->delimiter;
         goto process_csv_skip_eol;
         }

process_csv_next_column:
      // Field divider
      if (++colnum >= source->pp->value_col_count)
         goto process_csv_skip_to_eol;            
      field_start = output_pos + char_pos + 1;
      }
   output_pos += 32;
   source->input_pos += 32;
   goto process_csv_plain_repeat;

process_csv_skip_to_eol:
// Skipping rest of current string
   eolmask &= combined;
   while (!eolmask)
      {
      source->input_pos += 32;
      part = _mm256_loadu_si256((__m256i *)source->input_pos);
      eolmaskv = _mm256_cmpeq_epi8(part,eol_pat);
      lfmaskv = _mm256_cmpeq_epi8(part,lf_pat);
      eolmask = _mm256_movemask_epi8(eolmaskv) | _mm256_movemask_epi8(lfmaskv);
      }
   char_pos = _tzcnt_u32(eolmask);

process_csv_skip_eol:
 // Skiping all eols in source
   eolmask = ~(eolmask >> char_pos);
   int eols_count = _tzcnt_u32(eolmask);
   if (char_pos + eols_count >= 32)
      {
      do
         {
         source->input_pos += 32;
         part = _mm256_loadu_si256((__m256i *)source->input_pos);
         eolmaskv = _mm256_cmpeq_epi8(part,eol_pat);
         lfmaskv = _mm256_cmpeq_epi8(part,lf_pat);

         eolmask = _mm256_movemask_epi8(eolmaskv) | _mm256_movemask_epi8(lfmaskv);
         eols_count = _tzcnt_u32(~eolmask);
         }
      while (eols_count == 32);
      }

 // Finalization of current state
   if (keysize)
      {
      __m256i key0 = _mm256_loadu_si256((__m256i *)state->value_start);
      __m256i key1 = _mm256_loadu_si256((__m256i *)&state->value_start[32]);
      _mm256_storeu_si256((__m256i *)&state->key_buf[0],key0);
      _mm256_storeu_si256((__m256i *)&state->key_buf[32],key1);
      _mm_storeu_si128((__m128i *)&state->key_buf[keysize],_mm_setzero_si128());

      state->key_size = keysize;
      state->value_size = (int)(output_pos - state->value_start);
      loaded_set->indexes[loaded_set->last] = state->sp.offset;
      loaded_set->last++;
      loaded_set->count++;

      if (empty_set->first >= empty_set->last)
         return 0;

      state = nrs_get_first(states,empty_set);
      }

   source->input_pos += eols_count;
   output_pos = field_start = state->value_start;
   colnum = keysize = 0;

   if (source->input_pos < source->input_end)
      goto process_csv_plain_repeat;
   return 0;
   }


