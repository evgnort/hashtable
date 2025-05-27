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
   int diff = source->inputlen - source->inputpos;
   if (diff <= 0)
      longjmp(source->eod_exit,1);
   source->lastpart = _mm_loadu_si128((void *)&source->input[source->inputpos]);
   source->lp_size = (diff > 16) ? 16 : diff; 
   source->inputpos += source->lp_size;
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
   int bpos = __builtin_ctz(mask);
   mask >>= bpos; // Number of trailing not endlines in part;
   bpos += __builtin_ctz(~mask);
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

void reset_results(FSource *source,FProcessState *state)
   {
   state->col_num = 0;
   state->key_size = 0;
   state->key_pos = state->key_buf;
   state->value_pos = state->value_start;
   skip_to_next(source);
   }

// returns 1 if state is valid (key column present), 0 otherwise
int finalize_state(FProcessState *state)
   {
   if (!state->key_size)
      return 0;
   state->col_num++;
   while(state->col_num < state->pp->value_col_count)
      {
      state->col_num++;
      *(state->value_pos) = state->pp->delimiter;
      state->value_pos++;
      }
   state->value_size = state->value_pos - state->value_start;
   return 1;
   }

int finalize_eol_found(FSource *source,FProcessState *state)
   {
   skip_eols(source);
   state->col_num++;
   return finalize_state(state);
   }

#define skip_char(source)  if (--source->lp_size > 0) source->lastpart = _mm_shuffle_epi8(source->lastpart,shift_one); \
                           else get_next_part(source);

int process_row(FSource *source,FProcessState *state)
   {
   __m128i shift_one = _mm_loadu_si128((void *)&shifts[1]);
   __m128i characters = _mm_setr_epi8('\r','\n',state->pp->src_delimiter,'"',0,0,0,0,0,0,0,0,0,0,0,0);
   __m128i qset = _mm_set1_epi8('\"');
   __m256i zeropad = _mm256_setzero_si256();

   char **output;
   int idx,qmask,kc,kv;
   char sym;

   if (setjmp(source->eod_exit))
      { // End of data;

      if (state->col_num == state->pp->key_col_num)
         {
         state->key_size = state->key_pos - state->key_buf;
         _mm256_storeu_si256((__m256i  *)state->key_pos,zeropad);
         if (state->pp->value_col_mask & (1 << state->col_num))
            {
            memcpy(state->value_pos,state->key_buf,state->key_size);
            state->value_pos += state->key_size;
            }
         }

      return finalize_state(state);
      }

pr_next_column:

   kc = (state->col_num == state->pp->key_col_num) ? 1 : 0;
   kv = state->pp->value_col_mask & (1 << state->col_num);
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
         idx = __builtin_ctzl(qmask) + 1;
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
               if (sym != state->pp->src_delimiter)
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
      idx = __builtin_ctzl(qmask);
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
               state->key_size = state->key_pos - state->key_buf;
               _mm256_storeu_si256((__m256i *)state->key_pos,zeropad);
               if (kv)
                  {
                  memcpy(state->value_pos,state->key_buf,state->key_size);
                  state->value_pos += state->key_size;
                  }
               }
            if (finalize_eol_found(source,state))
               return 1;
            reset_results(source,state);
            goto pr_next_column;
         default: 
            if (sym != state->pp->src_delimiter)
               return reset_results(source,state),0;
            if (kc)
               {
               state->key_size = state->key_pos - state->key_buf;
               _mm256_storeu_si256((__m256i *)state->key_pos,zeropad);
               if (kv)
                  {
                  memcpy(state->value_pos,state->key_buf,state->key_size);
                  state->value_pos += state->key_size;
                  *output = state->value_pos;
                  }
               }
            state->col_num++;
            *(*output) = state->pp->delimiter;
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
            state->key_size = state->key_pos - state->key_buf;
            _mm256_storeu_si256((__m256i *)state->key_pos,zeropad);
            if (kv)
               {
               memcpy(state->value_pos,state->key_buf,state->key_size);
               state->value_pos += state->key_size;
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
            state->key_size = state->key_pos - state->key_buf;
            _mm256_storeu_si256((__m256i *)state->key_pos,zeropad);
            if (kv)
               {
               memcpy(state->value_pos,state->key_buf,state->key_size);
               state->value_pos += state->key_size;
               *output = state->value_pos;
               }
            }
         state->col_num++;
         *(*output) = state->pp->delimiter;
         (*output)++;
         skip_char(source);
      }
   goto pr_next_column;
   }