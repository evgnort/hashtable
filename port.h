# Copyright (C) Evgeniy Buevich

#ifndef _PORT_H
#define _PORT_H

#include <stdint.h>
#include <sys/types.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>

#define __builtin_popcount __popcnt

#define aligned_alloc(A,S) _aligned_malloc(S,A)
#define aligned_free(P) _aligned_free(P)

#define PART_64_256(A,B) A.m256i_u64[B]

#define read_call _read
#define close_call _close

#else
#include <unistd.h>

#define PART_64_256(A,B) A[B]

#define aligned_free(P) free(P)

#define read_call read
#define close_call close

#endif

int64_t get_nanotime(void);

size_t enable_large_pages(void);
void *alloc_large_pages(size_t size, size_t lp_size);
void free_large_pages(void *lp);

#endif

