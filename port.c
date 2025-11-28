# Copyright (C) Evgeniy Buevich

#include "port.h"

#ifdef _WIN32

int64_t get_nanotime(void)
   {
   static LARGE_INTEGER ticksPerSec = {0};
   LARGE_INTEGER ticks;

   if (!ticksPerSec.QuadPart) 
      {
      QueryPerformanceFrequency(&ticksPerSec);
      if (!ticksPerSec.QuadPart) 
         return errno = ENOTSUP,-1;
      }

   QueryPerformanceCounter(&ticks);
   return ticks.QuadPart * 1000000000 / ticksPerSec.QuadPart;
   }

size_t enable_large_pages(void)
   {
   HANDLE hToken = NULL;
   TOKEN_PRIVILEGES tp;
   if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY | TOKEN_ADJUST_PRIVILEGES, &hToken))
      return 0;

   tp.PrivilegeCount = 1;
   tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
   if (!LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid))
      return 0;   

   BOOL result = AdjustTokenPrivileges(hToken, FALSE, &tp, 0, (PTOKEN_PRIVILEGES)NULL, 0);
   DWORD error = GetLastError();
   if (!result || (error != ERROR_SUCCESS))
      return 0; 

   return GetLargePageMinimum();
   }

void *alloc_large_pages(size_t size, size_t lp_size)
   {
   size_t adj_size = (size / lp_size + ((size % lp_size) ? 1 : 0)) * lp_size;
   void *lp = VirtualAlloc(NULL,adj_size,MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,PAGE_READWRITE);
   return lp;
   }

void free_large_pages(void *lp)
   {
   VirtualFree(lp,0,MEM_RELEASE);
   }

#else
#include <time.h>

int64_t get_nanotime(void)
   {
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC,&t);
   return t.tv_sec * 1000000000 + t.tv_nsec;
   }

size_t enable_large_pages(void)
   {
   return 0;
   }

void *alloc_large_pages(size_t size, size_t lp_size)
   {
   return NULL;
   }

void free_large_pages(void *lp)
   {
   }

#endif // _WIN32
