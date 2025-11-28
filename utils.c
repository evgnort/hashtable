#include "port.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <fcntl.h>
# Copyright (C) Evgeniy Buevich

#include <sys/stat.h>

#ifdef _WIN32
#include <io.h>
#endif

#include "hashtable.h"
#include "utils.h"

FILE *logFile;

void init_log(const char *filename)
   {
#ifdef _WIN32
   fopen_s(&logFile,filename,"w");
#else
   logFile = fopen(filename,"w");
#endif
   }

void formattedLog(char *format,...)
   {
	if (!logFile) return;
	va_list args;

   va_start (args, format);
	vfprintf (logFile,format, args);
	fprintf (logFile,"\n");
	va_end (args);
   fflush(logFile);
   }

void close_log(void)
   {
   if (logFile)
      fclose(logFile);
   }

off_t file_size(const char *filename)
	{
	struct stat st;
	if (!filename || !filename[0] || stat(filename, &st) != 0)
		return -1;
	return st.st_size;
	}

char *read_file(const char *fname, int size)
   {
   char *filedata = (char *)malloc(size + CACHE_LINE_SIZE);
   if (!filedata)
      return NULL;

   off_t rdd = 0,readed;
   int fd;
#ifdef _WIN32
   _sopen_s(&fd,fname,O_RDONLY,_SH_DENYNO,_S_IREAD);
#else
   fd = open(fname,O_RDONLY,S_IREAD);
#endif
   while (rdd < size && (readed = read_call(fd,filedata + rdd,size - rdd) > 0))
      rdd += readed;
   close_call(fd);
   filedata[size++] = '\n';
   filedata[size] = 0;
   return filedata;
   }