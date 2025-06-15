#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>

#include <sys/stat.h>

#include "hashtable.h"
#include "utils.h"

FILE *logFile;

void init_log(const char *filename)
   {
   logFile = fopen(filename,"w");
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
	if (!filename || !filename[0] || stat(filename, &st) != 0 || !S_ISREG(st.st_mode))
		return -1;
	return st.st_size;
	}

char *read_file(const char *fname, int size)
   {
   char *filedata = (char *)malloc(size + CACHE_LINE_SIZE);

   off_t rdd = 0;
   int fd = open(fname,O_RDONLY);
   while (rdd < size)
      rdd += read(fd,filedata + rdd,size - rdd);
   close(fd);
   filedata[size] = 0;
   return filedata;
   }