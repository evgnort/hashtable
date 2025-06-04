#ifndef _UTILS_H
#define _UTILS_H

#ifdef LOGFILE
	#define INIT_LOG(A) init_log(A)
	#define LOG_RECORD(...) formattedLog(__VA_ARGS__)
	#define CLOSE_LOG close_
#else
	#define INIT_LOG
	#define LOG_RECORD(...)
	#define CLOSE_LOG
#endif

off_t file_size(const char *filename);
char *read_file(const char *fname, int size);

#endif