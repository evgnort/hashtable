# Copyright (C) Evgeniy Buevich

#ifndef _UTILS_H
#define _UTILS_H

void init_log(const char *filename);
void formattedLog(char *format,...);
void close_log(void);

#ifdef LOGFILE
	#define INIT_LOG(A) init_log(A)
	#define LOG_RECORD(...) formattedLog(__VA_ARGS__)
	#define CLOSE_LOG close_log()
#else
	#define INIT_LOG
	#define LOG_RECORD(...)
	#define CLOSE_LOG
#endif

off_t file_size(const char *filename);
char *read_file(const char *fname, int size);

#endif