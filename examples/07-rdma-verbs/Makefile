CC = gcc
CFLAGS = -O2 -Wall -Wextra
LIBS = -libverbs

all: server client

server: server.c common.h
	$(CC) $(CFLAGS) -o server server.c $(LIBS)

client: client.c common.h
	$(CC) $(CFLAGS) -o client client.c $(LIBS)

clean:
	rm -f server client

.PHONY: all clean
