CC = gcc
CFLAGS = -O2 -Wall -Wextra
LIBS = -libverbs

all: rdma_write_server rdma_write_client

rdma_write_server: rdma_write_server.c rdma_common.h
	$(CC) $(CFLAGS) -o $@ rdma_write_server.c $(LIBS)

rdma_write_client: rdma_write_client.c rdma_common.h
	$(CC) $(CFLAGS) -o $@ rdma_write_client.c $(LIBS)

clean:
	rm -f rdma_write_server rdma_write_client

.PHONY: all clean
