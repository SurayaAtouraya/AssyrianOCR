CC = g++
CFLAGS = -g -Wall
SRCS = ./src/main.cpp
PROG = test

OPENCV = `pkg-config opencv4 --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
