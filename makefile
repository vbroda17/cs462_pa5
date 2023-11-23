# This basically just does 
CC := nvcc
CFLAGS := -std=c++11

SRCS := mean-variance.cu
OBJS := mean-variance

all: $(OBJS)

$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(OBJS)
