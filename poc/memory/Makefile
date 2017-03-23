CC = clang
CFLAGS += -Wall -Wextra -pedantic -std=c99
CFLAGS += -O0 -g
CFLAGS += -fsanitize=address

main_c: main.c
	$(CC) $(CFLAGS) $< -o $@

clean:; rm -rf main_c
