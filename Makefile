cppflags=-std=c++20 -Wall -Wextra -Werror
files=$(shell find ./src -type f -name "*.cpp")
iflags=-I./include -I./src

main:
	clang++ -g $(cppflags) $(files) -o ./debug $(iflags)
	clang++ -O3 $(cppflags) $(files) -o ./p $(iflags)