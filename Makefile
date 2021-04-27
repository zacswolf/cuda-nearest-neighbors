# Compiler
CC = g++

# Compiler flags:
CFLAGS = -std=c++17 -Wall

# The build target 
TARGET = runner

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cpp matrix.cpp csv_data_loader.cpp nearest_neighbor_sequential.cpp

clean:
	$(RM) $(TARGET)
