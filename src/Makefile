TARGET = iterativo
INCLUDE_DIR = ./

CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -I $(INCLUDE_DIR)

SOURCES = iterativo.cpp

OBJECTS = $(SOURCES:.cpp=.o)

# Regla de compilación y enlazado
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regla para limpiar los archivos objeto y el ejecutable
clean:
	rm -f $(OBJECTS) $(TARGET)
