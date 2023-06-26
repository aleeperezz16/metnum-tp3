TARGET = iterativo
CC = g++
CFLAGS = -Wall -Wextra

SOURCES = iterativo.cpp

# Archivos objeto generados
OBJECTS = $(SOURCES:.cpp=.o)

# Regla de compilación y enlazado
$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS)

# Regla de compilación de archivos fuente a archivos objeto
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Regla para limpiar los archivos objeto y el ejecutable
clean:
	rm -f $(OBJECTS) $(TARGET)
