# File names
EXEC= 
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard ../*/*.h)
OBJECTS = $(SOURCES:.cpp=.o)
#MAINS = $(shell ls *.cpp | egrep -v "^main")

default: $(OBJECTS)

# To obtain object files
%.o: %.cpp $(HEADERS)
	$(CC) -c $(CC_FLAGS) $(IFLAGS) $< -o $@

# To remove generated files
clean:
	rm -f $(EXEC) $(OBJECTS)
