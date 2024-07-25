## Include flags
IFLAGS=-I../AccretionFlows -I../Include -I../Metrics -I../Misc -I../RadiativeTransfer -I../Rays -I../StopConditions

## Linker flags
LFLAGS=

# File names
EXEC = vrt2.la
SOURCES = $(wildcard *.cpp)
HEADERS = $(wildcard ../*/*.h)
OBJECTS = $(SOURCES:.cpp=.o)

default: $(OBJECTS)

# To obtain object files
%.o: %.cpp $(HEADERS)
	$(CC) -c $(CC_FLAGS) $(IFLAGS) $< -o $@

# To remove generated files
clean:
	rm -f $(EXEC) $(OBJECTS)