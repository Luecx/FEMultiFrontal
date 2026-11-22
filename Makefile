# FEMultiFrontal: standalone C++20 multifrontal sparse direct solver.
# Build: make -j4
# Tests: make -j4 check
# OpenMP: make -j4 WITH_OPENMP=1 all check
CXX      ?= g++
CPPFLAGS += -Isrc
CXXFLAGS ?= -O3 -march=native -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic
LDFLAGS  ?=
LDLIBS   ?=

# Keep serial, OpenMP and optional METIS objects separate by build variant.
WITH_METIS ?= 0
WITH_OPENMP ?= 0
VARIANT     := native
ifeq ($(WITH_OPENMP),1)
VARIANT  := $(VARIANT)-omp
CXXFLAGS += -fopenmp
LDFLAGS  += -fopenmp
endif
ifeq ($(WITH_METIS),1)
VARIANT := $(VARIANT)-metis
CPPFLAGS += -DMF_WITH_METIS=1
ifneq ($(CONDA_PREFIX),)
CPPFLAGS += -I$(CONDA_PREFIX)/include
LDFLAGS += -L$(CONDA_PREFIX)/lib -Wl,-rpath,$(CONDA_PREFIX)/lib
endif
LDLIBS += -lmetis
endif
BUILD_DIR := build/$(VARIANT)

LIB_SOURCES := $(filter-out src/main.cpp,$(wildcard src/*.cpp))
LIB_OBJECTS := $(LIB_SOURCES:src/%.cpp=$(BUILD_DIR)/%.o)
TEST_OBJECTS := $(BUILD_DIR)/tests/check.o $(BUILD_DIR)/tests/robustness.o $(BUILD_DIR)/tests/pivot_update.o
OBJECTS := $(LIB_OBJECTS) $(BUILD_DIR)/main.o $(TEST_OBJECTS)

.PHONY: all check test clean
all: $(BUILD_DIR)/mfsolve

$(BUILD_DIR)/mfsolve: $(LIB_OBJECTS) $(BUILD_DIR)/main.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/tests/check: $(LIB_OBJECTS) $(BUILD_DIR)/tests/check.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/tests/robustness: $(LIB_OBJECTS) $(BUILD_DIR)/tests/robustness.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/tests/pivot_update: $(LIB_OBJECTS) $(BUILD_DIR)/tests/pivot_update.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/tests/%.o: tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

check: $(BUILD_DIR)/tests/check $(BUILD_DIR)/tests/robustness $(BUILD_DIR)/tests/pivot_update
	./$(BUILD_DIR)/tests/check
	./$(BUILD_DIR)/tests/robustness
	./$(BUILD_DIR)/tests/pivot_update

test: check

clean:
	$(RM) -r build

-include $(OBJECTS:.o=.d)
