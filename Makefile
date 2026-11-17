# FEMultiFrontal: standalone C++20 multifrontal sparse direct solver.
# Build: make -j4
# Tests: make -j4 check
# OpenMP: make -j4 WITH_OPENMP=1 all check
CXX      ?= g++
CPPFLAGS += -Isrc
CXXFLAGS ?= -O3 -march=native -DNDEBUG -std=c++20 -Wall -Wextra -Wpedantic
LDFLAGS  ?=
LDLIBS   ?=

# Keep serial and OpenMP objects separate so configuration changes never mix.
WITH_OPENMP ?= 0
VARIANT     := native
ifeq ($(WITH_OPENMP),1)
VARIANT  := $(VARIANT)-omp
CXXFLAGS += -fopenmp
LDFLAGS  += -fopenmp
endif
# Alternate contribution layout. BPT=1 replaces only the SPD factorization.
BPT ?= 0
ifeq ($(BPT),1)
VARIANT := $(VARIANT)-bpt
LIB_SOURCES := $(filter-out src/main.cpp src/factor.cpp,$(wildcard src/*.cpp))
else
LIB_SOURCES := $(filter-out src/main.cpp src/factor_bpt.cpp,$(wildcard src/*.cpp))
endif
BUILD_DIR := build/$(VARIANT)
LIB_OBJECTS := $(LIB_SOURCES:src/%.cpp=$(BUILD_DIR)/%.o)
TEST_OBJECTS := $(BUILD_DIR)/tests/check.o $(BUILD_DIR)/tests/robustness.o
OBJECTS := $(LIB_OBJECTS) $(BUILD_DIR)/main.o $(TEST_OBJECTS)

.PHONY: all check test clean
all: $(BUILD_DIR)/mfsolve

$(BUILD_DIR)/mfsolve: $(LIB_OBJECTS) $(BUILD_DIR)/main.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/tests/check: $(LIB_OBJECTS) $(BUILD_DIR)/tests/check.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/tests/robustness: $(LIB_OBJECTS) $(BUILD_DIR)/tests/robustness.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD_DIR)/tests/%.o: tests/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

check: $(BUILD_DIR)/tests/check $(BUILD_DIR)/tests/robustness
	./$(BUILD_DIR)/tests/check
	./$(BUILD_DIR)/tests/robustness

test: check

clean:
	$(RM) -r build

-include $(OBJECTS:.o=.d)

ifeq ($(BPT),1)
.PHONY: check-bpt
$(BUILD_DIR)/tests/bpt: $(LIB_OBJECTS) $(BUILD_DIR)/tests/bpt.o
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@
check: check-bpt
check-bpt: $(BUILD_DIR)/tests/bpt
	./$(BUILD_DIR)/tests/bpt
endif
