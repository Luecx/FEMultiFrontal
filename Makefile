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
BUILD_DIR := build/$(VARIANT)

# The experimental LDLT fast path calls the SPD Factor implementation.
# Build a single relocatable pivot_factor.o with those dependencies so existing
# LDLT benchmark harnesses can keep linking their original explicit object list.
# Exclude the bundled sources from LIB_OBJECTS to avoid duplicate definitions.
PIVOT_BUNDLE_SOURCES := src/pivot_factor.cpp src/factor.cpp src/dense.cpp src/dense_panel.cpp src/dense_schur.cpp
PIVOT_BUNDLE_OBJECTS := $(PIVOT_BUNDLE_SOURCES:src/%.cpp=$(BUILD_DIR)/%.bundle.o)
LIB_SOURCES := $(filter-out src/main.cpp $(PIVOT_BUNDLE_SOURCES),$(wildcard src/*.cpp))
LIB_OBJECTS := $(LIB_SOURCES:src/%.cpp=$(BUILD_DIR)/%.o) $(BUILD_DIR)/pivot_factor.o
TEST_OBJECTS := $(BUILD_DIR)/tests/check.o $(BUILD_DIR)/tests/robustness.o $(BUILD_DIR)/tests/pivot_update.o
OBJECTS := $(LIB_OBJECTS) $(BUILD_DIR)/main.o $(TEST_OBJECTS) $(PIVOT_BUNDLE_OBJECTS)

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

$(BUILD_DIR)/pivot_factor.o: $(PIVOT_BUNDLE_OBJECTS)
	$(CXX) -r $^ -o $@

$(BUILD_DIR)/%.bundle.o: src/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

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
