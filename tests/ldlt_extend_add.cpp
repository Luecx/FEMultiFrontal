/**
 * @file ldlt_extend_add.cpp
 * @brief Independent packed-triangle extend-add regression for SIMD runs.
 *
 * Checks arbitrary child-to-parent index gaps, inherited physical row origins,
 * SIMD tails, full contiguous spans, tiny child fronts and all untouched
 * entries against the original scalar scatter-add operation.
 */
#include "ldlt_extend_add.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::size_t index(int origin, int row, int column) {
    const std::size_t physical = static_cast<std::size_t>(origin+row);
    return physical*(physical+1)/2 + origin + column;
}

static void check(const std::string& label,
                  const std::vector<int>& child_map,
                  int source_origin,int target_origin,bool expected_enabled) {
    const int count=static_cast<int>(child_map.size());
    const int source_size=source_origin+count+9;
    const int target_size=target_origin+(count ? child_map.back()+1 : 0)+9;
    std::vector<double> child(
        static_cast<std::size_t>(source_size)*(source_size+1)/2,0.0);
    for(int r=0;r<count;++r) {
        for(int c=0;c<=r;++c) {
            child[index(source_origin,r,c)] =
                std::sin(0.117*r+0.23*c) *
                ((r+c)%2 ? -0.29 : 0.43);
        }
    }
    std::vector<double> expected(
        static_cast<std::size_t>(target_size)*(target_size+1)/2,1.25);
    std::vector<double> actual=expected;
    const auto plan=mf::dense::plan_segmented_extend_add(child_map);
    if(plan.enabled!=expected_enabled) {
        throw std::runtime_error(label+": wrong segmented-plan enable state");
    }
    // Repeated assembly confirms that every entry is accumulated in the
    // incumbent child and source-element order and preserves result bits.
    for(int repeat=0;repeat<2;++repeat) {
        for(int r=0;r<count;++r) {
            const double* from=child.data()+index(source_origin,r,0);
            double* to=actual.data()+index(target_origin,child_map[r],0);
            if(plan.enabled) {
                mf::dense::extend_add_row_segmented(to,from,r,plan);
            } else {
                for(int c=0;c<=r;++c) { to[child_map[c]]+=from[c]; }
            }
            double* oracle=expected.data()+index(target_origin,child_map[r],0);
            for(int c=0;c<=r;++c) { oracle[child_map[c]]+=from[c]; }
        }
    }
    if(actual!=expected) {
        for(std::size_t i=0;i<actual.size();++i) {
            if(actual[i]!=expected[i]) {
                throw std::runtime_error(
                    label+": packed result differs from scalar baseline at "+
                    std::to_string(i));
            }
        }
        throw std::runtime_error(label+": wrong result size");
    }
}

int main() {
    try {
        for(int parent_origin : {0,7,29}) {
            for(int child_origin : {0,11}) {
                check("empty",{},child_origin,parent_origin,false);
                check("single",{4},child_origin,parent_origin,false);
                std::vector<int> contiguous(128);
                for(int i=0;i<128;++i) { contiguous[i]=i+17; }
                check("contiguous",contiguous,child_origin,parent_origin,true);
                std::vector<int> split(96);
                for(int i=0;i<96;++i) { split[i]=i+3+(i>=48 ? 9 : 0); }
                check("split",split,child_origin,parent_origin,true);
                std::vector<int> sparse(128);
                for(int i=0;i<128;++i) { sparse[i]=i*3+5; }
                check("sparse",sparse,child_origin,parent_origin,false);
                std::vector<int> staggered(95);
                for(int i=0;i<95;++i) { staggered[i]=i+7+i/7; }
                check("staggered",staggered,child_origin,parent_origin,false);
                std::vector<int> tail(35);
                for(int i=0;i<35;++i) { tail[i]=i+6; }
                check("simd-tail",tail,child_origin,parent_origin,true);
            }
        }
        std::cout << "PASS: 42 packed child-to-parent extend-add regressions\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
