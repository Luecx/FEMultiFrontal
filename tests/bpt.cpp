#include "block_packed_triangle.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

int main() {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    for (int n : {1, 2, 7, 47, 48, 49, 96, 97, 127, 193, 257}) {
        mf::dense::BlockPackedTriangle c;
        c.assign_triangle(n);
        std::vector<double> ref(static_cast<size_t>(n)*(n+1)/2, 0.0);
        for (int i=0;i<n;++i) for(int j=0;j<=i;++j) {
            const double value=dist(rng);
            c.at(i,j)=value; ref[static_cast<size_t>(i)*(i+1)/2+j]=value;
        }
        for(int origin : {0, 1, 13, 47, 48, 49, 73}) {
            if(origin>=n) continue;
            for(int k : {1, 2, 4, 8, 17}) {
                const int active=n-origin;
                std::vector<std::vector<double>> values(k,std::vector<double>(active));
                std::vector<const double*> ptrs;
                for(int p=0;p<k;++p) {
                    for(double& v:values[p]) v=dist(rng);
                    ptrs.push_back(values[p].data());
                }
                auto a=c; auto b=ref;
                mf::dense::rank_k_subtract_bpt(active,k,ptrs.data(),a,origin);
                for(int i=origin;i<n;++i)for(int j=origin;j<=i;++j) {
                    double& v=b[static_cast<size_t>(i)*(i+1)/2+j];
                    for(int p=0;p<k;++p) v-=values[p][i-origin]*values[p][j-origin];
                }
                for(int i=0;i<n;++i)for(int j=0;j<=i;++j){
                    const double error=std::abs(a.at(i,j)-b[static_cast<size_t>(i)*(i+1)/2+j]);
                    if(error>1e-13) {
                        std::printf("failed n=%d origin=%d k=%d at(%d,%d) err=%g\n",n,origin,k,i,j,error);
                        return 1;
                    }
                }
            }
        }
    }
    // Real panel assembly pattern, including inherited nonzero origin.
    for(int n : {8, 49, 101}) for(int width : {1,2,4,8}) {
        if(n<width)continue;
        mf::dense::BlockPackedTriangle c; c.assign_triangle(n+3);
        for(int i=0;i<c.dimension();++i)for(int j=0;j<=i;++j)c.at(i,j)=0.01*i+0.002*j;
        for(int origin : {0,1,3}){
            if(origin+n>c.dimension()) continue;
            std::vector<std::vector<double>> columns(width);
            std::vector<double*> ptrs;
            for(int j=0;j<width;++j){columns[j].assign(n-j,0.0);ptrs.push_back(columns[j].data());}
            mf::dense::add_block_packed_panel(n,width,ptrs.data(),c,origin);
            for(int j=1;j<width;++j)for(int r=j;r<n;++r)
                if(std::abs(columns[j][r-j] - c.at(origin+r-1,origin+j-1))>1e-15)return 2;
        }
    }
    // Test child Extend-add with noncontiguous parent indices and inherited
    // origins. Reference uses ordinary lower row-packed storage.
    for (int parent_n : {55, 97, 145}) {
        mf::dense::BlockPackedTriangle parent;
        parent.assign_triangle(parent_n);
        std::vector<double> packed_parent(static_cast<size_t>(parent_n)*(parent_n+1)/2, 0.0);
        const int child_size=parent_n/2;
        mf::dense::BlockPackedTriangle child;
        child.assign_triangle(child_size+5);
        const int origin=5;
        for(int i=0;i<child.dimension();++i)for(int j=0;j<=i;++j)
            child.at(i,j)=dist(rng);
        std::vector<int> mapping(child_size);
        for(int i=0;i<child_size;++i) mapping[i]=i*2;
        std::vector<double> parent_column(parent_n+1,0.0);
        std::vector<double> expected_column(parent_n+1,0.0);
        for(int i=0;i<child_size;++i){
            int r=mapping[i];
            parent_column[r]+=child.at(origin+i,origin);
            expected_column[r]+=child.at(origin+i,origin);
            for(int j=1;j<=i;++j){
                parent.at(r-1,mapping[j]-1)+=child.at(origin+i,origin+j);
                packed_parent[static_cast<size_t>(r-1)*r/2+mapping[j]-1]
                    +=child.at(origin+i,origin+j);
            }
        }
        for(int i=0;i<parent_n;++i){
            if(std::abs(parent_column[i]-expected_column[i])>1e-14)return 3;
            for(int j=0;j<=i;++j){
                if(std::abs(parent.at(i,j)-packed_parent[static_cast<size_t>(i)*(i+1)/2+j])>1e-14)
                    return 4;
            }
        }
    }
    std::puts("BPT randomized tile/origin/panel/extend-add checks: PASS");
}
