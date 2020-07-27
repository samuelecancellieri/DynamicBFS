#pragma once

#include "HornetAlg.hpp"
#include <BufferPool.cuh>

namespace hornets_nest 
{

using vid_t = int;
using vert_t = int;
using dist_t = int;

// using HornetInit  = ::hornet::HornetInit<vid_t>;
// using HornetDynamicGraph = ::hornet::gpu::Hornet<vid_t>;
// using HornetStaticGraph = ::hornet::gpu::HornetStatic<vid_t>;
// using DynBFS = BfsTopDown2<HornetDynamicGraph>;
// using BfsTopDown2Static  = BfsTopDown2<HornetStaticGraph>;
using HornetGraph = ::hornet::gpu::Hornet<vid_t>;

class DynBFS : public StaticAlgorithm<HornetGraph> {
public:
    DynBFS(HornetGraph& hornet, HornetGraph& hornet_in);
    ~DynBFS();

    void reset()    override;
    void run()      override;
    void release()  override;
    bool validate() override;

    void runAlg(int alg=0);

    void set_parameters(vid_t source,int load_balancing=0);
    void print_check(vert_t* update_src, vert_t* update_dst, int batch_size);
    void batch_update_directed(vert_t* update_dst, int update_size);
    void batch_update_undirected(vert_t* update_src, vert_t* update_dst, int update_size);
    void run_dynamic();
    void run_check(vid_t source);

    dist_t getLevels(){return current_level;}

private:
    BufferPool pool;
    TwoLevelQueue<vid_t> queue;
    TwoLevelQueue<vid_t> queue2;
    load_balancing::BinarySearch load_balancing;
    load_balancing::LogarthimRadixBinning32 lrb_lb;
    int lb_mechansim;

    HornetGraph& inverted_graph;

    dist_t* d_distances   { nullptr };
    dist_t* old_distances   { nullptr };
    // vert_t* update_batch {nullptr};
    vid_t   bfs_source    { 0 };
    dist_t  current_level { 0 };
};

} // namespace hornets_nest