/**
* @brief Breadth-first Search Top-Down test program
* @file
*/
#include "Dynamic/BFS/DynBFS.cuh"
#include "StandardAPI.hpp"
#include "Static/BreadthFirstSearch/TopDown2.cuh"
#include "Util/BatchFunctions.hpp"
#include "Util/RandomGraphData.cuh"
#include <Core/Static/Static.cuh>
#include <Device/Util/CudaUtil.cuh>     //xlib::deviceInfo
#include <Graph/GraphStd.hpp>
#include <Hornet.hpp>
#include <Host/FileUtil.hpp>            //xlib::extract_filepath_noextension
#include <StandardAPI.hpp>
#include <Util/CommandLineParam.hpp>
#include <algorithm>                    //std:.generate
#include <chrono>                       //std::chrono
#include <ctime>
#include <cuda_profiler_api.h> //--profile-from-start off
#include <random>                       //std::mt19937_64

template <typename HornetGraph, typename BFS>
int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    using namespace graph;
    using HornetGPU = hornet::gpu::Hornet<vert_t>;
    using UpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
    // using UpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::DEVICE>;
    using BatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1000000000000000000);

    graph::GraphStd<vid_t, eoff_t> graph(ENABLE_INGOING);
    CommandLineParam cmd(graph, argc, argv,false);

    HornetInit hornet_init_out(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    HornetInit hornet_init_in(graph.nV(), graph.nE(), graph.csr_in_offsets(), graph.csr_in_edges());

    Timer<DEVICE> TM;
    HornetGraph hornet_graph_out(hornet_init_out);
    HornetGraph hornet_graph_in(hornet_init_in);

    DynBFS dynamic_bfs(hornet_graph_out, hornet_graph_in);

    vid_t root = graph.max_out_degree_id();
    int alg = 0;
    int batch_size = 0;

    if (argc >= 3)
        root = atoi(argv[2]);
    if(argc >= 4)
        batch_size = atoi(argv[3]);

    dynamic_bfs.reset();
    dynamic_bfs.set_parameters(root,alg);

    std::cout << "My root is " << root << std::endl;

    TM.start();
    dynamic_bfs.run();
    TM.stop();
    TM.print("Static BFS");

    std::cout << "Number of levels is : " << dynamic_bfs.getLevels() << std::endl;

    std::cout<<"prima size "<<hornet_graph_out.nE()<<"\n";

    vert_t* batch_src = new vert_t[batch_size]();
    vert_t* batch_dst = new vert_t[batch_size]();

    generateBatch(graph, batch_size, batch_src, batch_dst, BatchGenType::INSERT, batch_gen_property::UNIQUE);

    UpdatePtr ptr(batch_size, batch_src, batch_dst);
    BatchUpdate batch_update_src_to_dst(ptr);

    UpdatePtr ptr2(batch_size, batch_dst, batch_src);
    BatchUpdate batch_update_dst_to_src(ptr2);

    if(graph.is_directed())
    {
        std::cout<<"entro directed "<<"\n";
        hornet_graph_in.insert(batch_update_dst_to_src);
        hornet_graph_out.insert(batch_update_src_to_dst);

    }
    else
    {
        hornet_graph_out.insert(batch_update_src_to_dst);
        hornet_graph_out.insert(batch_update_dst_to_src);
    }

    // batch_update_src_to_dst.print();
    // batch_update_dst_to_src.print();

    std::cout<<"dopo size "<<hornet_graph_out.nE()<<"\n";

    // cudaProfilerStart();

    TM.start();
    // dynamic_bfs.run();
    // dynamic_bfs.print_check(batch_src, batch_dst, batch_size);

    if(graph.is_directed())
        dynamic_bfs.batch_update_directed(batch_dst,batch_size);
    else
        dynamic_bfs.batch_update_undirected(batch_src,batch_dst,batch_size);

    dynamic_bfs.run_dynamic();

    TM.stop();
    TM.print("Dynamic BFS");


    TM.start();
    dynamic_bfs.run_check(root);
    TM.stop();
    TM.print("BFS check");

    dynamic_bfs.reset();
    dynamic_bfs.set_parameters(root,alg);

    std::cout << "My root is " << root << std::endl;

    TM.start();
    dynamic_bfs.run();
    TM.stop();
    TM.print("Static BFS recheck");


    std::cout << "Number of levels is : " << dynamic_bfs.getLevels() << std::endl;

    // cudaProfilerStop();
    // TM.print("TopDown2");

    return 0;
}

int main(int argc, char* argv[]) {
    int ret = 0;
    hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
    {//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.

    ret = exec<hornets_nest::HornetDynamicGraph, hornets_nest::BfsTopDown2Dynamic>(argc, argv);
    // ret = exec<hornets_nest::HornetStaticGraph,  hornets_nest::BfsTopDown2Static >(argc, argv);

    }//scoping technique to make sure that hornets_nest::gpu::finalizeRMMPoolAllocation is called after freeing all RMM allocations.
    hornets_nest::gpu::finalizeRMMPoolAllocation();

    return ret;
}

