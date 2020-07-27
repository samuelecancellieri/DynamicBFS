/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include "Static/BreadthFirstSearch/TopDown2.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off
#include "Static/BreadthFirstSearch/TopDown2.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off
// #include "Dynamic/BFS/DynBFS.cuh"
#include <StandardAPI.hpp>
#include <Graph/GraphStd.hpp>
#include <Util/CommandLineParam.hpp>
#include <cuda_profiler_api.h> //--profile-from-start off
#include <Hornet.hpp>
#include "StandardAPI.hpp"
#include "Util/BatchFunctions.hpp"
#include "Util/RandomGraphData.cuh"
#include <Host/FileUtil.hpp>            //xlib::extract_filepath_noextension
#include <Device/Util/CudaUtil.cuh>     //xlib::deviceInfo
#include <algorithm>                    //std:.generate
#include <chrono>                       //std::chrono
#include <random>                       //std::mt19937_64
#include <Core/Static/Static.cuh>
#include <ctime>

template <typename HornetGraph, typename BFS>
int exec(int argc, char* argv[]) {
    using namespace timer;
    using namespace hornets_nest;

    using namespace graph::structure_prop;
    using namespace graph::parsing_prop;
    using namespace graph;
    using HornetGPU = hornet::gpu::Hornet<vert_t>;
    using UpdatePtr = hornet::BatchUpdatePtr<vert_t, hornet::EMPTY, hornet::DeviceType::HOST>;
    using BatchUpdate = hornet::gpu::BatchUpdate<vert_t>;

    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1000000000000000000);


    // graph::GraphStd<vid_t, eoff_t> graph(UNDIRECTED );
    // graph::GraphStd<vid_t, eoff_t> graph(DIRECTED);
    // graph::GraphStd<vid_t, eoff_t> graph(ENABLE_INGOING);
    graph::GraphStd<vid_t, eoff_t> graph;
    CommandLineParam cmd(graph, argc, argv,false);
    // ParsingProp pp;
    // graph.read(argv[1],pp);

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(),
                           graph.csr_out_edges());

    Timer<DEVICE> TM;
    HornetGraph hornet_graph(hornet_init);

    BFS bfs_top_down(hornet_graph);

    vid_t root = graph.max_out_degree_id();
    // if (argc==3)
    //     root = atoi(argv[2]);
    int numberRoots = 1;
    if (argc>=3)
      numberRoots = atoi(argv[2]);

    int alg = 0;
    if (argc>=4)
      alg = atoi(argv[3]);

    std::cout << "My root is " << root << std::endl;

    std::cout<<"prima size "<<hornet_graph.nE()<<"\n";


    int batch_size = 10000;

    vert_t* batch_src = new vert_t[batch_size]();
    vert_t* batch_dst = new vert_t[batch_size]();

    generateBatch(graph, batch_size, batch_src, batch_dst, BatchGenType::INSERT, batch_gen_property::UNIQUE);

    UpdatePtr ptr(batch_size, batch_src, batch_dst);
    BatchUpdate batch_update_src_to_dst(ptr);

    UpdatePtr ptr2(batch_size, batch_dst, batch_src);
    BatchUpdate batch_update_dst_to_src(ptr2);

    hornet_graph.insert(batch_update_src_to_dst);
    hornet_graph.insert(batch_update_dst_to_src);

    batch_update_src_to_dst.print();

    std::cout<<"dopo size "<<hornet_graph.nE()<<"\n";

    cudaProfilerStart();
    for(int i=0; i<numberRoots; i++){
        bfs_top_down.reset();
        bfs_top_down.set_parameters((root+i)%graph.nV(),alg);
    TM.start();
        // bfs_top_down.run();
        bfs_top_down.print_check(batch_src, batch_dst, batch_size);
    TM.stop();
        std::cout << "Number of levels is : " << bfs_top_down.getLevels() << std::endl;
    }

    cudaProfilerStop();
    TM.print("TopDown2");

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

