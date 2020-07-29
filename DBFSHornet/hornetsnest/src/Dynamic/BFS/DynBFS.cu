#include "Dynamic/BFS/DynBFS.cuh"

// using BfsTopDown2Dynamic = BfsTopDown2<HornetDynamicGraph>;
// using BfsTopDown2Static  = BfsTopDown2<HornetStaticGraph>;

namespace hornets_nest 
{

    const dist_t INF = std::numeric_limits<dist_t>::max()-1;

//------------------------------------------------------------------------------
///////////////
// OPERATORS //
///////////////

struct ResetDistance
{
    dist_t* d_distances;

    OPERATOR(Vertex& vertex)
    {
        auto child = vertex.id();

        if(d_distances[child] != 0)
            d_distances[child] = INF;
    }

};

struct FatherUpdate
{
    dist_t* d_distances;
    dist_t* old_distances;

    OPERATOR(Vertex& vertex, Edge& edge)
    {
        auto child = vertex.id();
        auto father = edge.dst_id();

        // printf("BEFORE child %d childdeg %d father %d childd %d fatherd %d\n",child,vertex.degree(),father,old_distances[child],old_distances[father]);
        atomicMin(d_distances+child, old_distances[father]+1); //operation to find the father nearest to the source)
        // printf("AFTER child %d childdeg %d father %d childd %d fatherd %d\n",child,vertex.degree(),father,d_distances[child],old_distances[father]);

    }
};

struct LowestFather
{
    dist_t* d_distances;
    dist_t* old_distances;
    TwoLevelQueue<vert_t> queue;

    OPERATOR(Vertex& vertex)
    {
        auto child = vertex.id();

        if(d_distances[child] != old_distances[child])
        {
            old_distances[child] = d_distances[child];
            queue.insert(child);
        }
    }
};

struct BFSDynamicDeletion
{
    TwoLevelQueue<vert_t> queue2;

    OPERATOR(Vertex& vertex, Edge& edge) 
    {
        auto dst = edge.dst_id();
        queue2.insert(dst);
    }
};

struct BFSDynamicInsertion
{
    dist_t* d_distances;
    dist_t* old_distances;
    TwoLevelQueue<vert_t> queue;

    OPERATOR(Vertex& vertex, Edge& edge) 
    {
        auto dst = edge.dst_id();
        auto src = vertex.id();

        if(atomicMin(d_distances+dst, d_distances[src]+1) > d_distances[src]+1)
        {
            old_distances[dst] = d_distances[dst];
            queue.insert(dst);
        }
    }
};

struct BFSOperatorCheck
{
    dist_t* d_distances;
    dist_t* old_distances;
    dist_t current_level;
    TwoLevelQueue<vert_t> queue;

    OPERATOR(Vertex& vertex, Edge& edge)
    {
        auto dst = edge.dst_id();
        auto src = vertex.id();

        if (atomicCAS(d_distances + dst, INF, current_level) == INF)
        {
            if(old_distances[dst] != d_distances[dst])
            {
                printf("changed vertex %d old is %d correct is %d father is %d and dd is %d \n",dst,old_distances[dst],d_distances[dst],src,d_distances[src]);
            }
            queue.insert(dst);
        }
    }
};


struct BFSOperatorAtomic 
{                  //deterministic
    dist_t               current_level;
    dist_t*              d_distances;
    dist_t*              old_distances;
    TwoLevelQueue<vid_t> queue;

    OPERATOR(Vertex& vertex, Edge& edge) 
    {
        auto dst = edge.dst_id();
        // if(d_distances[dst]==INF){
        if (atomicCAS(d_distances + dst, INF, current_level) == INF)
        {
            queue.insert(dst);            
            old_distances[dst] = d_distances[dst];
        }
        // }
    }
};

struct PrintCheck
{
    dist_t* d_distances;

    OPERATOR(Vertex& vertex, Edge& edge)
    {
        auto dst = edge.dst_id();
        auto src = vertex.id();

        printf("src %d srcdeg %d dst %d srcdist %d dstdist %d\n",src,vertex.degree(),dst,d_distances[src],d_distances[dst]);
    }
};

//------------------------------------------------------------------------------
/////////////////
// DynBFS //
/////////////////

DynBFS::DynBFS(HornetGraph& hornet, HornetGraph& hornet_in) :
                                    StaticAlgorithm<HornetGraph>(hornet),
                                    queue(hornet, 5),
                                    queue2(hornet, 5),
                                    load_balancing(hornet),
                                    lrb_lb(hornet),
                                    inverted_graph(hornet_in) 
{
    pool.allocate(&d_distances, hornet.nV());
    pool.allocate(&old_distances, hornet.nV());
    reset();
}

DynBFS::~DynBFS() 
{
}

void DynBFS::reset()
{
    current_level = 1;
    queue.clear();

    auto distances = d_distances;
    lb_mechansim = 0;
    forAllnumV(StaticAlgorithm<HornetGraph>::hornet, [=] __device__ (int i){ distances[i] = INF; });

    distances = old_distances;
    forAllnumV(StaticAlgorithm<HornetGraph>::hornet, [=] __device__ (int i){ distances[i] = INF; });
}

void DynBFS::set_parameters(vid_t source,int load_balancing)
{
    bfs_source = source;
    queue.insert(bfs_source);               // insert bfs source in the frontier
    gpu::memsetZero(d_distances + bfs_source);  //reset source distance
    lb_mechansim = load_balancing;
}

void DynBFS::print_check(vert_t* update_src, vert_t* update_dst, int update_size)
{
    queue.clear();
    queue.insert(update_src, update_size);

    std::cout<<"check src queue"<<"\n";

    forAllEdges(StaticAlgorithm<HornetGraph>::hornet, queue, PrintCheck {d_distances}, lrb_lb);

    queue.clear();
    queue.insert(update_dst, update_size);
    std::cout<<"check dst queue"<<"\n";

    forAllEdges(StaticAlgorithm<HornetGraph>::hornet, queue, PrintCheck {d_distances}, lrb_lb);
}

void DynBFS::batch_update_directed(vert_t* update_dst,int update_size)
{
    queue.clear(); //bfs queue clear
    queue2.clear(); //temp queue clear
    queue2.insert(update_dst, update_size); //insert dst vertices in queue

    std::cout<<"VERTICES IN UPDATE QUEUE "<<queue2.size()<<"\n";

    // forAllVertices(inverted_graph, queue2, ResetDistance{d_distances}); //reset distances for each interested vertices in batch
    forAllEdges(inverted_graph, queue2, FatherUpdate {d_distances, old_distances}, lrb_lb); //find lowest father for each modified vertex
    forAllVertices(inverted_graph, queue2, LowestFather {d_distances, old_distances, queue}); //update distance based on lowest father
}

void DynBFS::batch_update_undirected(vert_t* update_src, vert_t* update_dst, int update_size)
{
    //UPDATE DISTANCES BASED ON SRC
    queue.clear(); //bfs queue clear
    queue.insert(update_src,update_size);
    queue.insert(update_src,update_size);
    // queue2.clear(); //temp queue clear
    // queue2.insert(update_src,update_size); //insert update batch in temp queue

    // std::cout<<"VERTICES IN UPDATE QUEUE "<<queue2.size()<<"\n";

    // forAllEdges(hornet, queue2, FatherUpdate {d_distances, old_distances}, lrb_lb); //find lowest father for each modified vertex
    // forAllVertices(hornet, queue2, LowestFather {d_distances, old_distances, queue}); //update distance based on lowest father

    // //UPDATE DISTANCES BASED ON DST 
    // queue2.clear();
    // queue2.insert(update_dst,update_size);

    // std::cout<<"VERTICES IN UPDATE QUEUE "<<queue2.size()<<"\n";

    // forAllEdges(hornet, queue2, FatherUpdate {d_distances, old_distances}, lrb_lb);
    // forAllVertices(hornet, queue2, LowestFather {d_distances, old_distances, queue});
}

void DynBFS::run()
{
    printf("bfs_source = %d\n",bfs_source);

    while (queue.size() > 0) {
    
        if(lb_mechansim==0){
            forAllEdges(
                StaticAlgorithm<HornetGraph>::hornet, queue,
                        BFSOperatorAtomic { current_level, d_distances, old_distances, queue },
                        lrb_lb);
        }else{
            forAllEdges(
                StaticAlgorithm<HornetGraph>::hornet, queue,
                        BFSOperatorAtomic { current_level, d_distances, old_distances, queue },
                        load_balancing);
        }
        queue.swap();
        current_level++;
    }
    // std::cout << "Number of levels is : " << current_level << std::endl;
}

void DynBFS::run_dynamic()
{
    int count = 0;
    int totalVisited = 0;

    queue.swap();

    // if(!batch_type)
    // {
    while(queue.size() > 0)
    {
        std::cout<<"VERTICES IN QUEUE "<<queue.size()<<"\n";
        totalVisited += queue.size();

        //execute BFS
        forAllEdges(hornet, queue, BFSDynamicInsertion {d_distances, old_distances, queue}, lrb_lb);
        queue.swap();
    
        ++count;
    }
    // }
    // else
    // {
    //     while(queue.size() > 0)
    //     {
    //         std::cout<<"VERTICES IN QUEUE "<<queue.size()<<"\n";
    //         totalVisited += queue.size();

    //         //execute BFS
    //         forAllEdges(hornet, queue, BFSDynamicDeletion {queue2}, load_balancing);

    //         queue2.swap();
    //         forAllVertices(hornet, queue2, ResetDistance{d_distances}); //reset distances for each interested vertices in batch
    //         forAllEdges(hornet, queue2, FatherUpdate {d_distances, old_distances}, load_balancing); //find lowest father for each modified vertex
    //         forAllVertices(hornet, queue2, LowestFather {d_distances, old_distances, queue}); //update distance based on lowest father
    //         queue.swap();

    //         ++count;
    //     }
    // }

    std::cout<<"BFS EXECUTED "<<count<<" TIMES AND VISITED "<<totalVisited<<" NODES\n";
}

void DynBFS::run_check(vid_t source)
{
    current_level = 1;
    auto distances = d_distances;
    forAllnumV(StaticAlgorithm<HornetGraph>::hornet, [=] __device__ (int i){ distances[i] = INF; });

    bfs_source = source;
    queue.clear();
    queue.insert(bfs_source);
    gpu::memsetZero(d_distances + bfs_source);  //reset source distance

    while(queue.size() > 0)
    {
        forAllEdges(hornet, queue, BFSOperatorCheck {d_distances, old_distances, current_level, queue }, lrb_lb);
        queue.swap();
        current_level++;
    }
}


void DynBFS::release()
{
    d_distances = nullptr;
}

bool DynBFS::validate()
{
    return true;
}

} // namespace hornets_nest
