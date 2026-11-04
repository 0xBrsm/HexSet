// SPDX-License-Identifier: GPL-3.0-only
// This adapter reads persistent native arrays directly; no Python objects.
#pragma once
#include <climits>
#include "state_primitives.hpp"

namespace hexset_native {
struct RoadGraph {
    const ArenaState& state;
    Int extra_edge=-1, extra_owner=-1, extra_vertex=-1, vertex_owner=-1;
    std::size_t vertices() const {return state.get(Block::vertex_owner).size();}
    std::size_t edges() const {return state.get(Block::edge_owner).size();}
    Int owner(Int edge) const {return edge==extra_edge ? extra_owner : state.get(Block::edge_owner)[edge];}
    Int building(Int vertex) const {return vertex==extra_vertex ? vertex_owner : state.get(Block::vertex_owner)[vertex];}
    std::pair<Int,Int> ends(Int edge) const {const auto& all=state.get(Block::edges);return {all[edge*2],all[edge*2+1]};}
    IndexRange incident(Int vertex) const {return adjacent(state,Block::vertex_edges_offsets,Block::vertex_edges_values,vertex);}
    bool passable(Int vertex,int seat) const {Int owner=building(vertex);return owner==-1 || owner==seat;}
};
struct RoadLink {std::size_t index;Int far;int cost;bool onward,joined;};
struct RoadSearch {
    const RoadGraph& graph;
    int seat,budget,target=INT_MAX;
    bool keep_paths,found=false;
    std::vector<Int> usable,starts,best;
    std::vector<std::vector<RoadLink>> links;
    std::vector<unsigned char> joins,used;
    std::vector<std::vector<unsigned char>> paths;
    std::size_t owned;
    RoadSearch(const RoadGraph& g,int s,int roads,bool keep=false):graph(g),seat(s),budget(roads),keep_paths(keep),
        best(static_cast<std::size_t>(roads)+1,0),links(g.vertices()),joins(g.vertices(),0) {
        for (std::size_t e=0;e<g.edges();++e) if (g.owner(static_cast<Int>(e))==seat) {
            usable.push_back(static_cast<Int>(e));auto ends=g.ends(static_cast<Int>(e));
            if (g.building(ends.first)==-1) joins[ends.first]=1;
            if (g.building(ends.second)==-1) joins[ends.second]=1;
        }
        owned=usable.size();std::vector<Int> frontier;
        for (std::size_t v=0;v<joins.size();++v) {
            if (g.building(static_cast<Int>(v))==seat) joins[v]=1;
            if (joins[v]) frontier.push_back(static_cast<Int>(v));
        }
        std::vector<unsigned char> near(g.edges(),0);
        for (int depth=0;depth<roads && !frontier.empty();++depth) {
            std::vector<Int> grown;std::vector<unsigned char> reached(g.vertices(),0);
            for (Int v:frontier) for (Int e:g.incident(v)) {
                if (g.owner(e)!=-1 || near[e]) continue;
                near[e]=1;auto ends=g.ends(e);
                for (Int w:{ends.first,ends.second}) if (g.passable(w,seat) && !reached[w]) {
                    grown.push_back(w);reached[w]=1;
                }
            }
            frontier.swap(grown);
        }
        for (std::size_t e=0;e<near.size();++e) if (near[e]) usable.push_back(static_cast<Int>(e));
        used.resize(usable.size(),0);
        if (keep_paths) paths.assign(static_cast<std::size_t>(roads)+1,used);
        for (std::size_t i=0;i<usable.size();++i) {
            auto ends=g.ends(usable[i]);int cost=i<owned ? 0 : 1;
            add(ends.first,ends.second,i,cost);add(ends.second,ends.first,i,cost);
        }
    }
    void add(Int v,Int w,std::size_t index,int cost) {
        if (links[v].empty()) starts.push_back(v);
        links[v].push_back({index,w,cost,graph.passable(w,seat),cost==0 || joins[w]!=0});
    }
    void improve(int spent,Int length) {
        if (length>=target) found=true;
        if (length>best[spent]) {best[spent]=length;if (keep_paths) paths[spent]=used;}
    }
    void walk(Int vertex,int spent,Int length,bool touched) {
        if (touched) improve(spent,length);
        for (const auto& link:links[vertex]) {
            if (found) return;
            if (used[link.index] || spent+link.cost>budget) continue;
            used[link.index]=1;bool reached=touched || link.joined;
            if (link.onward) walk(link.far,spent+link.cost,length+1,reached);
            else if (reached) improve(spent+link.cost,length+1);
            used[link.index]=0;
        }
    }
    void run(int stop_at=INT_MAX) {
        target=stop_at;
        for (Int vertex:starts) {walk(vertex,0,0,joins[vertex]!=0);if (found) return;}
        for (int j=1;j<=budget;++j) if (best[j-1]>=best[j]) {
            best[j]=best[j-1];if (keep_paths) paths[j]=paths[j-1];
        }
    }
};
inline Int longest_road(const RoadGraph& graph,int seat) {RoadSearch search(graph,seat,0);search.run();return search.best[0];}
inline Int longest_road(const ArenaState& state,int seat,Int extra_edge=-1) {
    state.knowledge.check_seat(seat);
    if (extra_edge!=-1) edge_index(state,extra_edge);
    return longest_road(RoadGraph{state,extra_edge,seat},seat);
}
inline std::vector<Int> road_lengths(const ArenaState& state) {
    std::vector<Int> result(state.knowledge.common.size());
    for (std::size_t s=0;s<result.size();++s) result[s]=longest_road(state,static_cast<int>(s));
    return result;
}
inline Int award(const std::vector<Int>& counts,Int holder,Int minimum) {
    Int best=*std::max_element(counts.begin(),counts.end());if (best<minimum) return -1;
    if (holder!=-1 && counts[holder]==best) return holder;
    Int result=-1;
    for (std::size_t s=0;s<counts.size();++s) if (counts[s]==best) {
        if (result!=-1) return -1;
        result=static_cast<Int>(s);
    }
    return result;
}
inline void check_road_cache(const ArenaState& state,int builder=-1) {
    const auto& lengths=state.get(Block::road_lengths);
    for (std::size_t s=0;s<lengths.size();++s)
        if (state.piece_counts[s][2]>0 && lengths[s]==0 && static_cast<int>(s)!=builder)
            throw std::invalid_argument("stale road-length cache");
}
inline std::vector<Int> projected_road_lengths(const RoadGraph& graph,int road_owner=-1,Int settlement_vertex=-1,int settlement_owner=-1) {
    const auto& state=graph.state;
    if (road_owner!=-1 || settlement_vertex!=-1) check_road_cache(state,road_owner);
    auto result=state.get(Block::road_lengths);
    if (road_owner!=-1) result[road_owner]=longest_road(graph,road_owner);
    else if (settlement_vertex!=-1) {
        std::vector<bool> affected(result.size(),false);
        for (Int edge:graph.incident(settlement_vertex)) {
            Int owner=graph.owner(edge);
            if (owner!=-1 && owner!=settlement_owner) affected[owner]=true;
        }
        for (std::size_t seat=0;seat<affected.size();++seat) if (affected[seat]) result[seat]=longest_road(graph,static_cast<int>(seat));
    } else for (std::size_t seat=0;seat<result.size();++seat) result[seat]=longest_road(graph,static_cast<int>(seat));
    return result;
}
inline void update_longest_road(ArenaState& state,int road_owner=-1,Int settlement_vertex=-1,int settlement_owner=-1) {
    auto lengths=projected_road_lengths(RoadGraph{state},road_owner,settlement_vertex,settlement_owner);
    Int holder=award(lengths,state.get(Block::state_control)[1],5);
    state.get(Block::road_lengths)=std::move(lengths);state.get(Block::state_control)[1]=holder;
    // A full refresh can change previously stale imported lengths/holder even
    // without an occupancy change; invalidate the dependent award-distance cache.
    state.road_cache_epoch=std::numeric_limits<std::uint64_t>::max();
}
inline Int update_largest_army(ArenaState& state) {
    Int holder=award(state.get(Block::knights_played),state.get(Block::state_control)[2],3);
    state.get(Block::state_control)[2]=holder;return holder;
}
inline void fill_road_caches(const ArenaState& state) {
    if (state.road_cache_epoch==state.occupancy_epoch) return;
    const auto& lengths=state.get(Block::road_lengths);Int holder=state.get(Block::state_control)[1];
    std::vector<Int> take(lengths.size(),4);
    RoadGraph graph{state};
    for (std::size_t s=0;s<lengths.size();++s) {
        if (static_cast<Int>(s)==holder) {take[s]=0;continue;}
        Int beat=5;
        for (std::size_t other=0;other<lengths.size();++other) if (other!=s) beat=std::max(beat,plus(lengths[other],1));
        Int budget=std::min<Int>(3,15-state.piece_counts[s][2]);
        for (Int k=std::max<Int>(1,beat-state.piece_counts[s][2]);k<=budget;++k) {
            RoadSearch search(graph,static_cast<int>(s),static_cast<int>(k));
            search.run(static_cast<int>(beat));if (search.found) {take[s]=k;break;}
        }
    }
    std::size_t vertices=graph.vertices();std::vector<Int> reach(lengths.size()*vertices,-1);
    for (std::size_t s=0;s<lengths.size();++s) {
        std::vector<Int> frontier;
        for (std::size_t v=0;v<vertices;++v) if (graph.building(static_cast<Int>(v))==static_cast<Int>(s)) frontier.push_back(static_cast<Int>(v));
        if (frontier.empty()) continue;
        for (std::size_t e=0;e<graph.edges();++e) if (graph.owner(static_cast<Int>(e))==static_cast<Int>(s)) {
            auto ends=graph.ends(static_cast<Int>(e));frontier.push_back(ends.first);frontier.push_back(ends.second);
        }
        for (Int v:frontier) reach[s*vertices+v]=0;
        Int left=15-state.piece_counts[s][2];
        for (Int step=1;!frontier.empty() && step<=left;++step) {
            std::vector<Int> grown;
            for (Int v:frontier) {
                if (!graph.passable(v,static_cast<int>(s))) continue;
                for (Int e:graph.incident(v)) if (graph.owner(e)==-1) {
                    auto ends=graph.ends(e);Int w=ends.first==v ? ends.second : ends.first;
                    if (reach[s*vertices+w]<0) {reach[s*vertices+w]=step;grown.push_back(w);}
                }
            }
            frontier.swap(grown);
        }
    }
    state.cached_roads_to_take=std::move(take);state.cached_reach=std::move(reach);state.road_cache_epoch=state.occupancy_epoch;
}
inline const std::vector<Int>& roads_to_take_all(const ArenaState& state) {fill_road_caches(state);return state.cached_roads_to_take;}
inline const std::vector<Int>& reach_all(const ArenaState& state) {fill_road_caches(state);return state.cached_reach;}
} // namespace hexset_native
