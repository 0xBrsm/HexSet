// SPDX-License-Identifier: GPL-3.0-only
// Projected native graph queries prepare award/win changes before any mutation.
#pragma once
#include "game_simple.hpp"
#include "roads_core.hpp"

namespace hexset_native {
inline bool has_placeable_road(const RoadGraph& graph,int seat) {
    for (std::size_t e=0;e<graph.edges();++e) if (graph.owner(static_cast<Int>(e))==-1) {
        auto ends=graph.ends(static_cast<Int>(e));
        for (Int vertex:{ends.first,ends.second}) {
            if (graph.building(vertex)==seat) return true;
            if (graph.building(vertex)==-1)
                for (Int edge:graph.incident(vertex)) if (graph.owner(edge)==seat) return true;
        }
    }
    return false;
}
inline bool pending_after(const ArenaState& state,const std::vector<Int>& game,const RoadGraph& graph,Int added_roads) {
    int actor=static_cast<int>(game[ActorColumn]);
    return game[FreeRoadsColumn]>0 && state.piece_counts[actor][2]+added_roads<15 && has_placeable_road(graph,actor);
}
inline void projected_win(const ArenaState& state,std::vector<Int>& game,Int holder,Int added_points) {
    int actor=static_cast<int>(game[ActorColumn]);
    Int award_change=(holder==actor ? 2 : 0)-(state.get(Block::state_control)[1]==actor ? 2 : 0);
    Int points=plus(plus(true_points(state,actor),added_points),award_change);
    if (points>=state.get(Block::rules)[0]) {game[WonByColumn]=actor;game[PhaseColumn]=6;}
}
inline void apply_placement(ArenaState& state,int type,Int index) {
    if (type<4 || type>7) throw std::invalid_argument("unsupported native placement action");
    auto game=state.get(Block::game_control);
    bool was_main=game[PhaseColumn]==5;
    int actor=static_cast<int>(game[ActorColumn]);
    std::optional<EconomyChange> economy;
    std::optional<Knowledge> knowledge;
    RoadGraph graph{state};
    std::vector<Int> lengths;
    Int holder=-1,added_roads=0;
    if (type==4) {
        if (game[PhaseColumn]!=0) throw std::invalid_argument("expected SETUP_SETTLEMENT phase");
        if (!can_place_settlement(state,actor,index,false)) throw std::invalid_argument("cannot settle vertex");
        graph.extra_vertex=index;graph.vertex_owner=actor;
        lengths=projected_road_lengths(graph,-1,index,actor);
        economy=EconomyChange{state.get(Block::hands),state.get(Block::bank)};
        if (game[SetupStepColumn]>=state.get(Block::players)[0])
            for (Int hex:adjacent(state,Block::vertex_hexes_offsets,Block::vertex_hexes_values,index)) {
                Int resource=state.get(Block::terrain)[hex];
                if (resource>=0 && resource<5 && economy->bank[resource]>0) {
                    --economy->bank[resource];
                    economy->hands[actor*5+resource]=plus(economy->hands[actor*5+resource],1);
                }
            }
        knowledge=public_diff_knowledge(state.knowledge,state.get(Block::hands),economy->hands);
        holder=award(lengths,state.get(Block::state_control)[1],5);
        game[LastSettlementColumn]=index;game[PhaseColumn]=1;
    } else if (type==5) {
        if (game[PhaseColumn]!=1) throw std::invalid_argument("expected SETUP_ROAD phase");
        edge_index(state,index);vertex_index(state,game[LastSettlementColumn]);
        bool touches=false;
        for (Int edge:adjacent(state,Block::vertex_edges_offsets,Block::vertex_edges_values,game[LastSettlementColumn]))
            if (edge==index && state.get(Block::edge_owner)[edge]==-1) touches=true;
        if (!touches || !can_place_road(state,actor,index)) throw std::invalid_argument("opening road must touch last settlement");
        graph.extra_edge=index;graph.extra_owner=actor;added_roads=1;
        lengths=projected_road_lengths(graph,actor);holder=award(lengths,state.get(Block::state_control)[1],5);
        const auto& queue=state.get(Block::setup_queue);
        if (queue.empty()) throw std::invalid_argument("empty native setup queue");
        game[SetupStepColumn]=plus(game[SetupStepColumn],1);
        while (game[SetupStepColumn]<static_cast<Int>(queue.size()) && locked(state,static_cast<int>(queue[game[SetupStepColumn]])))
            ++game[SetupStepColumn];
        if (game[SetupStepColumn]<static_cast<Int>(queue.size())) {
            game[ActorColumn]=queue[game[SetupStepColumn]];game[PhaseColumn]=0;
        } else {
            int first=static_cast<int>(queue[0]);
            game[ActorColumn]=locked(state,first) ? next_unlocked(state,first) : first;
            game[PhaseColumn]=2;
        }
    } else if (type==6) {
        bool free=game[FreeRoadsColumn]>0;
        if ((game[PhaseColumn]!=2 || !free) && game[PhaseColumn]!=5) throw std::invalid_argument("expected MAIN phase");
        if (!can_place_road(state,actor,index)) throw std::invalid_argument("cannot build road");
        if (free) --game[FreeRoadsColumn];
        else {
            economy=prepare_pay(state,actor,0);
            knowledge=public_diff_knowledge(state.knowledge,state.get(Block::hands),economy->hands);
        }
        graph.extra_edge=index;graph.extra_owner=actor;added_roads=1;
        lengths=projected_road_lengths(graph,actor);holder=award(lengths,state.get(Block::state_control)[1],5);
        projected_win(state,game,holder,0);
    } else {
        require_main(state);
        if (!can_place_settlement(state,actor,index,true)) throw std::invalid_argument("cannot settle vertex");
        economy=prepare_pay(state,actor,1);
        knowledge=public_diff_knowledge(state.knowledge,state.get(Block::hands),economy->hands);
        graph.extra_vertex=index;graph.vertex_owner=actor;
        lengths=projected_road_lengths(graph,-1,index,actor);holder=award(lengths,state.get(Block::state_control)[1],5);
        projected_win(state,game,holder,1);
    }
    if (was_main && game[PhaseColumn]==5 && !pending_after(state,game,graph,added_roads))
        game[TradeEventColumn]=game[TurnsColumn];
    // Everything that can reject or allocate is complete. These commits do
    // not rebuild/copy the board or call back to Python for any field/query.
    if (type==4 || type==7) place_settlement(state,actor,index,type==7);
    else place_road(state,actor,index);
    if (economy) economy->commit(state);
    if (knowledge) state.knowledge=std::move(*knowledge);
    state.get(Block::road_lengths)=std::move(lengths);state.get(Block::state_control)[1]=holder;
    state.get(Block::game_control)=std::move(game);
}
} // namespace hexset_native
