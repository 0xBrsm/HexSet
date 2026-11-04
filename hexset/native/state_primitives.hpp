// SPDX-License-Identifier: GPL-3.0-only
// These functions DO NOT implement Game phase checks, awards or recording yet.
#pragma once
#include "state_core.hpp"

namespace hexset_native {
struct IndexRange {
    std::vector<Int>::const_iterator begin_iterator;
    std::vector<Int>::const_iterator end_iterator;
    auto begin() const {return begin_iterator;}
    auto end() const {return end_iterator;}
};
inline IndexRange adjacent(const ArenaState& state,Block offsets,Block values,Int index) {
    const auto& at=state.get(offsets); const auto& data=state.get(values);
    // All callers validate the requested row; snapshot validates all offsets.
    return {data.cbegin()+at[static_cast<std::size_t>(index)],data.cbegin()+at[static_cast<std::size_t>(index)+1]};
}
inline void vertex_index(const ArenaState& state,Int vertex) {
    if (vertex<0 || static_cast<std::uint64_t>(vertex)>=state.get(Block::vertex_owner).size())
        throw std::invalid_argument("no such vertex");
}
inline void edge_index(const ArenaState& state,Int edge) {
    if (edge<0 || static_cast<std::uint64_t>(edge)>=state.get(Block::edge_owner).size())
        throw std::invalid_argument("no such edge");
}
inline bool settlement_placeable(const ArenaState& state,int seat,Int vertex,bool connected) {
    state.knowledge.check_seat(seat); vertex_index(state,vertex);
    const auto& kinds=state.get(Block::vertex_building);
    if (kinds[vertex]!=0) return false;
    for (Int neighbor:adjacent(state,Block::vertex_neighbors_offsets,Block::vertex_neighbors_values,vertex))
        if (kinds[neighbor]!=0) return false;
    if (!connected) return true;
    for (Int edge:adjacent(state,Block::vertex_edges_offsets,Block::vertex_edges_values,vertex))
        if (state.get(Block::edge_owner)[edge]==seat) return true;
    return false;
}
inline bool can_place_settlement(const ArenaState& state,int seat,Int vertex,bool connected) {
    state.knowledge.check_seat(seat); vertex_index(state,vertex);
    return state.piece_counts[seat][0]<5 && settlement_placeable(state,seat,vertex,connected);
}
inline bool city_upgradeable(const ArenaState& state,int seat,Int vertex) {
    state.knowledge.check_seat(seat); vertex_index(state,vertex);
    return state.get(Block::vertex_owner)[vertex]==seat && state.get(Block::vertex_building)[vertex]==1;
}
inline bool can_upgrade_to_city(const ArenaState& state,int seat,Int vertex) {
    state.knowledge.check_seat(seat); vertex_index(state,vertex);
    return state.piece_counts[seat][1]<4 && city_upgradeable(state,seat,vertex);
}
inline bool road_placeable(const ArenaState& state,int seat,Int edge) {
    state.knowledge.check_seat(seat); edge_index(state,edge);
    if (state.get(Block::edge_owner)[edge]!=-1) return false;
    for (int end=0;end<2;++end) {
        Int vertex=state.get(Block::edges)[edge*2+end],owner=state.get(Block::vertex_owner)[vertex];
        if (owner==seat) return true;
        if (owner==-1)
            for (Int neighbor:adjacent(state,Block::vertex_edges_offsets,Block::vertex_edges_values,vertex))
                if (state.get(Block::edge_owner)[neighbor]==seat) return true;
    }
    return false;
}
inline bool can_place_road(const ArenaState& state,int seat,Int edge) {
    state.knowledge.check_seat(seat); edge_index(state,edge);
    return state.piece_counts[seat][2]<15 && road_placeable(state,seat,edge);
}
inline std::vector<Int> placeable_roads(const ArenaState& state,int seat) {
    state.knowledge.check_seat(seat);
    const auto& owner=state.get(Block::vertex_owner); const auto& edges=state.get(Block::edges);
    const auto& roads=state.get(Block::edge_owner);
    std::vector<bool> reach(owner.size());
    for (std::size_t v=0;v<owner.size();++v) reach[v]=owner[v]==seat;
    for (std::size_t e=0;e<roads.size();++e) if (roads[e]==seat)
        for (int side=0;side<2;++side) if (owner[edges[e*2+side]]==-1) reach[edges[e*2+side]]=true;
    std::vector<Int> result;
    for (std::size_t e=0;e<roads.size();++e)
        if (roads[e]==-1 && (reach[edges[e*2]] || reach[edges[e*2+1]])) result.push_back(static_cast<Int>(e));
    return result;
}
inline void place_settlement(ArenaState& state,int seat,Int vertex,bool connected) {
    if (!can_place_settlement(state,seat,vertex,connected)) throw std::invalid_argument("cannot settle vertex");
    state.get(Block::vertex_owner)[vertex]=seat;
    state.get(Block::vertex_building)[vertex]=1;
    state.get(Block::building_enum)[vertex]=1;
    ++state.piece_counts[seat][0];
    ++state.occupancy_epoch;
}
inline void upgrade_to_city(ArenaState& state,int seat,Int vertex) {
    if (!can_upgrade_to_city(state,seat,vertex)) throw std::invalid_argument("cannot upgrade vertex");
    state.get(Block::vertex_building)[vertex]=2;
    state.get(Block::building_enum)[vertex]=1;
    --state.piece_counts[seat][0]; ++state.piece_counts[seat][1];
}
inline void place_road(ArenaState& state,int seat,Int edge) {
    if (!can_place_road(state,seat,edge)) throw std::invalid_argument("cannot build road");
    state.get(Block::edge_owner)[edge]=seat;
    ++state.piece_counts[seat][2];
    ++state.occupancy_epoch;
}

inline const std::array<Counts,4> costs={Counts{1,1,0,0,0},Counts{1,1,1,1,0},
                                       Counts{0,0,0,2,3},Counts{0,0,1,1,1}};
inline const Counts& cost(int purchase) {
    if (purchase<0 || purchase>=4) throw std::invalid_argument("unknown purchase");
    return costs[purchase];
}
inline bool can_afford(const ArenaState& state,int seat,int purchase) {
    state.knowledge.check_seat(seat); const auto& price=cost(purchase);
    for (int r=0;r<5;++r) if (state.get(Block::hands)[seat*5+r]<price[r]) return false;
    return true;
}
inline Counts trade_ratios(const ArenaState& state,int seat) {
    state.knowledge.check_seat(seat); Counts result{4,4,4,4,4};
    const auto& ports=state.get(Block::ports); const auto& owner=state.get(Block::vertex_owner);
    for (std::size_t i=0;i<ports.size();i+=5) {
        if (owner[ports[i+1]]!=seat && owner[ports[i+2]]!=seat) continue;
        Int resource=ports[i+3],ratio=ports[i+4];
        if (resource==-1) for (auto& r:result) r=std::min(r,ratio);
        else result[resource]=std::min(result[resource],ratio);
    }
    return result;
}
struct EconomyChange {
    std::vector<Int> hands,bank;
    void commit(ArenaState& state) {
        state.get(Block::hands)=std::move(hands); state.get(Block::bank)=std::move(bank);
    }
};
inline EconomyChange prepare_pay(const ArenaState& state,int seat,int purchase) {
    if (!can_afford(state,seat,purchase)) throw std::invalid_argument("cannot afford purchase");
    const auto& price=cost(purchase); EconomyChange result{state.get(Block::hands),state.get(Block::bank)};
    for (int r=0;r<5;++r) {result.hands[seat*5+r]-=price[r];result.bank[r]=plus(result.bank[r],price[r]);}
    return result;
}
inline void pay(ArenaState& state,int seat,int purchase) {auto change=prepare_pay(state,seat,purchase);change.commit(state);}
inline EconomyChange prepare_bank_trade(const ArenaState& state,int seat,int give,int receive) {
    state.knowledge.check_seat(seat); Knowledge::check_resource(give); Knowledge::check_resource(receive);
    if (give==receive) throw std::invalid_argument("cannot trade a resource for itself");
    Int ratio=trade_ratios(state,seat)[give]; const auto& hand=state.get(Block::hands); const auto& bank=state.get(Block::bank);
    if (hand[seat*5+give]<ratio || bank[receive]<1) throw std::invalid_argument("unavailable bank trade");
    EconomyChange result{hand,bank};
    result.hands[seat*5+give]-=ratio; result.bank[give]=plus(bank[give],ratio);
    result.hands[seat*5+receive]=plus(hand[seat*5+receive],1); --result.bank[receive];
    return result;
}
inline void bank_trade(ArenaState& state,int seat,int give,int receive) {
    auto change=prepare_bank_trade(state,seat,give,receive);change.commit(state);
}
inline std::vector<Int> production(const ArenaState& state,Int roll,bool gold=false) {
    if (roll<0 || roll>=13) throw std::invalid_argument("roll outside production table");
    auto players=static_cast<std::size_t>(state.get(Block::players)[0]);
    std::vector<Int> result(players*(gold ? 1 : 5));
    for (Int hex:adjacent(state,Block::roll_hexes_offsets,Block::roll_hexes_values,roll)) {
        Int terrain=state.get(Block::terrain)[hex];
        if (hex==state.get(Block::state_control)[0]) continue;
        if (gold ? terrain!=7 : terrain<0 || terrain>=5) continue;
        for (Int vertex:adjacent(state,Block::hex_vertices_offsets,Block::hex_vertices_values,hex)) {
            Int seat=state.get(Block::vertex_owner)[vertex];
            if (seat>=0) {
                auto at=static_cast<std::size_t>(gold ? seat : seat*5+terrain);
                result[at]=plus(result[at],state.get(Block::vertex_building)[vertex]);
            }
        }
    }
    return result;
}
struct Distribution {
    std::vector<Int> granted,hands,bank;
    void commit(ArenaState& state) {
        state.get(Block::hands)=std::move(hands); state.get(Block::bank)=std::move(bank);
    }
};
inline Distribution prepare_distribution(const ArenaState& state,Int roll) {
    auto gross=production(state,roll); auto players=static_cast<std::size_t>(state.get(Block::players)[0]);
    std::vector<Int> granted(players*5);
    const auto& bank=state.get(Block::bank);
    for (int r=0;r<5;++r) {
        std::vector<std::size_t> claimants; Int demand=0;
        for (std::size_t seat=0;seat<players;++seat) if (gross[seat*5+r]) {
            claimants.push_back(seat); demand=plus(demand,gross[seat*5+r]);
        }
        if (demand<=bank[r]) for (auto seat:claimants) granted[seat*5+r]=gross[seat*5+r];
        else if (claimants.size()==1) granted[claimants[0]*5+r]=bank[r];
    }
    auto after_hand=state.get(Block::hands),after_bank=bank;
    for (std::size_t seat=0;seat<players;++seat) for (int r=0;r<5;++r) {
        after_hand[seat*5+r]=plus(after_hand[seat*5+r],granted[seat*5+r]);
        after_bank[r]-=granted[seat*5+r];
    }
    return {std::move(granted),std::move(after_hand),std::move(after_bank)};
}
inline std::vector<Int> distribute(ArenaState& state,Int roll) {
    auto prepared=prepare_distribution(state,roll);
    prepared.commit(state); return std::move(prepared.granted);
}
inline Knowledge public_diff_knowledge(const Knowledge& knowledge,const std::vector<Int>& before,const std::vector<Int>& after) {
    if (before.size()!=after.size()) throw std::invalid_argument("invalid public hand-diff shape");
    Knowledge working=knowledge;
    for (std::size_t i=0;i<before.size();++i) {
        if (before[i]<0) throw std::invalid_argument("public hand counts must be nonnegative");
        Int delta=after[i]-before[i];
        if (delta>0) working.receive(static_cast<int>(i/5),static_cast<int>(i%5),delta);
        else if (delta<0) working.spend(static_cast<int>(i/5),static_cast<int>(i%5),-delta);
    }
    return working;
}
inline void public_hand_diff(ArenaState& state,const std::vector<Int>& before) {
    state.knowledge=public_diff_knowledge(state.knowledge,before,state.get(Block::hands));
}
inline Int public_points(const ArenaState& state,int seat) {
    state.knowledge.check_seat(seat);
    const auto& counts=state.piece_counts[seat];
    Int points=plus(counts[0],plus(counts[1],counts[1]));
    if (state.get(Block::state_control)[1]==seat) points=plus(points,2);
    if (state.get(Block::state_control)[2]==seat) points=plus(points,2);
    return points;
}
inline Int true_points(const ArenaState& state,int seat) {
    return plus(public_points(state,seat),plus(state.get(Block::dev_cards)[seat*5+1],state.get(Block::new_dev_cards)[seat*5+1]));
}
} // namespace hexset_native
