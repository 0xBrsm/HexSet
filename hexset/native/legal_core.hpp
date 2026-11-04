// SPDX-License-Identifier: GPL-3.0-only
// Pure native legality and contract-9 candidates. Debug bindings may copy the
// result, while a persistent worker writes these buffers directly to requests.
#pragma once
#include <set>
#include "game_chance.hpp"

namespace hexset_native {
struct AtomicAction { int type; Int a=0,b=0; };
inline bool optional_none(const std::vector<Int>& values) {
    return values.size()==1 && values[0]==-1;
}
inline std::vector<Int> ordered_roads(const RoadGraph& graph,int seat) {
    std::vector<Int> out;
    for(std::size_t e=0;e<graph.edges();++e) {
        if(graph.owner(static_cast<Int>(e))!=-1) continue;
        auto ends=graph.ends(static_cast<Int>(e)); bool joined=false;
        for(Int v:{ends.first,ends.second}) {
            if(graph.building(v)==seat) joined=true;
            else if(graph.building(v)==-1)
                for(Int other:graph.incident(v)) if(graph.owner(other)==seat) joined=true;
        }
        if(joined) out.push_back(static_cast<Int>(e));
    }
    return out;
}
inline std::vector<AtomicAction> native_robber_targets(const ArenaState& state,int actor) {
    std::vector<AtomicAction> out;
    for(Int hex:robber_targets(state,actor)) {
        auto victims=robber_victims(state,hex,actor);
        if(victims.empty()) out.push_back({9,hex,static_cast<Int>(state.knowledge.common.size())});
        else for(Int seat:victims) out.push_back({9,hex,seat});
    }
    return out;
}
inline std::vector<AtomicAction> native_card_actions(const ArenaState& state,int actor,bool before_roll=false) {
    std::vector<AtomicAction> out;
    if(state.get(Block::game_control)[CardPlayedColumn]) return out;
    const auto& cards=state.get(Block::dev_cards);
    const auto& permitted=state.get(Block::before_roll);
    auto holds=[&](int card) {return cards[actor*5+card]>0 && (!before_roll || optional_none(permitted) ||
        std::find(permitted.begin(),permitted.end(),card)!=permitted.end());};
    if(holds(0)) out.push_back({10});
    if(holds(2) && state.piece_counts[actor][2]<15 && !placeable_roads(state,actor).empty()) out.push_back({3});
    if(holds(4)) for(int r=0;r<5;++r) out.push_back({11,r});
    if(holds(3)) {
        int index=0; const auto& bank=state.get(Block::bank);
        for(int a=0;a<5;++a) for(int b=a;b<5;++b,++index)
            if(bank[a]>=(a==b ? 2:1) && bank[b]>=(a==b ? 2:1)) out.push_back({12,index});
    }
    return out;
}
inline std::vector<AtomicAction> native_legal_actions(const ArenaState& state,int seat=-1) {
    const auto& game=state.get(Block::game_control);
    int phase=static_cast<int>(game[PhaseColumn]),actor=static_cast<int>(game[ActorColumn]);
    if(phase==6) return {};
    const auto& quotas=state.get(Block::discard_quota);
    int owing=-1; for(std::size_t s=0;s<quotas.size();++s) if(quotas[s]>0) {owing=static_cast<int>(s);break;}
    if(seat!=-1) {
        if(seat<0 || static_cast<std::size_t>(seat)>=state.knowledge.common.size()) return {};
        if(phase==3 && owing!=-1) {if(quotas[seat]<=0) return {};}
        else if(seat!=actor) return {};
    }
    std::vector<AtomicAction> out;
    if(phase==0) {
        for(std::size_t v=0;v<state.get(Block::vertex_owner).size();++v)
            if(can_place_settlement(state,actor,static_cast<Int>(v),false)) out.push_back({4,static_cast<Int>(v)});
        return out;
    }
    if(phase==1) {
        for(Int e:adjacent(state,Block::vertex_edges_offsets,Block::vertex_edges_values,game[LastSettlementColumn]))
            if(state.get(Block::edge_owner)[e]==-1) out.push_back({5,e});
        return out;
    }
    if(phase==3) {
        if(owing==-1) return {};
        int discarding=seat==-1 ? owing:seat;
        for(int r=0;r<5;++r) if(state.get(Block::hands)[discarding*5+r]>0) out.push_back({14,r});
        return out;
    }
    if(phase==4) return native_robber_targets(state,actor);
    if(pending_roads(state)) {
        for(Int e:placeable_roads(state,actor)) out.push_back({6,e});return out;
    }
    if(phase==2) {
        out.push_back({0});auto cards=native_card_actions(state,actor,true);
        out.insert(out.end(),cards.begin(),cards.end());return out;
    }
    if((game[FreeRoadsColumn]>0 || can_afford(state,actor,0)) && state.piece_counts[actor][2]<15)
        for(Int e:placeable_roads(state,actor)) out.push_back({6,e});
    if(can_afford(state,actor,1) && state.piece_counts[actor][0]<5)
        for(std::size_t v=0;v<state.get(Block::vertex_owner).size();++v)
            if(settlement_placeable(state,actor,static_cast<Int>(v),true)) out.push_back({7,static_cast<Int>(v)});
    if(can_afford(state,actor,2) && state.piece_counts[actor][1]<4)
        for(std::size_t v=0;v<state.get(Block::vertex_owner).size();++v)
            if(city_upgradeable(state,actor,static_cast<Int>(v))) out.push_back({8,static_cast<Int>(v)});
    auto cards=native_card_actions(state,actor);out.insert(out.end(),cards.begin(),cards.end());
    auto ratios=trade_ratios(state,actor);const auto& hand=state.get(Block::hands);const auto& bank=state.get(Block::bank);
    for(int give=0;give<5;++give) for(int receive=0;receive<5;++receive)
        if(give!=receive && hand[actor*5+give]>=ratios[give] && bank[receive]>0) out.push_back({13,give,receive});
    if(!state.get(Block::deck).empty() && can_afford(state,actor,3)) out.push_back({2});
    out.push_back({1});return out;
}

struct NativeDecision {
    std::vector<unsigned char> flat,setup;
    std::vector<std::array<Int,2>> roads;
    std::vector<std::array<Int,10>> discards;
};
inline std::array<std::size_t,15> action_offsets(const ArenaState& state) {
    auto v=state.get(Block::vertex_owner).size(),e=state.get(Block::edge_owner).size();
    auto h=state.get(Block::terrain).size(),p=state.knowledge.common.size();
    std::array<std::size_t,15> sizes{1,1,1,1,v,e,e,v,v,h*(p+1),1,5,15,25,5},at{};
    for(std::size_t i=1;i<at.size();++i) at[i]=at[i-1]+sizes[i-1];return at;
}
inline std::vector<std::array<Int,2>> native_road_sets(const ArenaState& state,int actor) {
    auto first=placeable_roads(state,actor);std::vector<std::array<Int,2>> out;
    if(state.piece_counts[actor][2]>=14) {for(Int e:first) out.push_back({e,-1});return out;}
    bool near=plus(true_points(state,actor),2)>=state.get(Block::rules)[0];
    std::set<std::pair<Int,Int>> seen;
    for(Int e:first) {
        RoadGraph projected{state,e,actor};
        if(near && state.get(Block::state_control)[1]!=actor) {
            auto lengths=state.get(Block::road_lengths);lengths[actor]=longest_road(projected,actor);
            if(award(lengths,state.get(Block::state_control)[1],5)==actor) {out.push_back({e,-1});continue;}
        }
        auto second=ordered_roads(projected,actor);
        if(second.empty()) {out.push_back({e,-1});continue;}
        for(Int e2:second) if(seen.emplace(std::min(e,e2),std::max(e,e2)).second) out.push_back({e,e2});
    }
    return out;
}
inline void native_decision_fill(const ArenaState& state,int seat,NativeDecision& out) {
    state.knowledge.check_seat(seat);
    auto offsets=action_offsets(state);
    out.flat.assign(offsets.back()+5,0);out.roads.clear();out.discards.clear();
    const auto& vertices=state.get(Block::vertex_owner);
    const auto& game=state.get(Block::game_control);
    int phase=static_cast<int>(game[PhaseColumn]),actor=static_cast<int>(game[ActorColumn]);
    // Incidences are vertex-major and preserve each topology edge row order.
    const auto& row_offsets=state.get(Block::vertex_edges_offsets);
    const auto& row_edges=state.get(Block::vertex_edges_values);
    out.setup.assign(row_edges.size(),0);
    if(phase==0) {
        for(std::size_t v=0;v<vertices.size();++v)
            if(can_place_settlement(state,actor,static_cast<Int>(v),false))
                for(Int at=row_offsets[v];at<row_offsets[v+1];++at)
                    out.setup[at]=state.get(Block::edge_owner)[row_edges[at]]==-1;
        return;
    }
    if(phase==3) {
        Int quota=state.get(Block::discard_quota)[seat];std::array<Int,5> given{};
        auto enumerate=[&](auto&& self,int resource,Int left)->void {
            if(resource==5) {
                if(left) return;
                std::array<Int,10> row{};
                for(int r=0;r<5;++r) {row[r]=state.get(Block::hands)[seat*5+r]-given[r];row[r+5]=given[r];}
                out.discards.push_back(row);return;
            }
            Int maximum=std::min(state.get(Block::hands)[seat*5+resource],left);
            for(Int n=0;n<=maximum;++n) {given[resource]=n;self(self,resource+1,left-n);}
        };
        enumerate(enumerate,0,quota);return;
    }
    if(phase==1 || pending_roads(state)) throw std::invalid_argument("piece-at-a-time decision has no contract-9 tensors");
    bool road_card=false;
    for(auto action:native_legal_actions(state,seat)) {
        if(action.type==3) {road_card=true;continue;}
        Int stride=action.type==9 ? static_cast<Int>(state.knowledge.common.size()+1):action.type==13 ? 5:1;
        Int b=action.b;
        if(action.type==9 && b<static_cast<Int>(state.knowledge.common.size())) b=(b-seat+state.knowledge.common.size())%state.knowledge.common.size();
        out.flat[offsets[action.type]+action.a*stride+b]=1;
    }
    if(road_card) out.roads=native_road_sets(state,actor);
}
inline NativeDecision native_decision(const ArenaState& state,int seat) {
    NativeDecision out;native_decision_fill(state,seat,out);return out;
}
} // namespace hexset_native
