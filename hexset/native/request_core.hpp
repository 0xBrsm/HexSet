// SPDX-License-Identifier: GPL-3.0-only
// Persistent visible request packing over owned native states. No Python calls.
#pragma once
#include "encoding.hpp"
#include "legal_core.hpp"
#include <unordered_set>

namespace hexset_native {
inline int request_actor(const ArenaState& state) {
    if (state.get(Block::game_control)[PhaseColumn]==3) {
        const auto& quotas=state.get(Block::discard_quota);
        for(std::size_t seat=0;seat<quotas.size();++seat) if(quotas[seat]>0) return static_cast<int>(seat);
    }
    return static_cast<int>(state.get(Block::game_control)[ActorColumn]);
}
inline bool request_may_act(const ArenaState& state,int seat) {
    if(state.get(Block::game_control)[PhaseColumn]==3) {
        const auto& quotas=state.get(Block::discard_quota);
        if(std::any_of(quotas.begin(),quotas.end(),[](Int n){return n>0;})) return quotas[seat]>0;
    }
    return seat==request_actor(state);
}
inline bool option_closes(const ArenaState& state,int seat,const std::vector<Int>& types) {
    // Trading/gates are refused at state import. This is decisions.closes's
    // remaining complete contract, evaluated AFTER the option's atomic parts.
    state.knowledge.check_seat(seat);
    if(state.get(Block::game_control)[PhaseColumn]==6) return true;
    bool discard=false;
    for(Int type:types) {
        if(type<0 || type>14) throw std::invalid_argument("unknown atomic type in decision closure");
        if(type==0 || type==1 || type==2 || type==9 || type==11) return true;
        discard=discard || type==14;
    }
    if(discard) return state.get(Block::game_control)[PhaseColumn]!=3 || state.get(Block::discard_quota)[seat]<=0;
    return !request_may_act(state,seat) || request_actor(state)!=seat;
}

struct RequestBuffers {
    std::string contract;
    std::size_t width=0,flat_width=0,setup_width=0,players=0;
    std::vector<NativeDecision> scratch;
    std::vector<Int> ids,road_offsets,discard_offsets;
    std::vector<float> observations;
    std::vector<unsigned char> flat,setup;
    std::vector<std::int16_t> roads;
    std::vector<std::int8_t> discards;
    std::vector<std::uint64_t> state_epochs;
    std::vector<const ArenaState*> imported_states;
    std::uint64_t epoch=0;
    bool outstanding=false;

    RequestBuffers(const std::vector<const ArenaState*>& states,std::string name):contract(std::move(name)) {
        if(states.empty()) throw std::invalid_argument("a native request group requires at least one lane");
        const auto& first=*states.front();
        // A request batch counts every dealt seat: a contract-9d lane is a two-seat table.
        for(auto* state:states)
            if(duel_contract(contract) && state->knowledge.common.size()!=2)
                throw std::invalid_argument("a native request group holds contract 9d at two-seat tables only");
        width=encoded9_width(first,contract);players=first.knowledge.common.size();
        flat_width=action_offsets(first).back()+5;setup_width=first.get(Block::vertex_edges_values).size();
        for(auto* state:states) {
            if(encoded9_width(*state,contract)!=width || state->knowledge.common.size()!=players ||
               state->get(Block::hexes)!=first.get(Block::hexes) || state->get(Block::edges)!=first.get(Block::edges) ||
               state->get(Block::vertex_edges_offsets)!=first.get(Block::vertex_edges_offsets) ||
               state->get(Block::vertex_edges_values)!=first.get(Block::vertex_edges_values))
                throw std::invalid_argument("native request group mixes topology or player count");
        }
        scratch.resize(states.size());
        imported_states=states;
    }
    std::size_t rows() const {return ids.size()/7;}
    void fill(const std::vector<const ArenaState*>& states,const std::vector<Int>& identifiers) {
        if(states!=imported_states) throw std::invalid_argument("a native lane was replaced; import a new request group");
        if(outstanding) throw std::invalid_argument("answer or explicitly finish outstanding native requests first");
        if(identifiers.size()%7) throw std::invalid_argument("native request identity needs seven columns");
        std::unordered_set<Int> lanes;
        for(std::size_t at=0;at<identifiers.size();at+=7) {
            Int lane=identifiers[at+1],seat=identifiers[at+2];
            if(lane<0 || static_cast<std::size_t>(lane)>=states.size() || !lanes.insert(lane).second)
                throw std::invalid_argument("native request lane is invalid or repeated");
            const auto& state=*states[lane];state.knowledge.check_seat(static_cast<int>(seat));
            if(seat<0 || static_cast<std::size_t>(seat)>=players || !request_may_act(state,static_cast<int>(seat)) ||
               state.get(Block::game_control)[PhaseColumn]==6)
                throw std::invalid_argument("native request seat is inactive");
            for(int col:{0,3,4,5,6}) if(identifiers[at+col]<0) throw std::invalid_argument("negative native request identity");
        }
        if(epoch==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("native request epoch exhausted");
        ids=identifiers;auto n=rows();
        observations.resize(n*width);flat.resize(n*flat_width);setup.resize(n*setup_width);
        roads.clear();discards.clear();road_offsets.clear();discard_offsets.clear();state_epochs.clear();
        road_offsets.push_back(0);discard_offsets.push_back(0);
        for(std::size_t row=0;row<n;++row) {
            auto lane=ids[row*7+1];auto seat=static_cast<int>(ids[row*7+2]);const auto& state=*states[lane];
            auto& decision=scratch[lane];native_decision_fill(state,seat,decision);
            encode9_into(state,seat,contract,observations.data()+row*width,width);
            if(decision.flat.size()!=flat_width || decision.setup.size()!=setup_width) throw std::logic_error("native decision shape differs");
            std::copy(decision.flat.begin(),decision.flat.end(),flat.begin()+row*flat_width);
            std::copy(decision.setup.begin(),decision.setup.end(),setup.begin()+row*setup_width);
            for(const auto& pair:decision.roads) for(Int edge:pair) {
                if(edge < -1 || edge>std::numeric_limits<std::int16_t>::max()) throw std::overflow_error("road candidate does not fit request contract");
                roads.push_back(static_cast<std::int16_t>(edge));
            }
            for(const auto& candidate:decision.discards) for(Int count:candidate) {
                if(count<0 || count>std::numeric_limits<std::int8_t>::max()) throw std::overflow_error("discard candidate does not fit request contract");
                discards.push_back(static_cast<std::int8_t>(count));
            }
            road_offsets.push_back(static_cast<Int>(roads.size()/2));discard_offsets.push_back(static_cast<Int>(discards.size()/10));
            state_epochs.push_back(state.epoch);
        }
        ++epoch;outstanding=true;
    }
    void check(const std::vector<const ArenaState*>& states,std::uint64_t answer_epoch) const {
        if(states!=imported_states) throw std::invalid_argument("a native lane was replaced after its request");
        if(!outstanding || answer_epoch!=epoch) throw std::invalid_argument("stale native request epoch");
        for(std::size_t row=0;row<rows();++row)
            if(states[ids[row*7+1]]->epoch!=state_epochs[row]) throw std::invalid_argument("native lane changed after its request");
    }
};
} // namespace hexset_native
