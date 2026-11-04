// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <functional>
#include "game_cards.hpp"

namespace hexset_native {
using RollDraw=std::function<Int(int)>;
using StealDraw=std::function<std::optional<Int>(const Counts&)>;
inline Int referee_hand_size(const ArenaState& state,int seat) {
    state.knowledge.check_seat(seat);Int total=0;
    for (int r=0;r<5;++r) total=plus(total,state.get(Block::hands)[seat*5+r]);
    return total;
}
inline std::vector<Int> robber_occupants(const ArenaState& state,Int hex) {
    if (hex==-1) return {};
    if (hex<0 || static_cast<std::uint64_t>(hex)>=state.get(Block::terrain).size()) throw std::invalid_argument("no such hex");
    std::vector<Int> result;
    for (Int vertex:adjacent(state,Block::hex_vertices_offsets,Block::hex_vertices_values,hex)) {
        Int owner=state.get(Block::vertex_owner)[vertex];
        if (owner>=0 && std::find(result.begin(),result.end(),owner)==result.end()) result.push_back(owner);
    }
    std::sort(result.begin(),result.end());return result;
}
inline std::vector<Int> robber_victims(const ArenaState& state,Int hex,int thief) {
    state.knowledge.check_seat(thief);std::vector<Int> result;
    for (Int seat:robber_occupants(state,hex)) if (seat!=thief && referee_hand_size(state,static_cast<int>(seat))>0) result.push_back(seat);
    return result;
}
inline std::vector<Int> robber_targets(const ArenaState& state,int thief) {
    state.knowledge.check_seat(thief);std::vector<Int> legal;
    const auto& terrain=state.get(Block::terrain);const auto& explicit_allowed=state.get(Block::robber_allowed);
    bool supplied=explicit_allowed!=std::vector<Int>{-1};
    for (std::size_t h=0;h<terrain.size();++h)
        if (static_cast<Int>(h)!=state.get(Block::state_control)[0] && terrain[h]!=6 &&
            (!supplied || std::find(explicit_allowed.begin(),explicit_allowed.end(),static_cast<Int>(h))!=explicit_allowed.end()))
            legal.push_back(static_cast<Int>(h));
    if (supplied || !state.get(Block::rules)[2]) return legal;
    std::vector<Int> protected_seats,allowed;
    for (int seat=0;seat<static_cast<int>(state.knowledge.common.size());++seat)
        if (seat!=thief && public_points(state,seat)<=2) protected_seats.push_back(seat);
    if (protected_seats.empty()) return legal;
    for (Int h:legal) {
        bool safe=true;
        for (Int seat:robber_occupants(state,h))
            if (std::find(protected_seats.begin(),protected_seats.end(),seat)!=protected_seats.end()) {safe=false;break;}
        if (safe) allowed.push_back(h);
    }
    return allowed.empty() ? legal : allowed;
}
struct DiceMutation {
    std::vector<Int> control,left,recent,sevens;
    explicit DiceMutation(const ArenaState& state):control(state.get(Block::dice_control)),left(state.get(Block::dice_left)),
        recent(state.get(Block::dice_recent)),sevens(state.get(Block::dice_sevens)) {}
    void rolled(int actor,Int total) {
        if (!control[0]) return;
        --left[total-2];--control[1];recent.push_back(total);
        if (recent.size()>5) recent.erase(recent.begin(),recent.end()-5);
        std::size_t at=0;while (at<sevens.size() && sevens[at]<actor) at+=2;
        if (at==sevens.size() || sevens[at]!=actor) {sevens.insert(sevens.begin()+at,{actor,0});}
        if (total==7) {sevens[at+1]=plus(sevens[at+1],1);control[3]=control[2]==actor ? plus(control[3],1) : 1;control[2]=actor;}
        if (control[1]<=12) {left={1,2,3,4,5,6,5,4,3,2,1};control[1]=36;}
    }
    void commit(ArenaState& state) {
        state.get(Block::dice_control)=std::move(control);state.get(Block::dice_left)=std::move(left);
        state.get(Block::dice_recent)=std::move(recent);state.get(Block::dice_sevens)=std::move(sevens);
    }
};
inline void apply_roll(ArenaState& state,const RollDraw& draw) {
    auto game=state.get(Block::game_control);int actor=static_cast<int>(game[ActorColumn]);
    if (game[PhaseColumn]!=2) throw std::invalid_argument("expected ROLL phase");
    if (pending_roads(state)) throw std::invalid_argument("place remaining free roads before rolling");
    Int roll=draw(actor);
    if (roll<2 || roll>12) throw std::invalid_argument("two dice cannot produce this total");
    game[FreeRoadsColumn]=0;game[LastRollColumn]=roll;
    DiceMutation dice(state);dice.rolled(actor,roll);
    std::optional<Distribution> distribution;std::optional<Knowledge> knowledge;auto quota=state.get(Block::discard_quota);
    if (roll==7) {
        for (int seat=0;seat<static_cast<int>(quota.size());++seat) {
            Int held=referee_hand_size(state,seat);quota[seat]=!locked(state,seat) && held>state.get(Block::rules)[1] ? held/2 : 0;
        }
        game[ResumeColumn]=5;game[PhaseColumn]=std::any_of(quota.begin(),quota.end(),[](Int n){return n!=0;}) ? 3 : 4;
    } else {
        distribution=prepare_distribution(state,roll);
        knowledge=public_diff_knowledge(state.knowledge,state.get(Block::hands),distribution->hands);
        game[PhaseColumn]=5;game[TradeEventColumn]=game[TurnsColumn];
    }
    dice.commit(state);if (distribution) distribution->commit(state);if (knowledge) state.knowledge=std::move(*knowledge);
    state.get(Block::discard_quota)=std::move(quota);state.get(Block::game_control)=std::move(game);
}
inline void apply_robber(ArenaState& state,Int target,Int victim_slot,const StealDraw& draw) {
    auto game=state.get(Block::game_control);int actor=static_cast<int>(game[ActorColumn]);
    if (game[PhaseColumn]!=4) throw std::invalid_argument("expected ROBBER phase");
    auto allowed=robber_targets(state,actor);
    if (std::find(allowed.begin(),allowed.end(),target)==allowed.end()) throw std::invalid_argument("the robber cannot move to this hex");
    auto victims=robber_victims(state,target,actor);
    if (victim_slot<0) throw std::invalid_argument("no such robber victim slot");
    Int victim=victim_slot>=static_cast<Int>(state.knowledge.common.size()) ? -1 : victim_slot;
    if (victim==-1 ? !victims.empty() : std::find(victims.begin(),victims.end(),victim)==victims.end())
        throw std::invalid_argument("robber victim is not one of this hex's seats with cards");
    auto hands=state.get(Block::hands);std::optional<Knowledge> knowledge;
    if (victim!=-1) {
        Counts held{};std::copy_n(hands.begin()+victim*5,5,held.begin());
        auto stolen=draw(held);
        if (stolen) {
            if (*stolen<0 || *stolen>=5 || hands[victim*5+*stolen]<1) throw std::invalid_argument("chance returned an unavailable resource");
            --hands[victim*5+*stolen];hands[actor*5+*stolen]=plus(hands[actor*5+*stolen],1);
            knowledge=state.knowledge;knowledge->steal(actor,static_cast<int>(victim),static_cast<int>(*stolen));
        }
    }
    game[PhaseColumn]=game[ResumeColumn];if (game[PhaseColumn]==5) game[TradeEventColumn]=game[TurnsColumn];
    state.get(Block::hands)=std::move(hands);if (knowledge) state.knowledge=std::move(*knowledge);
    state.get(Block::state_control)[0]=target;state.get(Block::robber_allowed)={-1};state.get(Block::game_control)=std::move(game);
}
inline void apply_discard(ArenaState& state,Int resource,Int seat) {
    auto game=state.get(Block::game_control);auto quota=state.get(Block::discard_quota);
    if (game[PhaseColumn]!=3) throw std::invalid_argument("expected DISCARD phase");
    if (seat==-1) {auto found=std::find_if(quota.begin(),quota.end(),[](Int n){return n>0;});seat=std::distance(quota.begin(),found);}
    if (seat<0 || static_cast<std::uint64_t>(seat)>=quota.size()) throw std::invalid_argument("no such discarding seat");
    if (resource<0 || resource>=5) throw std::invalid_argument("no such resource");
    EconomyChange economy{state.get(Block::hands),state.get(Block::bank)};
    if (quota[seat]<1 || economy.hands[seat*5+resource]<1) throw std::invalid_argument("this seat cannot discard this resource");
    --economy.hands[seat*5+resource];economy.bank[resource]=plus(economy.bank[resource],1);--quota[seat];
    auto knowledge=state.knowledge;knowledge.spend(static_cast<int>(seat),static_cast<int>(resource),1);
    std::optional<TurnClosure> closure;
    if (std::none_of(quota.begin(),quota.end(),[](Int n){return n!=0;})) {
        if (locked(state,static_cast<int>(game[ActorColumn]))) closure=prepare_turn_close(state,game);
        else game[PhaseColumn]=4;
    }
    economy.commit(state);state.knowledge=std::move(knowledge);state.get(Block::discard_quota)=std::move(quota);
    if (closure) closure->commit(state);else state.get(Block::game_control)=std::move(game);
}
} // namespace hexset_native
