// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "state_primitives.hpp"

namespace hexset_native {
enum GameColumn : std::size_t {
    PhaseColumn,ActorColumn,SetupStepColumn,LastSettlementColumn,LastRollColumn,
    CardPlayedColumn,FreeRoadsColumn,ResumeColumn,TurnsColumn,WonByColumn,TradesMadeColumn,
    CounterStepsColumn,TradeEventColumn,TradeModeColumn,TradeRuleColumn,OpenOffersColumn,
    FirstColumn,TurnCapColumn,GatesColumn
};
inline bool pending_roads(const ArenaState& state) {
    const auto& game=state.get(Block::game_control); auto actor=static_cast<int>(game[ActorColumn]);
    return game[FreeRoadsColumn]>0 && state.piece_counts[actor][2]<15 && !placeable_roads(state,actor).empty();
}
inline void require_main(const ArenaState& state) {
    if (state.get(Block::game_control)[PhaseColumn]!=5) throw std::invalid_argument("expected MAIN phase");
    if (pending_roads(state)) throw std::invalid_argument("place remaining free roads first");
}
inline bool locked(const ArenaState& state,int seat) {
    const auto& seats=state.get(Block::locked);
    return std::find(seats.begin(),seats.end(),seat)!=seats.end();
}
inline int next_unlocked(const ArenaState& state,int after) {
    int players=static_cast<int>(state.get(Block::players)[0]);
    for (int step=1;step<=players;++step) {int seat=(after+step)%players;if (!locked(state,seat)) return seat;}
    throw std::invalid_argument("every seat is locked");
}
struct TurnClosure {
    std::vector<Int> dev,fresh,ages,game,robber_allowed;
    void commit(ArenaState& state) {
        state.get(Block::dev_cards)=std::move(dev);state.get(Block::new_dev_cards)=std::move(fresh);
        state.get(Block::age_values)=std::move(ages);state.get(Block::robber_allowed)=std::move(robber_allowed);
        state.get(Block::game_control)=std::move(game);
    }
};
inline TurnClosure prepare_turn_close(const ArenaState& state,std::vector<Int> game) {
    int actor=static_cast<int>(game[ActorColumn]);
    auto dev=state.get(Block::dev_cards),fresh=state.get(Block::new_dev_cards),ages=state.get(Block::age_values);
    const auto& offsets=state.get(Block::age_offsets);
    if (state.get(Block::age_known)[actor])
        for (Int at=offsets[actor];at<offsets[actor+1];++at) ages[at]=plus(ages[at],1);
    for (int card=0;card<5;++card) {dev[actor*5+card]=plus(dev[actor*5+card],fresh[actor*5+card]);fresh[actor*5+card]=0;}
    game[CardPlayedColumn]=0; game[TradesMadeColumn]=0; game[FreeRoadsColumn]=0;game[ResumeColumn]=5;
    game[TurnsColumn]=plus(game[TurnsColumn],1);
    if (game[TurnsColumn]>=game[TurnCapColumn]) game[PhaseColumn]=6;
    else {
        int next=next_unlocked(state,actor); game[ActorColumn]=next;game[PhaseColumn]=2;
        if (true_points(state,next)>=state.get(Block::rules)[0]) {game[WonByColumn]=next;game[PhaseColumn]=6;}
    }
    return {std::move(dev),std::move(fresh),std::move(ages),std::move(game),{-1}};
}
inline void apply_simple(ArenaState& state,int type,Int a,Int b) {
    // Unsupported actions are refused BEFORE phase checks or mutation. There
    // is no hidden Python fallback and this subset is never used in a worker.
    if (type!=1 && type!=8 && type!=13) throw std::invalid_argument("unsupported native game action");
    require_main(state);
    auto game=state.get(Block::game_control);
    int actor=static_cast<int>(game[ActorColumn]);
    if (type==8) {
        if (!can_upgrade_to_city(state,actor,a)) throw std::invalid_argument("cannot upgrade vertex");
        auto economy=prepare_pay(state,actor,2);
        auto knowledge=public_diff_knowledge(state.knowledge,state.get(Block::hands),economy.hands);
        if (plus(true_points(state,actor),1)>=state.get(Block::rules)[0]) {
            game[WonByColumn]=actor;game[PhaseColumn]=6;
        }
        if (game[PhaseColumn]==5) game[TradeEventColumn]=game[TurnsColumn];
        upgrade_to_city(state,actor,a);
        economy.commit(state);state.knowledge=std::move(knowledge);
        state.get(Block::game_control)=std::move(game);
        return;
    }
    if (type==13) {
        if (a<0 || a>=5 || b<0 || b>=5) throw std::invalid_argument("no such resource");
        auto economy=prepare_bank_trade(state,actor,static_cast<int>(a),static_cast<int>(b));
        auto knowledge=public_diff_knowledge(state.knowledge,state.get(Block::hands),economy.hands);
        game[TradeEventColumn]=game[TurnsColumn];
        economy.commit(state);state.knowledge=std::move(knowledge);
        state.get(Block::game_control)=std::move(game);
        return;
    }
    // END_TURN: prepare maturation, age updates, rotation and win/turn-cap
    // changes completely before committing. No board snapshot/copy is needed.
    auto closure=prepare_turn_close(state,std::move(game));closure.commit(state);
}
} // namespace hexset_native
