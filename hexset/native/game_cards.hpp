// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "game_placement.hpp"

namespace hexset_native {
struct CardMutation {
    std::vector<Int> dev,fresh,played,knights,ages,offsets,deck,deck_enum;
    explicit CardMutation(const ArenaState& state):dev(state.get(Block::dev_cards)),fresh(state.get(Block::new_dev_cards)),
        played(state.get(Block::dev_cards_played)),knights(state.get(Block::knights_played)),
        ages(state.get(Block::age_values)),offsets(state.get(Block::age_offsets)),deck(state.get(Block::deck)),deck_enum(state.get(Block::deck_enum)) {}
    void commit(ArenaState& state) {
        state.get(Block::dev_cards)=std::move(dev);state.get(Block::new_dev_cards)=std::move(fresh);
        state.get(Block::dev_cards_played)=std::move(played);state.get(Block::knights_played)=std::move(knights);
        state.get(Block::age_values)=std::move(ages);state.get(Block::age_offsets)=std::move(offsets);
        state.get(Block::deck)=std::move(deck);state.get(Block::deck_enum)=std::move(deck_enum);
    }
    void spend(const ArenaState& state,int actor,int card) {
        if (card==1 || card<0 || card>=5 || dev[actor*5+card]<=0) throw std::invalid_argument("cannot play development card");
        if (state.get(Block::age_known)[actor]) {
            Int fresh_count=0;for (int k=0;k<5;++k) fresh_count=plus(fresh_count,fresh[actor*5+k]);
            Int within=offsets[actor+1]-offsets[actor]-fresh_count-1;
            if (within>=0) {ages.erase(ages.begin()+offsets[actor]+within);for (std::size_t s=static_cast<std::size_t>(actor)+1;s<offsets.size();++s) --offsets[s];}
        }
        --dev[actor*5+card];
    }
};
inline void check_turn_card(const ArenaState& state) {
    const auto& game=state.get(Block::game_control);
    if (game[PhaseColumn]!=2 && game[PhaseColumn]!=5) throw std::invalid_argument("cannot play card in this phase");
    if (game[PhaseColumn]==5) require_main(state);
    if (game[CardPlayedColumn]) throw std::invalid_argument("only one development card per turn");
}
inline void apply_cards(ArenaState& state,int type,Int argument) {
    if (type!=2 && type!=3 && type!=10 && type!=11 && type!=12) throw std::invalid_argument("unsupported native card action");
    auto game=state.get(Block::game_control);int actor=static_cast<int>(game[ActorColumn]);bool was_main=game[PhaseColumn]==5;
    CardMutation cards(state);std::optional<EconomyChange> economy;std::optional<Knowledge> knowledge;
    Int army_holder=state.get(Block::state_control)[2];
    bool pending_after_card=false;
    if (type==2) {
        require_main(state);
        if (cards.deck.empty()) throw std::invalid_argument("empty development deck");
        economy=prepare_pay(state,actor,3);
        Int card=cards.deck.back();cards.deck.pop_back();cards.deck_enum.pop_back();
        cards.fresh[actor*5+card]=plus(cards.fresh[actor*5+card],1);
        if (state.get(Block::age_known)[actor]) {
            cards.ages.insert(cards.ages.begin()+cards.offsets[actor+1],0);
            for (std::size_t s=static_cast<std::size_t>(actor)+1;s<cards.offsets.size();++s) ++cards.offsets[s];
        }
        knowledge=public_diff_knowledge(state.knowledge,state.get(Block::hands),economy->hands);
        if (plus(true_points(state,actor),card==1 ? 1 : 0)>=state.get(Block::rules)[0]) {game[WonByColumn]=actor;game[PhaseColumn]=6;}
    } else {
        check_turn_card(state);
        int card=type==3 ? 2 : type==10 ? 0 : type==11 ? 4 : 3;
        if (type==3 && (state.piece_counts[actor][2]>=15 || placeable_roads(state,actor).empty()))
            throw std::invalid_argument("road building needs a piece and a legal edge");
        cards.spend(state,actor,card);game[CardPlayedColumn]=1;
        if (type==10) {
            cards.knights[actor]=plus(cards.knights[actor],1);army_holder=award(cards.knights,army_holder,3);
            Int change=(army_holder==actor ? 2 : 0)-(state.get(Block::state_control)[2]==actor ? 2 : 0);
            if (plus(true_points(state,actor),change)>=state.get(Block::rules)[0]) {game[WonByColumn]=actor;game[PhaseColumn]=6;}
            else {game[ResumeColumn]=game[PhaseColumn];game[PhaseColumn]=4;}
        } else {
            cards.played[card]=plus(cards.played[card],1);
            if (type==3) {game[FreeRoadsColumn]=plus(game[FreeRoadsColumn],2);pending_after_card=true;}
            else {
                economy=EconomyChange{state.get(Block::hands),state.get(Block::bank)};
                if (type==11) {
                    if (argument<0 || argument>=5) throw std::invalid_argument("no such resource");
                    Int taken=0;
                    for (std::size_t s=0;s<state.knowledge.common.size();++s) if (static_cast<int>(s)!=actor) {
                        taken=plus(taken,economy->hands[s*5+argument]);economy->hands[s*5+argument]=0;
                    }
                    economy->hands[actor*5+argument]=plus(economy->hands[actor*5+argument],taken);
                } else {
                    if (argument<0 || argument>=15) throw std::invalid_argument("invalid year-of-plenty pair");
                    Counts wanted{};Int at=0;
                    for (int first=0;first<5;++first) for (int second=first;second<5;++second)
                        if (at++==argument) {++wanted[first];++wanted[second];}
                    for (int r=0;r<5;++r) {
                        if (wanted[r]>economy->bank[r]) throw std::invalid_argument("bank cannot supply year of plenty");
                        economy->hands[actor*5+r]=plus(economy->hands[actor*5+r],wanted[r]);economy->bank[r]-=wanted[r];
                    }
                }
                knowledge=public_diff_knowledge(state.knowledge,state.get(Block::hands),economy->hands);
                if (type==11) for (std::size_t s=0;s<state.knowledge.common.size();++s)
                    if (static_cast<int>(s)!=actor) knowledge->emptied(static_cast<int>(s),static_cast<int>(argument));
            }
        }
    }
    if (was_main && game[PhaseColumn]==5 && !pending_after_card) game[TradeEventColumn]=game[TurnsColumn];
    cards.commit(state);if (economy) economy->commit(state);if (knowledge) state.knowledge=std::move(*knowledge);
    state.get(Block::state_control)[2]=army_holder;state.get(Block::game_control)=std::move(game);
}
} // namespace hexset_native
