// SPDX-License-Identifier: GPL-3.0-only
// One native dispatch point for persistent lanes. The caller owns the epoch
// increment and record append AFTER a successful transition, as for the
// individual action helpers. No callback runs for a chance-free action.
#pragma once
#include "game_chance.hpp"

namespace hexset_native {
inline void apply_atomic(ArenaState& state,int type,Int a,Int b,Int seat,
                         const RollDraw& roll_draw,const StealDraw& steal_draw) {
    switch (type) {
    case 0: apply_roll(state,roll_draw);break;
    case 1:case 8:case 13: apply_simple(state,type,a,b);break;
    case 2:case 3:case 10:case 11:case 12: apply_cards(state,type,a);break;
    case 4:case 5:case 6:case 7: apply_placement(state,type,a);break;
    case 9: apply_robber(state,a,b,steal_draw);break;
    case 14: apply_discard(state,a,seat);break;
    default: throw std::invalid_argument("unsupported native atomic action");
    }
}
} // namespace hexset_native
