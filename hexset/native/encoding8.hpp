// SPDX-License-Identifier: GPL-3.0-only
// Native contract-8 (and 8.1) observations: `hexset.encoding.encode` from a
// persistent ArenaState, the same float32 bytes. One packed row per position:
// hexes, vertices, edges, then globals, as `encode_batch` packs a row.
#pragma once
#include "encoding.hpp"

namespace hexset_native {

using U128=unsigned __int128;

inline int bit_length128(U128 x) {int n=0;while (x) {++n;x>>=1;}return n;}
inline U128 checked_mul128(U128 a,U128 b) {
    U128 out;if (__builtin_mul_overflow(a,b,&out)) throw std::overflow_error("hypergeometric weight exceeds 128 bits");
    return out;
}
inline U128 checked_add128(U128 a,U128 b) {
    U128 out;if (__builtin_add_overflow(a,b,&out)) throw std::overflow_error("hypergeometric weight exceeds 128 bits");
    return out;
}
// math.comb(n, k) for 0 <= k <= n: each partial product is itself a binomial.
inline U128 comb128(Int n,Int k) {
    if (k<0 || k>n) return 0;
    k=std::min(k,n-k);U128 out=1;
    for (Int i=1;i<=k;++i) out=checked_mul128(out,static_cast<U128>(n-k+i))/static_cast<U128>(i);
    return out;
}
// a / b correctly rounded to the nearest double, ties to even: Python's
// int / int. Long division to 55 significant bits plus a sticky bit.
inline double ratio128(U128 a,U128 b) {
    if (!b) throw std::invalid_argument("division by zero");
    if (!a) return 0.0;
    U128 q=a/b,rem=a%b;int exp=0;bool sticky=false;
    while (bit_length128(q)>55) {sticky=sticky || (q&1);q>>=1;++exp;}
    while (bit_length128(q)<55) {rem<<=1;q<<=1;if (rem>=b) {rem-=b;q|=1;}--exp;}
    sticky=sticky || rem!=0;
    bool half=(q>>1)&1,beyond=(q&1) || sticky;U128 m=q>>2;
    if (half && (beyond || (m&1))) ++m;
    return std::ldexp(static_cast<double>(m),exp+2);
}

// View.expected_hand: known plus the hidden cards spread in the pool's
// proportions, each resource held to its room; Python's operation order.
inline std::array<double,5> expected_hand8(const VisibleView& view,int seat) {
    std::array<double,5> out{};const auto& known=view.known[seat];Int hidden=view.unknown[seat],size=total(view.pool);
    for (int r=0;r<5;++r) out[r]=static_cast<double>(known[r]);
    if (!hidden || !size) return out;
    if (!view.bounded[seat]) {
        double share=static_cast<double>(hidden)/static_cast<double>(size);
        for (int r=0;r<5;++r) out[r]=static_cast<double>(known[r])+share*static_cast<double>(view.pool[r]);
        return out;
    }
    const auto& room=view.room[seat];std::array<double,5> spread{};
    std::vector<int> open;for (int r=0;r<5;++r) if (room[r]>0) open.push_back(r);
    double left=static_cast<double>(hidden);
    while (!open.empty() && left>1e-9) {
        Int weight=0;for (int r:open) weight+=view.pool[r];
        std::array<double,5> weights{};
        for (int r:open) weights[r]=weight<=0 ? 1.0/static_cast<double>(open.size())
                                              : static_cast<double>(view.pool[r])/static_cast<double>(weight);
        std::vector<int> capped;
        for (int r:open) if (spread[r]+left*weights[r]>=static_cast<double>(room[r])) capped.push_back(r);
        if (capped.empty()) {for (int r:open) spread[r]+=left*weights[r];break;}
        for (int r:capped) {
            left-=static_cast<double>(room[r])-spread[r];spread[r]=static_cast<double>(room[r]);
            open.erase(std::find(open.begin(),open.end(),r));
        }
    }
    for (int r=0;r<5;++r) out[r]=static_cast<double>(known[r])+spread[r];
    return out;
}

// View.expected_counts: every split of the hidden cards within the seat's
// caps weighed by its exact hypergeometric chance, in 128-bit integers.
inline std::array<double,5> expected_counts8(const VisibleView& view,int seat) {
    Int hidden=view.unknown[seat];
    if (!hidden || !total(view.pool) || !view.bounded[seat]) return expected_hand8(view,seat);
    Counts caps{};for (int r=0;r<5;++r) caps[r]=std::min(view.room[seat][r],view.pool[r]);
    U128 all=0;std::array<U128,5> sums{};Counts split{};
    std::function<void(int,Int,U128)> deal=[&](int r,Int left,U128 weight) {
        if (r==4) {
            if (left>caps[r]) return;
            split[r]=left;U128 w=checked_mul128(weight,comb128(view.pool[r],left));all=checked_add128(all,w);
            for (int i=0;i<5;++i) sums[i]=checked_add128(sums[i],checked_mul128(w,static_cast<U128>(split[i])));
            return;
        }
        for (Int n=0;n<=std::min(left,caps[r]);++n) {split[r]=n;deal(r+1,left-n,checked_mul128(weight,comb128(view.pool[r],n)));}
    };
    deal(0,hidden,1);
    if (!all) return expected_hand8(view,seat);
    std::array<double,5> out{};
    for (int r=0;r<5;++r) out[r]=static_cast<double>(view.known[seat][r])+ratio128(sums[r],all);
    return out;
}

// `hexset.game.to_move`: the current player, or in a discard round the
// lowest-indexed seat still owing cards.
inline Int to_move8(const ArenaState& state) {
    const auto& game=state.get(Block::game_control);
    if (game[PhaseColumn]==3) {
        const auto& quotas=state.get(Block::discard_quota);
        for (std::size_t seat=0;seat<quotas.size();++seat) if (quotas[seat]>0) return static_cast<Int>(seat);
    }
    return game[ActorColumn];
}
inline bool contract8(const std::string& contract) {return contract=="8" || contract=="8.1";}
inline std::size_t global8_width(std::size_t p,const std::string& contract) {
    std::size_t o=p-1;
    return 5+5+4+o*5+o+o*3+p+p+p+o+5+1+5+6+1+1+(contract=="8.1" ? 4+11 : 0);
}
inline std::size_t encoded8_width(const ArenaState& state,const std::string& contract="8") {
    if (!contract8(contract)) throw std::invalid_argument("unknown native contract-8 observation contract");
    const auto& fixed=encoding_template(state);std::size_t p=fixed.players;
    return fixed.hexes*7+fixed.vertices*(2*p+6)+fixed.edges*p+global8_width(p,contract);
}

inline void encode8_into(const ArenaState& state,int perspective,const std::string& contract,float* result,std::size_t width) {
    if (!contract8(contract)) throw std::invalid_argument("unknown native contract-8 observation contract");
    state.knowledge.check_seat(perspective);const auto& fixed=encoding_template(state);
    const std::size_t p=fixed.players,h=fixed.hexes,v=fixed.vertices,e=fixed.edges,vw=2*p+6;
    const std::size_t at_vertices=h*7,at_edges=at_vertices+v*vw,at_globals=at_edges+e*p;
    if (width!=at_globals+global8_width(p,contract)) throw std::invalid_argument("native contract-8 destination width differs");
    std::fill_n(result,width,0.0f);
    auto slot=[&](Int seat) {return static_cast<std::size_t>((seat-perspective+static_cast<Int>(p))%static_cast<Int>(p));};
    // Hexes: the producing terrain, pips over 5, the robber.
    const auto& terrain=state.get(Block::terrain);const auto& tokens=state.get(Block::tokens);
    Int robber=state.get(Block::state_control)[0];
    for (std::size_t at=0;at<h;++at) {
        float* hex=result+at*7;
        if (terrain[at]>=0 && terrain[at]<5) hex[terrain[at]]=1.0f;
        Int pip=tokens[at] ? 6-std::abs(7-tokens[at]) : 0;
        hex[5]=static_cast<float>(static_cast<double>(pip)/5.0);
        if (robber==static_cast<Int>(at)) hex[6]=1.0f;
    }
    // Vertices: each seat's building, the ports, each seat's reach.
    const auto& owner=state.get(Block::vertex_owner);const auto& building=state.get(Block::vertex_building);
    const auto& ports=state.get(Block::ports);
    for (std::size_t at=0;at+4<ports.size();at+=5) {
        std::size_t column=p+(ports[at+3]<0 ? 0 : static_cast<std::size_t>(ports[at+3])+1);
        result[at_vertices+static_cast<std::size_t>(ports[at+1])*vw+column]=1.0f;
        result[at_vertices+static_cast<std::size_t>(ports[at+2])*vw+column]=1.0f;
    }
    const auto& distances=reach_all(state);
    if (distances.size()!=p*v) throw std::logic_error("native road feature shape differs");
    for (std::size_t at=0;at<v;++at) {
        float* row=result+at_vertices+at*vw;
        if (owner[at]>=0 && building[at]>0) row[slot(owner[at])]=static_cast<float>(building[at]);
        for (std::size_t i=0;i<p;++i) {
            Int n=distances[((static_cast<std::size_t>(perspective)+i)%p)*v+at];
            row[p+6+i]=n<0 ? 0.0f : 1.0f/(1.0f+static_cast<float>(n));
        }
    }
    const auto& edge_owner=state.get(Block::edge_owner);
    for (std::size_t at=0;at<e;++at) if (edge_owner[at]>=0) result[at_edges+at*p+slot(edge_owner[at])]=1.0f;
    // Globals, in `encoding._global_blocks` order.
    float* out=result+at_globals;std::size_t k=0;
    auto put=[&](double x) {out[k++]=static_cast<float>(x);};
    const auto& game=state.get(Block::game_control);const auto& hands=state.get(Block::hands);
    const auto& mature=state.get(Block::dev_cards);const auto& fresh=state.get(Block::new_dev_cards);
    VisibleView view(state,perspective);Counts unseen=unseen_development(state,perspective);
    for (int r=0;r<5;++r) put(static_cast<double>(hands[perspective*5+r])/10.0);
    for (int c=0;c<5;++c) put(static_cast<double>(plus(mature[perspective*5+c],fresh[perspective*5+c]))/5.0);
    for (int c:{0,2,3,4}) put(game[CardPlayedColumn] ? 0.0 : static_cast<double>(mature[perspective*5+c])/5.0);
    for (std::size_t i=1;i<p;++i) {
        auto hand=expected_counts8(view,static_cast<int>((perspective+i)%p));
        for (int r=0;r<5;++r) out[k++]=static_cast<float>(hand[r])/10.0f;
    }
    auto held=[&](int seat) {return plus(public_pile_size(state,Block::dev_cards,seat),public_pile_size(state,Block::new_dev_cards,seat));};
    for (std::size_t i=1;i<p;++i) put(static_cast<double>(held(static_cast<int>((perspective+i)%p)))/5.0);
    // `encoding.dev_age_rows`: the three oldest held ages, else AGE_FILL per card held, in float32.
    const auto& offsets=state.get(Block::age_offsets);const auto& values=state.get(Block::age_values);
    constexpr std::array<double,3> fill{5.2,3.1,2.2};
    for (std::size_t i=1;i<p;++i) {
        int seat=static_cast<int>((perspective+i)%p);std::array<float,3> ages{-1.0f,-1.0f,-1.0f};
        if (!state.get(Block::age_known)[seat]) {
            for (Int a=0;a<std::min<Int>(held(seat),3);++a) ages[a]=static_cast<float>(fill[a]);
        } else {
            std::vector<Int> list(values.begin()+offsets[seat],values.begin()+offsets[seat+1]);
            std::sort(list.begin(),list.end(),std::greater<Int>());
            for (std::size_t a=0;a<std::min<std::size_t>(list.size(),3);++a) ages[a]=static_cast<float>(list[a]);
        }
        for (float a:ages) out[k++]=a<0.0f ? 0.0f : (a+1.0f)/20.0f;
    }
    float target=contract=="8.1" ? static_cast<float>(state.get(Block::rules)[0]) : 10.0f;
    for (std::size_t i=0;i<p;++i) out[k++]=static_cast<float>(public_points(state,static_cast<int>((perspective+i)%p)))/target;
    const auto& roads=roads_to_take_all(state);
    auto close=[](Int n) {return n<0 ? 0.0 : 1.0/(1.0+static_cast<double>(n));};
    for (std::size_t i=0;i<p;++i) put(close(roads[(perspective+i)%p]));
    auto knights=visible_knights_needed(state,perspective,unseen[0]);
    for (std::size_t i=0;i<p;++i) put(close(knights[(perspective+i)%p]));
    std::size_t mover=slot(to_move8(state));
    for (std::size_t i=1;i<p;++i) put(mover==i ? 1.0 : 0.0);
    for (int r=0;r<5;++r) put(static_cast<double>(state.get(Block::bank)[r])/19.0);
    put(static_cast<double>(state.get(Block::deck).size())/25.0);
    Int left=total(unseen);
    for (int c=0;c<5;++c) put(left ? static_cast<double>(unseen[c])/static_cast<double>(left) : 0.0);
    for (Int phase=0;phase<6;++phase) put(game[PhaseColumn]==phase ? 1.0 : 0.0);
    put(static_cast<double>(game[FreeRoadsColumn])/2.0);
    put(std::min(static_cast<double>(game[TurnsColumn])/200.0,1.0));
    if (contract=="8.1") {
        const auto& settings=state.get(Block::rules);
        put(static_cast<double>(settings[0])/10.0);put(static_cast<double>(settings[1])/10.0);
        put(settings[2] ? 1.0 : 0.0);put(settings[3] ? 1.0 : 0.0);
        auto odds=encoding_roll_odds(state,false);for (double x:odds) put(x);
    }
    if (k!=global8_width(p,contract)) throw std::logic_error("native contract-8 globals width differs");
}
inline std::vector<float> encode8(const ArenaState& state,int perspective,const std::string& contract="8") {
    std::vector<float> result(encoded8_width(state,contract));
    encode8_into(state,perspective,contract,result.data(),result.size());
    return result;
}

// `hexset.onnx_record`'s record of one position for `perspective`: every
// field but the action mask, int64 but for `expected_hand`, `dev_ages` and
// `roll_odds` (float32), flattened, board-seat order throughout.
struct NativeRecord {
    std::vector<std::pair<std::string,std::vector<Int>>> ints;
    std::vector<std::pair<std::string,std::vector<float>>> floats;
};
inline NativeRecord record8(const ArenaState& state,int perspective) {
    state.knowledge.check_seat(perspective);
    const auto& fixed=encoding_template(state);const std::size_t p=fixed.players,v=fixed.vertices;
    const auto& game=state.get(Block::game_control);const auto& control=state.get(Block::state_control);
    const auto& mature=state.get(Block::dev_cards);const auto& fresh=state.get(Block::new_dev_cards);
    NativeRecord out;auto& I=out.ints;auto& F=out.floats;
    auto copy=[&](Block block) {const auto& b=state.get(block);return std::vector<Int>(b.begin(),b.end());};
    std::vector<Int> ports(v,-1);const auto& data=state.get(Block::ports);
    for (std::size_t at=0;at+4<data.size();at+=5) {
        Int code=data[at+3]<0 ? 0 : data[at+3]+1;ports[data[at+1]]=code;ports[data[at+2]]=code;
    }
    I.push_back({"terrain",copy(Block::terrain)});I.push_back({"token",copy(Block::tokens)});I.push_back({"port_code",ports});
    I.push_back({"robber",{control[0]}});
    I.push_back({"vertex_owner",copy(Block::vertex_owner)});I.push_back({"vertex_building",copy(Block::vertex_building)});
    I.push_back({"edge_owner",copy(Block::edge_owner)});I.push_back({"bank",copy(Block::bank)});
    I.push_back({"knights_played",copy(Block::knights_played)});
    std::vector<Int> awards(p,0);
    for (std::size_t s=0;s<p;++s) awards[s]=(control[1]==static_cast<Int>(s) ? 2 : 0)+(control[2]==static_cast<Int>(s) ? 2 : 0);
    I.push_back({"award_points",awards});
    I.push_back({"longest_road_holder",{control[1]}});I.push_back({"largest_army_holder",{control[2]}});
    I.push_back({"phase",{game[PhaseColumn]}});I.push_back({"free_roads",{game[FreeRoadsColumn]}});
    I.push_back({"deck_size",{static_cast<Int>(state.get(Block::deck).size())}});I.push_back({"turns",{game[TurnsColumn]}});
    const auto& roads=roads_to_take_all(state);I.push_back({"roads_to_take",std::vector<Int>(roads.begin(),roads.end())});
    I.push_back({"to_move",{to_move8(state)}});I.push_back({"perspective",{perspective}});
    const auto& hands=state.get(Block::hands);
    I.push_back({"own_hand",std::vector<Int>(hands.begin()+perspective*5,hands.begin()+perspective*5+5)});
    std::vector<Int> sizes(p),devs(p),own(5),matured(5),known,unknown(p);
    for (std::size_t s=0;s<p;++s) {
        sizes[s]=public_pile_size(state,Block::hands,static_cast<int>(s));
        devs[s]=plus(public_pile_size(state,Block::dev_cards,static_cast<int>(s)),public_pile_size(state,Block::new_dev_cards,static_cast<int>(s)));
    }
    for (int c=0;c<5;++c) {own[c]=plus(mature[perspective*5+c],fresh[perspective*5+c]);matured[c]=mature[perspective*5+c];}
    I.push_back({"hand_totals",sizes});I.push_back({"own_dev",own});I.push_back({"dev_totals",devs});
    const auto& rows=state.knowledge.visible(perspective);
    for (std::size_t s=0;s<p;++s) {known.insert(known.end(),rows[s].known.begin(),rows[s].known.end());unknown[s]=rows[s].unknown;}
    I.push_back({"ledger_known",known});I.push_back({"ledger_unknown",unknown});
    const auto& reach=reach_all(state);I.push_back({"reach",std::vector<Int>(reach.begin(),reach.end())});
    I.push_back({"own_matured",matured});I.push_back({"dev_card_played",{game[CardPlayedColumn] ? 1 : 0}});
    VisibleView view(state,perspective);std::vector<float> expected;
    for (std::size_t s=0;s<p;++s) {auto hand=expected_counts8(view,static_cast<int>(s));for (double x:hand) expected.push_back(static_cast<float>(x));}
    F.push_back({"expected_hand",expected});
    const auto& offsets=state.get(Block::age_offsets);const auto& values=state.get(Block::age_values);
    constexpr std::array<double,3> fill{5.2,3.1,2.2};std::vector<float> ages;
    for (std::size_t s=0;s<p;++s) {
        std::array<float,3> row{-1.0f,-1.0f,-1.0f};
        if (!state.get(Block::age_known)[s]) {
            for (Int a=0;a<std::min<Int>(devs[s],3);++a) row[a]=static_cast<float>(fill[a]);
        } else {
            std::vector<Int> list(values.begin()+offsets[s],values.begin()+offsets[s+1]);
            std::sort(list.begin(),list.end(),std::greater<Int>());
            for (std::size_t a=0;a<std::min<std::size_t>(list.size(),3);++a) row[a]=static_cast<float>(list[a]);
        }
        ages.insert(ages.end(),row.begin(),row.end());
    }
    F.push_back({"dev_ages",ages});
    Counts unseen=unseen_development(state,perspective);
    auto knights=visible_knights_needed(state,perspective,unseen[0]);
    I.push_back({"knights_needed",std::vector<Int>(knights.begin(),knights.begin()+p)});
    I.push_back({"unseen_dev",std::vector<Int>(unseen.begin(),unseen.end())});
    const auto& settings=state.get(Block::rules);
    I.push_back({"winning_points",{settings[0]}});I.push_back({"discard_limit",{settings[1]}});
    I.push_back({"friendly_robber",{settings[2] ? 1 : 0}});I.push_back({"balanced_dice",{settings[3] ? 1 : 0}});
    auto odds=encoding_roll_odds(state,false);std::vector<float> rolled;for (double x:odds) rolled.push_back(static_cast<float>(x));
    F.push_back({"roll_odds",rolled});
    return out;
}
} // namespace hexset_native
