// SPDX-License-Identifier: GPL-3.0-only
// Native observation packing over persistent ArenaState. No Python
// callback, referee snapshot, hidden-opponent card identity or game copy.
#pragma once
#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include "state_primitives.hpp"
#include "game_simple.hpp"
#include "roads_core.hpp"

namespace hexset_native {

struct EncodingCache {
    std::size_t players,hexes,vertices,edges;
    std::vector<float> hex9,number,ports;
    explicit EncodingCache(const ArenaState& state)
        : players(state.get(Block::players)[0]),hexes(state.get(Block::terrain).size()),
          vertices(state.get(Block::vertex_owner).size()),edges(state.get(Block::edge_owner).size()),
          hex9(hexes*7,0.0f),number(hexes,0.0f),ports(vertices*7,0.0f) {
        const auto& terrain=state.get(Block::terrain); const auto& tokens=state.get(Block::tokens);
        for (std::size_t h=0;h<hexes;++h) {
            if (terrain[h]>=0 && terrain[h]<5) hex9[h*7+terrain[h]]=1.0f;
            Int pip=tokens[h] ? 6-std::abs(7-tokens[h]) : 0;
            hex9[h*7+5]=static_cast<float>(static_cast<double>(pip)/10.0);
            number[h]=static_cast<float>(tokens[h])/12.0f;
        }
        const auto& data=state.get(Block::ports);
        for (std::size_t at=0;at<data.size();at+=5) {
            std::size_t column=data[at+3]<0 ? 0 : static_cast<std::size_t>(data[at+3])+1;
            ports[static_cast<std::size_t>(data[at+1])*7+column]=1.0f;
            ports[static_cast<std::size_t>(data[at+2])*7+column]=1.0f;
        }
    }
};

inline const EncodingCache& encoding_template(const ArenaState& state) {
    if (!state.encoding_cache) state.encoding_cache=std::make_shared<const EncodingCache>(state);
    return *state.encoding_cache;
}
inline Int public_pile_size(const ArenaState& state,Block block,int seat) {
    // Opponent rows are read ONLY through this total, never by card kind.
    Int size=0; const auto& pile=state.get(block);
    for (int r=0;r<5;++r) size=plus(size,pile[seat*5+r]);
    return size;
}

struct VisibleView {
    std::array<Counts,4> known{},room{};
    std::array<Int,4> sizes{},unknown{};
    // Per seat, whether its upper bounds bind (`View.bounded`).
    std::array<bool,4> bounded{};
    Counts pool{};
    explicit VisibleView(const ArenaState& state,int perspective) {
        state.knowledge.check_seat(perspective);
        int players=static_cast<int>(state.get(Block::players)[0]);
        const auto& rows=state.knowledge.visible(perspective);
        for (int seat=0;seat<players;++seat) {
            sizes[seat]=public_pile_size(state,Block::hands,seat);
            if (seat==perspective) {
                std::copy_n(state.get(Block::hands).begin()+seat*5,5,known[seat].begin());
                continue;
            }
            known[seat]=rows[seat].known;
            Int excess=total(known[seat])-sizes[seat];
            while (excess>0) {
                auto largest=std::max_element(known[seat].begin(),known[seat].end());
                Int taken=std::min(excess,*largest);*largest-=taken;excess-=taken;
            }
            Int hidden=sizes[seat]-total(known[seat]);unknown[seat]=hidden;
            Counts gaps{};
            for (int r=0;r<5;++r) gaps[r]=rows[seat].most[r]-known[seat][r];
            bool bounded=hidden>0 && *std::min_element(gaps.begin(),gaps.end())<hidden;
            if (bounded) {
                for (int r=0;r<5;++r) room[seat][r]=std::clamp<Int>(gaps[r],0,hidden);
                if (total(room[seat])<hidden) bounded=false;
            }
            if (!bounded) room[seat].fill(hidden);
            this->bounded[seat]=bounded;
        }
        const auto& bank=state.get(Block::bank);
        for (int r=0;r<5;++r) {
            Int certified=0;for (const auto& row:known) certified=plus(certified,row[r]);
            pool[r]=std::max<Int>(0,19-bank[r]-certified);
        }
        Int hidden=0;for (Int count:unknown) hidden=plus(hidden,count);
        Int population=total(pool),deficit=hidden-population;
        if (deficit>0) {
            // view._padded: proportional largest remainders, stable resource
            // order for tied fractions, with Python's double operation order.
            std::array<double,5> shares{};Counts whole{};
            for (int r=0;r<5;++r) {
                double weight=population ? static_cast<double>(pool[r])/static_cast<double>(population) : 1.0/5.0;
                shares[r]=weight*static_cast<double>(deficit);whole[r]=static_cast<Int>(shares[r]);
                pool[r]=plus(pool[r],whole[r]);
            }
            std::array<int,5> order{0,1,2,3,4};
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){
                return shares[a]-static_cast<double>(whole[a])>shares[b]-static_cast<double>(whole[b]);
            });
            Int left=deficit-total(whole);
            for (Int at=0;at<std::min<Int>(left,5);++at) ++pool[order[at]];
        }
    }
};

inline Counts unseen_development(const ArenaState& state,int perspective) {
    state.knowledge.check_seat(perspective);
    Counts result{14,5,2,2,2};
    const auto& mature=state.get(Block::dev_cards); const auto& fresh=state.get(Block::new_dev_cards);
    const auto& played=state.get(Block::dev_cards_played);
    for (int card=0;card<5;++card) result[card]-=plus(plus(mature[perspective*5+card],fresh[perspective*5+card]),played[card]);
    for (Int n:state.get(Block::knights_played)) result[0]-=n;
    for (auto& n:result) n=std::max<Int>(0,n);
    return result;
}
inline std::array<Int,4> visible_knights_needed(const ArenaState& state,int perspective,Int unseen) {
    int players=static_cast<int>(state.get(Block::players)[0]);
    Int holder=state.get(Block::state_control)[2];const auto& played=state.get(Block::knights_played);
    Int target=holder<0 ? 3 : std::max<Int>(3,plus(played[holder],1));
    std::array<Int,4> result{};
    for (int seat=0;seat<players;++seat) {
        if (seat==holder) {result[seat]=0;continue;}
        Int need=std::max<Int>(1,target-played[seat]),can=unseen;
        if (seat==perspective) can=plus(can,plus(state.get(Block::dev_cards)[seat*5],state.get(Block::new_dev_cards)[seat*5]));
        result[seat]=need<=can ? need : -1;
    }
    return result;
}
inline bool encoding_rolled(const ArenaState& state) {
    const auto& game=state.get(Block::game_control);Int phase=game[PhaseColumn];
    if (phase==0 || phase==1 || phase==2) return false;
    return phase!=4 || game[ResumeColumn]!=2;
}
inline std::array<double,11> encoding_roll_odds(const ArenaState& state,bool next_roller) {
    std::array<double,11> result{};
    const auto& control=state.get(Block::dice_control);
    if (!control[0]) {
        for (int i=0;i<11;++i) result[i]=static_cast<double>(6-std::abs(7-(i+2)))/36.0;
        return result;
    }
    Int roller=state.get(Block::game_control)[ActorColumn];
    if (next_roller && encoding_rolled(state)) {
        Int players=state.get(Block::players)[0];const auto& excluded=state.get(Block::locked);
        for (Int step=1;step<=players;++step) {
            Int seat=(roller+step)%players;
            if (std::find(excluded.begin(),excluded.end(),seat)==excluded.end()) {roller=seat;break;}
        }
    }
    // DiceDeck._seven_scale first converts the public tuple to a dictionary.
    std::vector<std::pair<Int,Int>> sevens;
    const auto& counts=state.get(Block::dice_sevens);
    for (std::size_t at=0;at<counts.size();at+=2) {
        auto found=std::find_if(sevens.begin(),sevens.end(),[&](const auto& item){return item.first==counts[at];});
        if (found==sevens.end()) sevens.emplace_back(counts[at],counts[at+1]);else found->second=counts[at+1];
    }
    auto own=std::find_if(sevens.begin(),sevens.end(),[&](const auto& item){return item.first==roller;});
    if (own==sevens.end()) {sevens.emplace_back(roller,0);own=sevens.end()-1;}
    Int rolled_sevens=0;for (const auto& item:sevens) rolled_sevens=plus(rolled_sevens,item.second);
    double balance=1.0;
    if (rolled_sevens>=static_cast<Int>(sevens.size())) {
        double share=static_cast<double>(own->second)/static_cast<double>(rolled_sevens);
        double ideal=1.0/static_cast<double>(sevens.size());balance=1.0+(ideal-share)/ideal;
    }
    double streak=0.4*static_cast<double>(control[3])*(control[2]==roller ? -1.0 : 1.0);
    double scale=std::min(2.0,std::max(0.0,balance+streak));
    if (control[1]<=0) throw std::invalid_argument("nonpositive dice deck size");
    const auto& left=state.get(Block::dice_left);const auto& recent=state.get(Block::dice_recent);
    double mass=0.0;
    for (int i=0;i<11;++i) {
        auto seen=std::count(recent.begin(),recent.end(),i+2);
        double weight=static_cast<double>(left[i])/static_cast<double>(control[1])*
                      std::max(0.0,1.0-0.34*static_cast<double>(seen));
        if (i==5) weight*=scale;
        result[i]=weight;if (left[i]) mass+=weight;
    }
    if (mass>0.0) {for (int i=0;i<11;++i) result[i]=left[i] ? result[i]/mass : 0.0;}
    else {
        Int cards=0;for (Int n:left) cards=plus(cards,n);
        for (int i=0;i<11;++i) result[i]=left[i] ? static_cast<double>(left[i])/static_cast<double>(cards) : 0.0;
    }
    return result;
}
inline float direct_scale(Int n,double scale) {return static_cast<float>(static_cast<double>(n)/scale);}
inline float count_scale(Int n) {return static_cast<float>(n)/10.0f;}
inline float award_close(Int n) {return n<0 ? 0.0f : static_cast<float>(1.0/(1.0+static_cast<double>(n)));}
inline std::array<float,5> encoded_ages(const ArenaState& state,int seat,Int held) {
    std::array<float,5> result{};
    if (!state.get(Block::age_known)[seat]) {
        constexpr std::array<double,5> fill{5.2,3.1,2.2,2.2,2.2};
        for (Int at=0;at<std::min<Int>(held,5);++at) result[at]=static_cast<float>((fill[at]+1.0)/20.0);
    } else {
        const auto& offsets=state.get(Block::age_offsets);const auto& data=state.get(Block::age_values);
        std::array<Int,5> oldest{};std::size_t count=0;
        for (Int at=offsets[seat];at<offsets[seat+1];++at) {
            Int age=data[at];std::size_t position=0;
            while (position<count && oldest[position]>=age) ++position;
            if (position==5) continue;
            for (std::size_t k=std::min<std::size_t>(count,4);k>position;--k) oldest[k]=oldest[k-1];
            oldest[position]=age;if (count<5) ++count;
        }
        for (std::size_t at=0;at<count;++at)
            result[at]=static_cast<float>((static_cast<double>(oldest[at])+1.0)/20.0);
    }
    return result;
}

inline bool duel_contract(const std::string& contract) {return contract=="9d" || contract=="9d.1" || contract=="11d";}
// The seats a two-seat (9d, 9d.1, 11d) row counts: every dealt seat at a
// two-seat table, else the live ones (`seat_frame`).
inline std::size_t duel_live(const ArenaState& state) {
    int dealt=static_cast<int>(state.get(Block::players)[0]);std::size_t live=0;
    for (int seat=0;seat<dealt;++seat) live+=dealt==2 || !locked(state,seat);
    return live;
}

// contract9.seat_production: vertices ascending, each vertex's hexes in topology order, summed in double and
// rounded once, as Python's float accumulation into a float32 row.
inline std::array<double,5> encoded_production(const ArenaState& state,int seat) {
    std::array<double,5> made{};
    const auto& owner=state.get(Block::vertex_owner);const auto& building=state.get(Block::vertex_building);
    const auto& terrain=state.get(Block::terrain);const auto& tokens=state.get(Block::tokens);
    Int robber=state.get(Block::state_control)[0];
    for (std::size_t vertex=0;vertex<owner.size();++vertex) {
        if (owner[vertex]!=seat) continue;
        for (Int hex:adjacent(state,Block::vertex_hexes_offsets,Block::vertex_hexes_values,static_cast<Int>(vertex))) {
            Int kind=terrain[hex],token=tokens[hex];
            if (kind<0 || kind>=5 || hex==robber) continue;
            Int pip=token ? 6-std::abs(7-token) : 0;
            made[kind]+=static_cast<double>(building[vertex]*pip)/36.0;
        }
    }
    return made;
}

// Contract 11d's seat block: my token (35) then the opponent's (27), end to end; its table is the board block
// (the turn count, the next roll's chance of a seven, whether the dice have been thrown).
constexpr std::size_t kMine11d=35,kOther11d=27,kBoard11d=3;
// Contract 11d's hex: terrain 5, the number one-hot over 2-6 and 8-12 (none for the desert), odds, robber.
constexpr std::size_t kHex11d=17;
inline int number_slot11d(Int token) {return token>=2 && token<=6 ? static_cast<int>(token)-2 : token>=8 && token<=12 ? static_cast<int>(token)-3 : -1;}

inline std::size_t encoded9_width(const ArenaState& state,const std::string& contract="9") {
    if (contract!="9" && contract!="9.1" && !duel_contract(contract)) throw std::invalid_argument("unknown native observation contract");
    const auto& fixed=encoding_template(state);
    std::size_t p=fixed.players;
    if (duel_contract(contract)) {
        // A table of two: dealt two seats, or more with all but two retired (`duel_live`).
        if (duel_live(state)!=2) throw std::invalid_argument("contract 9d requires two live seats");
        p=2;
    }
    auto sw=contract=="9d.1" ? 32 : contract=="9d" ? 22 : contract=="9.1" ? 40 : 39;
    std::size_t tw=contract=="11d" ? kBoard11d : contract=="9.1" ? 35 : 20,hw=contract=="11d" ? kHex11d : contract=="9" ? 7 : 8;
    std::size_t seats=contract=="11d" ? kMine11d+kOther11d : p*sw;
    return seats+tw+fixed.hexes*hw+fixed.vertices*8+fixed.edges+p*fixed.vertices;
}

inline void encode9_into(const ArenaState& state,int perspective,const std::string& contract,
                         float* result,std::size_t width) {
    if (contract!="9" && contract!="9.1" && !duel_contract(contract)) throw std::invalid_argument("unknown native observation contract");
    state.knowledge.check_seat(perspective);const auto& fixed=encoding_template(state);
    bool duel=duel_contract(contract),rules=contract=="9.1",economy=contract=="9d.1",two=contract=="11d";
    if (duel && duel_live(state)!=2) throw std::invalid_argument("contract 9d requires two live seats");
    // Row i is board seat frame.seats[i]: every dealt seat from the perspective,
    // or under a two-seat contract the two live seats, a retired seat having no row.
    const std::size_t dealt=fixed.players,h=fixed.hexes,v=fixed.vertices,e=fixed.edges;
    const SeatFrame frame=seat_frame(state,perspective,duel ? 2 : dealt);
    const std::size_t p=frame.size;
    std::size_t sw=duel ? (economy ? 32 : 22) : (rules ? 40 : 39),tw=two ? kBoard11d : (rules ? 35 : 20),
                hw=two ? kHex11d : contract=="9" ? 7 : 8;
    std::size_t table=two ? kMine11d+kOther11d : p*sw,hexes=table+tw,vertices=hexes+h*hw,owner=vertices+v*7,edge=owner+v,
                reach=edge+e;
    if (width!=reach+p*v) throw std::invalid_argument("native observation destination width differs");
    std::fill_n(result,width,0.0f);
    VisibleView view(state,perspective);Counts unseen=unseen_development(state,perspective);
    const auto& roads=roads_to_take_all(state);const auto& distances=reach_all(state);
    if (roads.size()!=dealt || distances.size()!=dealt*v) throw std::logic_error("native road feature shape differs");
    auto knights=visible_knights_needed(state,perspective,unseen[0]);
    const auto& game=state.get(Block::game_control);const auto& mature=state.get(Block::dev_cards);
    const auto& fresh=state.get(Block::new_dev_cards);const auto& quota=state.get(Block::discard_quota);
    constexpr std::array<int,4> playable{0,2,3,4};
    std::array<double,11> odds{};
    if (duel || rules) odds=encoding_roll_odds(state,duel);
    for (std::size_t row=0;row<p;++row) {
        int seat=frame.seats[row];auto* out=result+row*(two ? kMine11d : sw);
        Int old=public_pile_size(state,Block::dev_cards,seat),bought=public_pile_size(state,Block::new_dev_cards,seat);
        Int held=plus(old,bought);bool played=game[CardPlayedColumn] && seat==game[ActorColumn];
        auto ages=encoded_ages(state,seat,held);const auto& pieces=state.piece_counts[seat];
        if (two) {
            // contract9._encode11d: the public columns, then mine or the opponent's own.
            for (int r=0;r<5;++r) out[r]=count_scale(view.known[seat][r]+(seat==perspective ? 0 : view.pool[r]));
            out[5]=direct_scale(public_points(state,seat),static_cast<double>(state.get(Block::rules)[0]));
            out[6]=award_close(roads[seat]);out[7]=award_close(knights[seat]);
            out[8]=direct_scale(5-pieces[0],10.0);out[9]=direct_scale(4-pieces[1],10.0);out[10]=direct_scale(15-pieces[2],10.0);
            auto made=encoded_production(state,seat);auto ratios=trade_ratios(state,seat);
            for (int r=0;r<5;++r) {out[11+r]=static_cast<float>(made[r]);out[16+r]=static_cast<float>(ratios[r])/4.0f;}
            if (row==0) {
                for (int card=0;card<5;++card) out[21+card]=count_scale(plus(mature[seat*5+card],fresh[seat*5+card]));
                for (int at=0;at<4;++at) out[26+at]=count_scale(played ? 0 : mature[seat*5+playable[at]]);
                for (int card=0;card<5;++card) out[30+card]=count_scale(unseen[card]);
            } else {
                out[21]=direct_scale(held,10.0);std::copy(ages.begin(),ages.end(),out+22);
            }
        } else if (duel) {
            out[0]=seat==game[FirstColumn];out[1]=seat==game[ActorColumn];out[2]=seat==perspective;
            for (int r=0;r<5;++r) out[3+r]=count_scale(view.known[seat][r]+(seat==perspective ? 0 : view.pool[r]));
            out[8]=direct_scale(held,10.0);out[9]=direct_scale(bought,10.0);
            std::copy(ages.begin(),ages.end(),out+10);
            out[15]=direct_scale(public_points(state,seat),static_cast<double>(state.get(Block::rules)[0]));
            out[16]=award_close(roads[seat]);out[17]=award_close(knights[seat]);
            out[18]=direct_scale(quota.empty() ? 0 : quota[seat],10.0);
            out[19]=direct_scale(5-pieces[0],10.0);out[20]=direct_scale(4-pieces[1],10.0);out[21]=direct_scale(15-pieces[2],10.0);
            if (economy) {
                auto made=encoded_production(state,seat);auto ratios=trade_ratios(state,seat);
                for (int r=0;r<5;++r) {out[22+r]=static_cast<float>(made[r]);out[27+r]=static_cast<float>(ratios[r])/4.0f;}
            }
        } else {
            out[(seat-game[FirstColumn]+static_cast<Int>(p))%p]=1.0f;
            out[4]=seat==game[ActorColumn];out[5]=seat==perspective;
            for (int r=0;r<5;++r) {
                out[6+r]=count_scale(view.known[seat][r]);
                out[11+r]=count_scale(view.known[seat][r]+view.room[seat][r]);
                Int most=seat==perspective ? plus(mature[seat*5+r],fresh[seat*5+r]) : std::min(held,unseen[r]);
                out[17+r]=count_scale(most);
            }
            out[16]=direct_scale(view.sizes[seat],10.0);out[22]=direct_scale(held,10.0);
            std::copy(ages.begin(),ages.end(),out+23);
            for (int at=0;at<4;++at) out[28+at]=count_scale(played ? 0 :
                (seat==perspective ? mature[seat*5+playable[at]] : std::min(old,unseen[playable[at]])));
            out[32]=direct_scale(public_points(state,seat),rules ? static_cast<double>(state.get(Block::rules)[0]) : 10.0);
            out[33]=award_close(roads[seat]);out[34]=award_close(knights[seat]);
            out[35]=direct_scale(quota.empty() ? 0 : quota[seat],10.0);
            out[36]=direct_scale(5-pieces[0],10.0);out[37]=direct_scale(4-pieces[1],10.0);out[38]=direct_scale(15-pieces[2],10.0);
            if (rules) out[39]=std::find(state.get(Block::locked).begin(),state.get(Block::locked).end(),seat)==state.get(Block::locked).end();
        }
        for (std::size_t vertex=0;vertex<v;++vertex) {
            Int n=distances[seat*v+vertex];
            result[reach+row*v+vertex]=n<0 ? 0.0f : 1.0f/(1.0f+static_cast<float>(n));
        }
    }
    auto* out=result+table;
    if (two) {
        out[0]=direct_scale(game[TurnsColumn],100.0);out[1]=static_cast<float>(odds[5]);out[2]=encoding_rolled(state);
    } else if (duel) {
        for (int card=0;card<5;++card) out[card]=count_scale(plus(mature[perspective*5+card],fresh[perspective*5+card]));
        bool played=game[CardPlayedColumn] && perspective==game[ActorColumn];
        for (int at=0;at<4;++at) out[5+at]=count_scale(played ? 0 : mature[perspective*5+playable[at]]);
        for (int card=0;card<5;++card) out[9+card]=count_scale(unseen[card]);
        out[14]=game[CardPlayedColumn]!=0;out[15]=direct_scale(game[TurnsColumn],100.0);
        out[16]=game[PhaseColumn]==0 || game[PhaseColumn]==1;out[17]=encoding_rolled(state);
        out[18]=game[PhaseColumn]==3 || game[PhaseColumn]==4;out[19]=static_cast<float>(odds[5]);
    } else {
        for (int r=0;r<5;++r) {out[r]=count_scale(state.get(Block::bank)[r]);out[5+r]=count_scale(view.pool[r]);out[11+r]=count_scale(unseen[r]);}
        out[10]=direct_scale(static_cast<Int>(state.get(Block::deck).size()),10.0);out[16]=direct_scale(game[TurnsColumn],100.0);
        out[17]=game[PhaseColumn]==0 || game[PhaseColumn]==1;out[18]=encoding_rolled(state);
        out[19]=game[PhaseColumn]==3 || game[PhaseColumn]==4;
        if (rules) {
            const auto& settings=state.get(Block::rules);out[20]=direct_scale(settings[0],10.0);out[21]=direct_scale(settings[1],10.0);
            out[22]=settings[2]!=0;out[23]=settings[3]!=0;
            for (int i=0;i<11;++i) out[24+i]=static_cast<float>(odds[i]);
        }
    }
    // A piece's owner by row; no owner, or a seat with no row, reads -1.
    auto pointer=[&](Int who) {return who<0 || frame.slot[who]==static_cast<Int>(p) ? -1.0f : static_cast<float>(frame.slot[who]);};
    for (std::size_t at=0;at<h;++at) {
        auto* hex=result+hexes+at*hw;
        Int token=state.get(Block::tokens)[at];
        if (two) {
            std::copy_n(fixed.hex9.data()+at*7,5,hex);
            int slot=number_slot11d(token);if (slot>=0) hex[5+slot]=1.0f;
            hex[15]=token>=2 && token<=12 ? static_cast<float>(odds[token-2]) : 0.0f;
        } else if (duel) {
            std::copy_n(fixed.hex9.data()+at*7,5,hex);hex[5]=fixed.number[at];
            hex[6]=token>=2 && token<=12 ? static_cast<float>(odds[token-2]) : 0.0f;
        } else {std::copy_n(fixed.hex9.data()+at*7,7,hex);if (rules) hex[7]=fixed.number[at];}
        if (state.get(Block::state_control)[0]==static_cast<Int>(at)) hex[two ? 16 : duel ? 7 : 6]=1.0f;
    }
    std::copy(fixed.ports.begin(),fixed.ports.end(),result+vertices);
    for (std::size_t at=0;at<v;++at) {
        result[vertices+at*7+6]=count_scale(state.get(Block::vertex_building)[at]);
        result[owner+at]=pointer(state.get(Block::vertex_owner)[at]);
    }
    for (std::size_t at=0;at<e;++at) result[edge+at]=pointer(state.get(Block::edge_owner)[at]);
}
inline std::vector<float> encode9(const ArenaState& state,int perspective,const std::string& contract="9") {
    std::vector<float> result(encoded9_width(state,contract));
    encode9_into(state,perspective,contract,result.data(),result.size());
    return result;
}
} // namespace hexset_native
