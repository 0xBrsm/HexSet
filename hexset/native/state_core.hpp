// SPDX-License-Identifier: GPL-3.0-only
// Owned integer state, preserving iteration order and private steal knowledge.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

namespace hexset_native {
struct EncodingCache;
using Int = std::int64_t;
using Counts = std::array<Int, 5>;

inline Int plus(Int a, Int b) {
    if ((b > 0 && a > std::numeric_limits<Int>::max() - b) ||
        (b < 0 && a < std::numeric_limits<Int>::min() - b))
        throw std::overflow_error("native integer overflow");
    return a + b;
}
inline Int total(const Counts& v) {
    Int result = 0;
    for (Int n : v) result = plus(result, n);
    return result;
}

struct LedgerRow {
    Counts known{}, want{}, waste{}, most{};
    Int unknown = 0;
    Int size() const { return plus(total(known), unknown); }
    void settle() {
        for (int r = 0; r < 5; ++r)
            most[r] = std::min(std::max(most[r], known[r]), plus(known[r], unknown));
        if (!unknown) return;
        Counts room{};
        for (int r = 0; r < 5; ++r) room[r] = most[r] - known[r];
        Int spare = total(room);
        if (spare < unknown) {
            for (int r = 0; r < 5; ++r) most[r] = plus(known[r], unknown);
            return;
        }
        // room remains the original row's gaps as in Python; spare and unknown
        // move after every forced resource, preserving the source tie/order.
        for (int r = 0; r < 5; ++r) {
            Int forced = unknown - (spare - room[r]);
            if (forced > 0) {
                known[r] = plus(known[r], forced);
                unknown -= forced;
                spare -= forced;
            }
        }
    }
    void spend(int resource, Int n, bool shown = true) {
        if (shown) waste[resource] = std::max<Int>(0, waste[resource] - n);
        most[resource] = std::max<Int>(0, most[resource] - n);
        Int taken = std::min(n, known[resource]);
        known[resource] -= taken;
        Int deficit = n - taken;
        if (deficit) {
            unknown = std::max<Int>(0, unknown - deficit);
            settle();
        }
    }
};

using Rows = std::vector<LedgerRow>;

struct Knowledge {
    Rows common;
    bool private_present = false;
    std::vector<std::optional<Rows>> private_rows;

    explicit Knowledge(std::size_t players = 0) : common(players), private_rows(players) {}
    void check_seat(int seat) const {
        if (seat < 0 || static_cast<std::size_t>(seat) >= common.size())
            throw std::invalid_argument("invalid seat");
    }
    static void check_resource(int resource) {
        if (resource < 0 || resource >= 5) throw std::invalid_argument("invalid resource");
    }
    template<class Function> void each(Function function) {
        function(common);
        if (private_present)
            for (auto& own : private_rows) if (own) function(*own);
    }
    const Rows& visible(int perspective) const {
        check_seat(perspective);
        return private_present && private_rows[perspective] ? *private_rows[perspective] : common;
    }
    void receive(int seat, int resource, Int n) {
        check_seat(seat); check_resource(resource);
        if (n <= 0) return;
        each([&](Rows& rows) {
            auto& row = rows[seat];
            row.known[resource] = plus(row.known[resource], n);
            row.most[resource] = plus(row.most[resource], n);
            row.want[resource] = std::max<Int>(0, row.want[resource] - n);
        });
    }
    void spend(int seat, int resource, Int n) {
        check_seat(seat); check_resource(resource);
        if (n <= 0) return;
        each([&](Rows& rows) { rows[seat].spend(resource, n); });
    }
    void gain_unknown(int seat, Int n) {
        check_seat(seat);
        if (n <= 0) return;
        each([&](Rows& rows) {
            rows[seat].unknown = plus(rows[seat].unknown, n);
            for (auto& m : rows[seat].most) m = plus(m, n);
        });
    }
    void show(int seat, const Counts& received, bool holds) {
        check_seat(seat);
        each([&](Rows& rows) {
            auto& row = rows[seat];
            for (int r = 0; r < 5; ++r) {
                Int n = received[r];
                if (n > 0) { row.want[r] = std::max(row.want[r], n); row.waste[r] = 0; }
                else if (n < 0) {
                    if (n == std::numeric_limits<Int>::min()) throw std::overflow_error("offer overflow");
                    row.waste[r] = std::max(row.waste[r], -n); row.want[r] = 0;
                    if (holds) {
                        Int rise = std::min(-n - row.known[r], row.unknown);
                        if (rise > 0) { row.known[r] = plus(row.known[r], rise); row.unknown -= rise; }
                    }
                }
            }
            row.settle();
        });
    }
    static void steal_unseen(Rows& rows, int thief, int victim) {
        auto& source = rows[victim];
        auto& target = rows[thief];
        source.settle();
        std::vector<int> could;
        for (int r = 0; r < 5; ++r) if (source.most[r] > 0) could.push_back(r);
        if (could.size() == 1) {
            int resource = could[0];
            source.spend(resource, 1, false);
            target.known[resource] = plus(target.known[resource], 1);
            target.most[resource] = plus(target.most[resource], 1);
            return;
        }
        Int old_total = source.size();
        for (auto& k : source.known) k = std::max<Int>(0, k - 1);
        source.unknown = std::max<Int>(0, old_total - 1 - total(source.known));
        source.settle();
        target.unknown = plus(target.unknown, 1);
        for (int r : could) target.most[r] = plus(target.most[r], 1);
        target.settle();
    }
    void steal(int thief, int victim, std::optional<int> resource) {
        check_seat(thief); check_seat(victim);
        bool seen = resource && *resource >= 0;
        if (seen) {
            check_resource(*resource);
            private_present = true;
            for (int seat : {thief, victim}) if (!private_rows[seat]) private_rows[seat] = common;
        }
        steal_unseen(common, thief, victim);
        if (private_present) {
            for (std::size_t seat = 0; seat < private_rows.size(); ++seat) {
                auto& own = private_rows[seat];
                if (!own) continue;
                if (seen && (static_cast<int>(seat) == thief || static_cast<int>(seat) == victim)) {
                    (*own)[victim].spend(*resource, 1, false);
                    auto& target = (*own)[thief];
                    target.known[*resource] = plus(target.known[*resource], 1);
                    target.most[*resource] = plus(target.most[*resource], 1);
                } else steal_unseen(*own, thief, victim);
            }
        }
    }
    void emptied(int seat, int resource) {
        check_seat(seat); check_resource(resource);
        each([&](Rows& rows) { rows[seat].known[resource] = 0; rows[seat].most[resource] = 0;
                              rows[seat].settle(); });
    }
};

// Indexed blocks avoid a Python/dictionary lookup in future native transitions.
// All integer values remain 64-bit: no game-rule cap is invented to narrow them.
#define COLLECTOR_BLOCKS(X) \
 X(players) X(hexes) X(terrain) X(tokens) X(ports) X(edges) \
 X(hex_vertices_offsets) X(hex_vertices_values) X(hex_edges_offsets) X(hex_edges_values) \
 X(hex_neighbors_offsets) X(hex_neighbors_values) X(vertex_hexes_offsets) X(vertex_hexes_values) \
 X(vertex_edges_offsets) X(vertex_edges_values) X(vertex_neighbors_offsets) X(vertex_neighbors_values) \
 X(edge_hexes_offsets) X(edge_hexes_values) X(roll_hexes_offsets) X(roll_hexes_values) \
 X(vertex_owner) X(vertex_building) X(building_enum) X(edge_owner) X(state_control) X(hands) X(bank) X(deck) X(deck_enum) \
 X(dev_cards) X(new_dev_cards) X(knights_played) X(dev_cards_played) \
 X(age_known) X(age_offsets) X(age_values) X(road_lengths) X(rules) \
 X(dice_control) X(dice_left) X(dice_recent) X(dice_sevens) \
 X(game_control) X(setup_queue) X(discard_quota) X(robber_allowed) X(locked) X(embargoes) X(before_roll)

enum class Block {
#define ENUM_ITEM(name) name,
    COLLECTOR_BLOCKS(ENUM_ITEM)
#undef ENUM_ITEM
    count
};
inline const std::array<const char*, static_cast<std::size_t>(Block::count)> block_names = {
#define NAME_ITEM(name) #name,
    COLLECTOR_BLOCKS(NAME_ITEM)
#undef NAME_ITEM
};

struct ArenaState {
    std::array<std::vector<Int>, static_cast<std::size_t>(Block::count)> blocks;
    Knowledge knowledge;
    // Native-only derived counts, updated by placements rather than repeatedly
    // rescanning the board in legal enumeration, encoding and win checks.
    std::vector<std::array<Int, 3>> piece_counts;
    std::uint64_t epoch = 0;
    std::uint64_t occupancy_epoch = 0;
    mutable std::uint64_t road_cache_epoch = std::numeric_limits<std::uint64_t>::max();
    mutable std::vector<Int> cached_roads_to_take, cached_reach;
    mutable std::shared_ptr<const EncodingCache> encoding_cache;
    std::vector<Int>& get(Block block) { return blocks[static_cast<std::size_t>(block)]; }
    const std::vector<Int>& get(Block block) const { return blocks[static_cast<std::size_t>(block)]; }
    void rebuild_counts() {
        piece_counts.assign(static_cast<std::size_t>(get(Block::players)[0]), {0,0,0});
        const auto& owners=get(Block::vertex_owner); const auto& kinds=get(Block::vertex_building);
        for (std::size_t v=0;v<owners.size();++v)
            if (owners[v]>=0 && kinds[v]>0) ++piece_counts[static_cast<std::size_t>(owners[v])][kinds[v]-1];
        for (Int owner:get(Block::edge_owner)) if (owner>=0) ++piece_counts[static_cast<std::size_t>(owner)][2];
    }
    void require_size(Block block, std::size_t count) const {
        if (get(block).size() != count) throw std::invalid_argument("invalid native field size");
    }
    void ragged(Block offsets, Block values, std::size_t count, std::size_t bound) const {
        const auto& at = get(offsets); const auto& data = get(values);
        require_size(offsets, count + 1);
        if (at.front() != 0 || at.back() != static_cast<Int>(data.size()))
            throw std::invalid_argument("invalid ragged endpoints");
        for (std::size_t i = 1; i < at.size(); ++i)
            if (at[i] < at[i-1]) throw std::invalid_argument("non-monotone ragged field");
        for (Int value : data) if (value < 0 || static_cast<std::uint64_t>(value) >= bound)
            throw std::invalid_argument("invalid ragged index");
    }
    void validate() const {
        require_size(Block::players, 1);
        Int players = get(Block::players)[0];
        if (players != 2 && players != 4) throw std::invalid_argument("native target supports two/four seats");
        std::size_t p = static_cast<std::size_t>(players), v = get(Block::vertex_owner).size();
        std::size_t e = get(Block::edge_owner).size(), h = get(Block::terrain).size();
        if (!v || !e || !h) throw std::invalid_argument("empty native topology");
        require_size(Block::vertex_building, v); require_size(Block::building_enum, v);
        require_size(Block::deck_enum,get(Block::deck).size()); require_size(Block::hexes, h*3);
        require_size(Block::edges, e*2); require_size(Block::tokens, h);
        require_size(Block::state_control, 3); require_size(Block::hands, p*5);
        require_size(Block::bank, 5); require_size(Block::dev_cards, p*5);
        require_size(Block::new_dev_cards, p*5); require_size(Block::knights_played, p);
        require_size(Block::dev_cards_played, 5); require_size(Block::age_known, p);
        require_size(Block::age_offsets, p+1); require_size(Block::road_lengths, p);
        require_size(Block::rules, 4); require_size(Block::dice_control, 4);
        require_size(Block::game_control, 19);
        if (get(Block::ports).size()%5 || get(Block::dice_sevens).size()%2 || get(Block::embargoes).size()%2)
            throw std::invalid_argument("invalid native record stride");
        const auto& ports=get(Block::ports);
        for (std::size_t i=0;i<ports.size();i+=5)
            if (ports[i]<0 || ports[i]>=static_cast<Int>(e) || ports[i+1]<0 || ports[i+1]>=static_cast<Int>(v) ||
                ports[i+2]<0 || ports[i+2]>=static_cast<Int>(v) || ports[i+3]<-1 || ports[i+3]>=5 || ports[i+4]<=0)
                throw std::invalid_argument("invalid native port");
        ragged(Block::hex_vertices_offsets, Block::hex_vertices_values, h, v);
        ragged(Block::hex_edges_offsets, Block::hex_edges_values, h, e);
        ragged(Block::hex_neighbors_offsets, Block::hex_neighbors_values, h, h);
        ragged(Block::vertex_hexes_offsets, Block::vertex_hexes_values, v, h);
        ragged(Block::vertex_edges_offsets, Block::vertex_edges_values, v, e);
        ragged(Block::vertex_neighbors_offsets, Block::vertex_neighbors_values, v, v);
        ragged(Block::edge_hexes_offsets, Block::edge_hexes_values, e, h);
        ragged(Block::roll_hexes_offsets, Block::roll_hexes_values, 13, h);
        const auto& ages = get(Block::age_offsets);
        if (ages.front() != 0 || ages.back() != static_cast<Int>(get(Block::age_values).size()))
            throw std::invalid_argument("invalid age offsets");
        for (std::size_t i=1; i<ages.size(); ++i)
            if (ages[i]<ages[i-1]) throw std::invalid_argument("invalid age offsets");
        if (knowledge.common.size()!=p || knowledge.private_rows.size()!=p)
            throw std::invalid_argument("invalid ledger player count");
        auto range = [&](Block block, Int least, Int most) {
            for (Int value:get(block)) if (value<least || value>most)
                throw std::invalid_argument("native field value outside its index/count domain");
        };
        range(Block::vertex_owner,-1,players-1); range(Block::edge_owner,-1,players-1);
        range(Block::vertex_building,0,2); range(Block::building_enum,0,1); range(Block::deck_enum,0,1);
        range(Block::edges,0,static_cast<Int>(v)-1); range(Block::deck,0,4);
        for (Block block:{Block::hands,Block::bank,Block::dev_cards,Block::new_dev_cards,
                          Block::knights_played,Block::dev_cards_played,Block::road_lengths,Block::age_values})
            range(block,0,std::numeric_limits<Int>::max());
        range(Block::age_known,0,1); range(Block::locked,0,players-1); range(Block::embargoes,0,players-1);
        range(Block::setup_queue,0,players-1);
        const auto& sc=get(Block::state_control);
        if (sc[0]<-1 || sc[0]>=static_cast<Int>(h) || sc[1]<-1 || sc[1]>=players || sc[2]<-1 || sc[2]>=players)
            throw std::invalid_argument("invalid robber/award owner");
        const auto& gc=get(Block::game_control);
        if (gc[0]<0 || gc[0]>6 || gc[1]<0 || gc[1]>=players || gc[7]<0 || gc[7]>6 ||
            gc[2]<0 || gc[2]>static_cast<Int>(get(Block::setup_queue).size()) || gc[3]<-1 || gc[3]>=static_cast<Int>(v) ||
            gc[13]<0 || gc[13]>2 || gc[14]<0 || gc[14]>2 || gc[16]<0 || gc[16]>=players ||
            (gc[5]!=0 && gc[5]!=1) || (gc[15]!=0 && gc[15]!=1) || (gc[18]!=0 && gc[18]!=1))
            throw std::invalid_argument("invalid phase/actor/control settings");
        for (std::size_t seat=0;seat<p;++seat)
            if (!get(Block::age_known)[seat] && ages[seat]!=ages[seat+1])
                throw std::invalid_argument("unknown card ages cannot contain values");
        const auto& dice=get(Block::dice_control);
        if (dice[0]!=0 && dice[0]!=1) throw std::invalid_argument("invalid dice presence");
        require_size(Block::dice_left,dice[0] ? 11 : 0);
        if (!dice[0] && (!get(Block::dice_recent).empty() || !get(Block::dice_sevens).empty()))
            throw std::invalid_argument("absent dice state has history");
    }
};
} // namespace hexset_native
