// SPDX-License-Identifier: GPL-3.0-only
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <memory>
#include <string>
#include <utility>
#include "state_core.hpp"
#include "state_primitives.hpp"
#include "game_simple.hpp"
#include "roads_core.hpp"
#include "game_placement.hpp"
#include "game_cards.hpp"
#include "game_chance.hpp"
#include "atomic_dispatch.hpp"
#include "legal_core.hpp"
#include "encoding.hpp"
#include "encoding8.hpp"
#include "request_core.hpp"
using namespace hexset_native;
namespace {
struct PythonError {};
struct Object {
    PyObject* value;
    explicit Object(PyObject* v=nullptr) : value(v) {}
    ~Object() { Py_XDECREF(value); }
    Object(const Object&)=delete;
    Object& operator=(const Object&)=delete;
    PyObject* release() { auto* v=value; value=nullptr; return v; }
};
void ensure(bool ok, const char* message) { if (!ok) throw std::invalid_argument(message); }
PyObject* required(PyObject* object, const char* key) {
    ensure(PyDict_Check(object), "native snapshot must be a dictionary");
    auto* value = PyDict_GetItemString(object, key);
    ensure(value != nullptr, "missing native snapshot field");
    return value;
}
Int integer(PyObject* object) {
    ensure(PyIndex_Check(object), "native snapshot requires integers");
    Object indexed(PyNumber_Index(object)); if (!indexed.value) throw PythonError{};
    Int value = PyLong_AsLongLong(indexed.value); if (PyErr_Occurred()) throw PythonError{};
    return value;
}
std::vector<Int> integers(PyObject* object) {
    Object sequence(PySequence_Fast(object, "native field must be an integer sequence"));
    if (!sequence.value) throw PythonError{};
    std::vector<Int> values;
    auto count = PySequence_Fast_GET_SIZE(sequence.value);
    values.reserve(static_cast<std::size_t>(count));
    for (Py_ssize_t i=0; i<count; ++i) values.push_back(integer(PySequence_Fast_GET_ITEM(sequence.value, i)));
    return values;
}
Rows read_rows(PyObject* object, std::size_t players) {
    Object rows(PySequence_Fast(object, "ledger rows must be sequences"));
    if (!rows.value) throw PythonError{};
    ensure(PySequence_Fast_GET_SIZE(rows.value)==static_cast<Py_ssize_t>(players), "invalid ledger row count");
    Rows result(players);
    for (std::size_t s=0; s<players; ++s) {
        auto values=integers(PySequence_Fast_GET_ITEM(rows.value, static_cast<Py_ssize_t>(s)));
        ensure(values.size()==21, "ledger row requires 21 integers");
        for (Int value:values) ensure(value>=0, "ledger counts must be nonnegative");
        auto& row=result[s];
        std::copy_n(values.begin(),5,row.known.begin()); row.unknown=values[5];
        std::copy_n(values.begin()+6,5,row.want.begin());
        std::copy_n(values.begin()+11,5,row.waste.begin());
        std::copy_n(values.begin()+16,5,row.most.begin());
    }
    return result;
}
ArenaState read_snapshot(PyObject* snapshot) {
    ensure(PyDict_Size(snapshot)==4, "native snapshot has unknown/missing top-level fields");
    ensure(integer(required(snapshot,"version"))==1,"unsupported native snapshot version");
    Int epoch=integer(required(snapshot,"epoch")); ensure(epoch>=0,"invalid native epoch");
    ArenaState state; state.epoch=static_cast<std::uint64_t>(epoch);
    auto* arrays=required(snapshot,"arrays");
    ensure(PyDict_Check(arrays) && PyDict_Size(arrays)==static_cast<Py_ssize_t>(block_names.size()),
           "native snapshot has unknown/missing arrays");
    for (std::size_t i=0;i<block_names.size();++i) state.blocks[i]=integers(required(arrays,block_names[i]));
    state.require_size(Block::players,1);
    Int players=state.get(Block::players)[0]; ensure(players==2 || players==4,"native target supports two/four seats");
    auto count=static_cast<std::size_t>(players);
    auto* ledger=required(snapshot,"ledger");
    ensure(PyDict_Check(ledger) && PyDict_Size(ledger)==2,"invalid native ledger snapshot");
    state.knowledge=Knowledge(count);
    state.knowledge.common=read_rows(required(ledger,"common"),count);
    auto* private_rows=required(ledger,"private");
    if (private_rows!=Py_None) {
        state.knowledge.private_present=true;
        Object entries(PySequence_Fast(private_rows,"private ledger must be a sequence"));
        if (!entries.value) throw PythonError{};
        ensure(PySequence_Fast_GET_SIZE(entries.value)==players,"invalid private ledger count");
        for (std::size_t s=0;s<count;++s) {
            auto* own=PySequence_Fast_GET_ITEM(entries.value,static_cast<Py_ssize_t>(s));
            if (own!=Py_None) state.knowledge.private_rows[s]=read_rows(own,count);
        }
    }
    state.validate();
    state.rebuild_counts();
    return state;
}
PyObject* tuple(const std::vector<Int>& values) {
    Object result(PyTuple_New(static_cast<Py_ssize_t>(values.size())));
    if (!result.value) throw PythonError{};
    for (std::size_t i=0;i<values.size();++i) {
        auto* item=PyLong_FromLongLong(values[i]); if (!item) throw PythonError{};
        PyTuple_SET_ITEM(result.value,static_cast<Py_ssize_t>(i),item);
    }
    return result.release();
}
PyObject* rows_object(const Rows& rows) {
    Object result(PyTuple_New(static_cast<Py_ssize_t>(rows.size())));
    if (!result.value) throw PythonError{};
    for (std::size_t s=0;s<rows.size();++s) {
        const auto& row=rows[s]; std::vector<Int> values;
        values.insert(values.end(),row.known.begin(),row.known.end()); values.push_back(row.unknown);
        values.insert(values.end(),row.want.begin(),row.want.end());
        values.insert(values.end(),row.waste.begin(),row.waste.end());
        values.insert(values.end(),row.most.begin(),row.most.end());
        PyTuple_SET_ITEM(result.value,static_cast<Py_ssize_t>(s),tuple(values));
    }
    return result.release();
}
void set(PyObject* dictionary, const char* key, PyObject* value) {
    Object owned(value); if (!value || PyDict_SetItemString(dictionary,key,value)<0) throw PythonError{};
}
PyObject* snapshot_object(const ArenaState& state) {
    Object result(PyDict_New()), arrays(PyDict_New()), ledger(PyDict_New());
    if (!result.value || !arrays.value || !ledger.value) throw PythonError{};
    set(result.value,"version",PyLong_FromLong(1));
    set(result.value,"epoch",PyLong_FromUnsignedLongLong(state.epoch));
    for (std::size_t i=0;i<block_names.size();++i) set(arrays.value,block_names[i],tuple(state.blocks[i]));
    set(result.value,"arrays",arrays.release());
    set(ledger.value,"common",rows_object(state.knowledge.common));
    if (!state.knowledge.private_present) { Py_INCREF(Py_None); set(ledger.value,"private",Py_None); }
    else {
        Object own(PyTuple_New(static_cast<Py_ssize_t>(state.knowledge.private_rows.size())));
        if (!own.value) throw PythonError{};
        for (std::size_t s=0;s<state.knowledge.private_rows.size();++s) {
            auto& entry=state.knowledge.private_rows[s];
            PyObject* row;
            if (entry) row=rows_object(*entry); else { row=Py_None; Py_INCREF(row); }
            PyTuple_SET_ITEM(own.value,static_cast<Py_ssize_t>(s),row);
        }
        set(ledger.value,"private",own.release());
    }
    set(result.value,"ledger",ledger.release());
    return result.release();
}
void error() {
    try { throw; }
    catch (const PythonError&) { if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError,"native Python API failure"); }
    catch (const std::overflow_error& e) { PyErr_SetString(PyExc_OverflowError,e.what()); }
    catch (const std::invalid_argument& e) { PyErr_SetString(PyExc_ValueError,e.what()); }
    catch (const std::bad_alloc&) { PyErr_NoMemory(); }
    catch (const std::exception& e) { PyErr_SetString(PyExc_RuntimeError,e.what()); }
}

struct Core { PyObject_HEAD ArenaState* state; };
ArenaState& checked(Core* self) {
    if (!self->state) throw std::invalid_argument("uninitialized native state");
    return *self->state;
}
PyObject* core_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self=reinterpret_cast<Core*>(type->tp_alloc(type,0));
    if (self) self->state=nullptr;
    return reinterpret_cast<PyObject*>(self);
}
int core_init(Core* self, PyObject* args, PyObject* kwargs) {
    static const char* names[]={"snapshot",nullptr}; PyObject* snapshot;
    if (!PyArg_ParseTupleAndKeywords(args,kwargs,"O",const_cast<char**>(names),&snapshot)) return -1;
    try {
        auto state=std::make_unique<ArenaState>(read_snapshot(snapshot));
        delete self->state; self->state=state.release(); return 0;
    } catch (...) { error(); return -1; }
}
void core_free(Core* self) { delete self->state; Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self)); }
PyObject* core_snapshot(Core* self, PyObject*) {
    try { return snapshot_object(checked(self)); } catch (...) { error(); return nullptr; }
}
PyObject* core_clone(Core* self, PyObject*) {
    try {
        auto state=std::make_unique<ArenaState>(checked(self));
        auto* cloned=reinterpret_cast<Core*>(Py_TYPE(self)->tp_alloc(Py_TYPE(self),0));
        if (!cloned) return nullptr;
        cloned->state=state.release(); return reinterpret_cast<PyObject*>(cloned);
    } catch (...) { error(); return nullptr; }
}
template<class Function> PyObject* transaction(Core* self, Function function) {
    try {
        auto& state=checked(self);
        // All operations are atomic even for malicious integer-overflow inputs.
        // Only the small ledger is copied; board/card storage remains persistent.
        Knowledge working=state.knowledge;
        function(working);
        ensure(state.epoch<static_cast<std::uint64_t>(std::numeric_limits<Int>::max()),"epoch exhausted");
        state.knowledge=std::move(working); ++state.epoch;
        Py_RETURN_NONE;
    } catch (...) { error(); return nullptr; }
}
PyObject* core_receive(Core* self,PyObject* args) {
    int seat,resource; long long n=1;
    if (!PyArg_ParseTuple(args,"ii|L",&seat,&resource,&n)) return nullptr;
    return transaction(self,[&](Knowledge& k){k.receive(seat,resource,n);});
}
PyObject* core_spend(Core* self,PyObject* args) {
    int seat,resource; long long n=1;
    if (!PyArg_ParseTuple(args,"ii|L",&seat,&resource,&n)) return nullptr;
    return transaction(self,[&](Knowledge& k){k.spend(seat,resource,n);});
}
PyObject* core_gain(Core* self,PyObject* args) {
    int seat; long long n=1;
    if (!PyArg_ParseTuple(args,"i|L",&seat,&n)) return nullptr;
    return transaction(self,[&](Knowledge& k){k.gain_unknown(seat,n);});
}
PyObject* core_emptied(Core* self,PyObject* args) {
    int seat,resource;
    if (!PyArg_ParseTuple(args,"ii",&seat,&resource)) return nullptr;
    return transaction(self,[&](Knowledge& k){k.emptied(seat,resource);});
}
PyObject* core_steal(Core* self,PyObject* args) {
    int thief,victim; PyObject* resource=Py_None;
    if (!PyArg_ParseTuple(args,"ii|O",&thief,&victim,&resource)) return nullptr;
    try {
        std::optional<int> r;
        if (resource!=Py_None) { Int value=integer(resource); ensure(value>=-1 && value<5,"invalid stolen resource"); r=static_cast<int>(value); }
        return transaction(self,[&](Knowledge& k){k.steal(thief,victim,r);});
    } catch (...) { error(); return nullptr; }
}
PyObject* core_show(Core* self,PyObject* args) {
    int seat,holds=1; PyObject* received;
    if (!PyArg_ParseTuple(args,"iO|p",&seat,&received,&holds)) return nullptr;
    try {
        auto values=integers(received); ensure(values.size()==5,"offer requires five resources");
        Counts bundle{}; std::copy(values.begin(),values.end(),bundle.begin());
        return transaction(self,[&](Knowledge& k){k.show(seat,bundle,holds!=0);});
    } catch (...) { error(); return nullptr; }
}
PyObject* core_visible_ledger(Core* self,PyObject* args) {
    int seat; if (!PyArg_ParseTuple(args,"i",&seat)) return nullptr;
    try {return rows_object(checked(self).knowledge.visible(seat));} catch (...) {error();return nullptr;}
}
PyObject* core_visible_cards(Core* self,PyObject* args) {
    int seat; if (!PyArg_ParseTuple(args,"i",&seat)) return nullptr;
    try {
        const auto& state=checked(self); state.knowledge.check_seat(seat);
        auto p=state.knowledge.common.size();
        Object result(PyDict_New()); if (!result.value) throw PythonError{};
        for (auto name_block : {std::pair<const char*,Block>{"own_hand",Block::hands},
                               {"own_dev",Block::dev_cards},{"own_new_dev",Block::new_dev_cards}}) {
            const auto& values=state.get(name_block.second);
            set(result.value,name_block.first,tuple(std::vector<Int>(values.begin()+seat*5,values.begin()+(seat+1)*5)));
        }
        std::vector<Int> hand_sizes(p),dev_sizes(p),new_sizes(p);
        for (std::size_t s=0;s<p;++s) for (int r=0;r<5;++r) {
            hand_sizes[s]=plus(hand_sizes[s],state.get(Block::hands)[s*5+r]);
            dev_sizes[s]=plus(dev_sizes[s],state.get(Block::dev_cards)[s*5+r]);
            new_sizes[s]=plus(new_sizes[s],state.get(Block::new_dev_cards)[s*5+r]);
        }
        set(result.value,"hand_sizes",tuple(hand_sizes)); set(result.value,"dev_sizes",tuple(dev_sizes));
        set(result.value,"new_dev_sizes",tuple(new_sizes));
        set(result.value,"deck_size",PyLong_FromSize_t(state.get(Block::deck).size()));
        return result.release();
    } catch (...) {error();return nullptr;}
}
PyObject* core_true_points(Core* self,PyObject*) {
    try {
        const auto& state=checked(self); std::vector<Int> points(state.knowledge.common.size());
        for (std::size_t s=0;s<points.size();++s) points[s]=true_points(state,static_cast<int>(s));
        return tuple(points);
    } catch (...) {error();return nullptr;}
}
template<class Function> PyObject* state_transaction(Core* self, Function function) {
    try {
        auto& state=checked(self);
        ensure(state.epoch<static_cast<std::uint64_t>(std::numeric_limits<Int>::max()),"epoch exhausted");
        // Primitive functions validate/prepare ALL changes before their first
        // write; no copy of immutable board arrays per engine action.
        function(state); ++state.epoch; Py_RETURN_NONE;
    } catch (...) {error();return nullptr;}
}
PyObject* core_placement(Core* self,PyObject* args) {
    int seat; if (!PyArg_ParseTuple(args,"i",&seat)) return nullptr;
    try {
        const auto& state=checked(self); state.knowledge.check_seat(seat);
        std::vector<Int> setup,settlements,cities;
        for (std::size_t v=0;v<state.get(Block::vertex_owner).size();++v) {
            if (can_place_settlement(state,seat,static_cast<Int>(v),false)) setup.push_back(static_cast<Int>(v));
            if (can_place_settlement(state,seat,static_cast<Int>(v),true)) settlements.push_back(static_cast<Int>(v));
            if (can_upgrade_to_city(state,seat,static_cast<Int>(v))) cities.push_back(static_cast<Int>(v));
        }
        Object result(PyDict_New()); if (!result.value) throw PythonError{};
        set(result.value,"setup",tuple(setup)); set(result.value,"settlements",tuple(settlements));
        set(result.value,"cities",tuple(cities)); set(result.value,"roads",tuple(placeable_roads(state,seat)));
        const auto& counts=state.piece_counts[seat];
        set(result.value,"counts",tuple(std::vector<Int>(counts.begin(),counts.end())));
        return result.release();
    } catch (...) {error();return nullptr;}
}
PyObject* core_place_settlement(Core* self,PyObject* args) {
    int seat,connected=1; long long vertex;
    if (!PyArg_ParseTuple(args,"iL|p",&seat,&vertex,&connected)) return nullptr;
    return state_transaction(self,[&](ArenaState& state){place_settlement(state,seat,vertex,connected!=0);});
}
PyObject* core_place_road(Core* self,PyObject* args) {
    int seat; long long edge; if (!PyArg_ParseTuple(args,"iL",&seat,&edge)) return nullptr;
    return state_transaction(self,[&](ArenaState& state){place_road(state,seat,edge);});
}
PyObject* core_place_city(Core* self,PyObject* args) {
    int seat; long long vertex; if (!PyArg_ParseTuple(args,"iL",&seat,&vertex)) return nullptr;
    return state_transaction(self,[&](ArenaState& state){upgrade_to_city(state,seat,vertex);});
}
PyObject* core_afford(Core* self,PyObject* args) {
    int seat,purchase; if (!PyArg_ParseTuple(args,"ii",&seat,&purchase)) return nullptr;
    try {return PyBool_FromLong(can_afford(checked(self),seat,purchase));} catch (...) {error();return nullptr;}
}
PyObject* core_ratios(Core* self,PyObject* args) {
    int seat; if (!PyArg_ParseTuple(args,"i",&seat)) return nullptr;
    try {auto values=trade_ratios(checked(self),seat); return tuple(std::vector<Int>(values.begin(),values.end()));}
    catch (...) {error();return nullptr;}
}
PyObject* core_pay(Core* self,PyObject* args) {
    int seat,purchase; if (!PyArg_ParseTuple(args,"ii",&seat,&purchase)) return nullptr;
    return state_transaction(self,[&](ArenaState& state){pay(state,seat,purchase);});
}
PyObject* core_bank_trade(Core* self,PyObject* args) {
    int seat,give,receive; if (!PyArg_ParseTuple(args,"iii",&seat,&give,&receive)) return nullptr;
    return state_transaction(self,[&](ArenaState& state){bank_trade(state,seat,give,receive);});
}
PyObject* core_production(Core* self,PyObject* args) {
    long long roll; int gold=0; if (!PyArg_ParseTuple(args,"L|p",&roll,&gold)) return nullptr;
    try {return tuple(production(checked(self),roll,gold!=0));} catch (...) {error();return nullptr;}
}
PyObject* core_distribute(Core* self,PyObject* args) {
    long long roll; if (!PyArg_ParseTuple(args,"L",&roll)) return nullptr;
    try {
        auto& state=checked(self);
        ensure(state.epoch<static_cast<std::uint64_t>(std::numeric_limits<Int>::max()),"epoch exhausted");
        auto prepared=prepare_distribution(state,roll);
        Object granted(tuple(prepared.granted));
        prepared.commit(state); ++state.epoch; return granted.release();
    } catch (...) {error();return nullptr;}
}
PyObject* core_public_diff(Core* self,PyObject* args) {
    PyObject* before; if (!PyArg_ParseTuple(args,"O",&before)) return nullptr;
    try {
        auto values=integers(before);
        return state_transaction(self,[&](ArenaState& state){public_hand_diff(state,values);});
    } catch (...) {error();return nullptr;}
}
PyObject* core_apply_simple(Core* self,PyObject* args) {
    int type;long long a=0,b=0;
    if (!PyArg_ParseTuple(args,"i|LL",&type,&a,&b)) return nullptr;
    if (type!=1 && type!=8 && type!=13) {
        PyErr_SetString(PyExc_NotImplementedError,"native game controller currently implements END_TURN, BUILD_CITY and BANK_TRADE only");
        return nullptr;
    }
    return state_transaction(self,[&](ArenaState& state){apply_simple(state,type,a,b);});
}
PyObject* core_road_queries(Core* self,PyObject* args) {
    int seat,budget=3;
    if (!PyArg_ParseTuple(args,"i|i",&seat,&budget)) return nullptr;
    try {
        const auto& state=checked(self);state.knowledge.check_seat(seat);
        ensure(budget>=0 && budget<=15,"road budget outside piece supply");
        RoadGraph graph{state};RoadSearch search(graph,seat,budget,true);search.run();
        Object result(PyDict_New()),paths(PyTuple_New(budget+1));
        if (!result.value || !paths.value) throw PythonError{};
        set(result.value,"lengths",tuple(road_lengths(state)));
        set(result.value,"with_new",tuple(search.best));
        for (int k=0;k<=budget;++k) {
            std::vector<Int> path;
            for (std::size_t i=search.owned;i<search.usable.size();++i) if (search.paths[k][i]) path.push_back(search.usable[i]);
            PyTuple_SET_ITEM(paths.value,k,tuple(path));
        }
        set(result.value,"paths",paths.release());set(result.value,"to_take",tuple(roads_to_take_all(state)));
        set(result.value,"reach",tuple(reach_all(state)));return result.release();
    } catch (...) {error();return nullptr;}
}
PyObject* core_refresh_awards(Core* self,PyObject* args) {
    int owner=-1,settlement_owner=-1;long long vertex=-1;
    if (!PyArg_ParseTuple(args,"|iLi",&owner,&vertex,&settlement_owner)) return nullptr;
    return state_transaction(self,[&](ArenaState& state){
        if (owner!=-1) state.knowledge.check_seat(owner);
        if (vertex!=-1) vertex_index(state,vertex);
        if (settlement_owner!=-1) state.knowledge.check_seat(settlement_owner);
        update_longest_road(state,owner,vertex,settlement_owner);update_largest_army(state);
    });
}
PyObject* core_apply_placement(Core* self,PyObject* args) {
    int type;long long index;
    if (!PyArg_ParseTuple(args,"iL",&type,&index)) return nullptr;
    if (type<4 || type>7) {PyErr_SetString(PyExc_NotImplementedError,"native placement controller implements action types4..7 only");return nullptr;}
    return state_transaction(self,[&](ArenaState& state){apply_placement(state,type,index);});
}
PyObject* core_apply_cards(Core* self,PyObject* args) {
    int type;long long argument=0;
    if (!PyArg_ParseTuple(args,"i|L",&type,&argument)) return nullptr;
    if (type!=2 && type!=3 && type!=10 && type!=11 && type!=12) {
        PyErr_SetString(PyExc_NotImplementedError,"native card controller implements action types2/3/10/11/12 only");return nullptr;
    }
    return state_transaction(self,[&](ArenaState& state){apply_cards(state,type,argument);});
}
// The callback is borrowed for this call only: Core never owns a Python
// adapter, so native state and the Python Chance wrapper cannot form a cycle.
PyObject* core_control(Core* self,PyObject*) {
    try {return tuple(checked(self).get(Block::game_control));} catch (...) {error();return nullptr;}
}
PyObject* core_phase(Core* self,PyObject*) {
    try {return PyLong_FromLongLong(checked(self).get(Block::game_control)[PhaseColumn]);}
    catch (...) {error();return nullptr;}
}
void apply_with_python_chance(ArenaState& state,int type,Int a,Int b,Int seat,PyObject* callback) {
    // Pure C++ lanes can call apply_atomic directly with their existing Chance
    // bridge. Only the CPython entry point constructs these borrowed closures.
    apply_atomic(state,type,a,b,seat,
        [&](int actor){
            Object result(PyObject_CallFunction(callback,"si","roll",actor));
            if (!result.value) throw PythonError{};
            return integer(result.value);
        },
        [&](const Counts& held)->std::optional<Int>{
            Object hand(tuple(std::vector<Int>(held.begin(),held.end())));
            Object result(PyObject_CallFunction(callback,"sO","steal",hand.value));
            if (!result.value) throw PythonError{};
            if (result.value==Py_None) return std::nullopt;
            return integer(result.value);
        });
}
PyObject* core_apply_atomic(Core* self,PyObject* args) {
    int type;long long a,b,seat;PyObject* callback;
    if (!PyArg_ParseTuple(args,"iLLLO",&type,&a,&b,&seat,&callback)) return nullptr;
    if ((type==0 || type==9) && !PyCallable_Check(callback)) {PyErr_SetString(PyExc_TypeError,"chance callback must be callable");return nullptr;}
    return state_transaction(self,[&](ArenaState& state){apply_with_python_chance(state,type,a,b,seat,callback);});
}
PyObject* core_apply_chance(Core* self,PyObject* args) {
    int type;long long a,b,seat;PyObject* callback;
    if (!PyArg_ParseTuple(args,"iLLLO",&type,&a,&b,&seat,&callback)) return nullptr;
    if (type!=0 && type!=9 && type!=14) {PyErr_SetString(PyExc_NotImplementedError,"native chance controller implements0/9/14 only");return nullptr;}
    if (type!=14 && !PyCallable_Check(callback)) {PyErr_SetString(PyExc_TypeError,"chance callback must be callable");return nullptr;}
    return state_transaction(self,[&](ArenaState& state){apply_with_python_chance(state,type,a,b,seat,callback);});
}
#include "encoding_binding.hpp"
#include "request_binding.hpp"
#include "legal_binding.hpp"
PyMethodDef methods[]={
 {"referee_snapshot",reinterpret_cast<PyCFunction>(core_snapshot),METH_NOARGS,"Owned state snapshot; privileged state, never policy input."},
 {"clone",reinterpret_cast<PyCFunction>(core_clone),METH_NOARGS,"Independent native state and every private ledger."},
 {"ledger_receive",reinterpret_cast<PyCFunction>(core_receive),METH_VARARGS,nullptr},
 {"ledger_spend",reinterpret_cast<PyCFunction>(core_spend),METH_VARARGS,nullptr},
 {"ledger_gain_unknown",reinterpret_cast<PyCFunction>(core_gain),METH_VARARGS,nullptr},
 {"ledger_emptied",reinterpret_cast<PyCFunction>(core_emptied),METH_VARARGS,nullptr},
 {"ledger_steal",reinterpret_cast<PyCFunction>(core_steal),METH_VARARGS,nullptr},
 {"ledger_show",reinterpret_cast<PyCFunction>(core_show),METH_VARARGS,nullptr},
 {"visible_ledger",reinterpret_cast<PyCFunction>(core_visible_ledger),METH_VARARGS,"Only perspective's ledger, common fallback."},
 {"visible_cards",reinterpret_cast<PyCFunction>(core_visible_cards),METH_VARARGS,"Own card identities and public opponent totals; ledger bounds are separate."},
 {"privileged_true_points",reinterpret_cast<PyCFunction>(core_true_points),METH_NOARGS,"Referee victory points only; never a visible observation."},
 {"placement_queries",reinterpret_cast<PyCFunction>(core_placement),METH_VARARGS,"Ascending primitive placement candidates and cached piece counts."},
 {"place_settlement",reinterpret_cast<PyCFunction>(core_place_settlement),METH_VARARGS,"State primitive only: no payment, phase, award or record."},
 {"place_road",reinterpret_cast<PyCFunction>(core_place_road),METH_VARARGS,"State primitive only: no payment, phase, award or record."},
 {"upgrade_to_city",reinterpret_cast<PyCFunction>(core_place_city),METH_VARARGS,"State primitive only: no payment, phase, award or record."},
 {"can_afford",reinterpret_cast<PyCFunction>(core_afford),METH_VARARGS,nullptr},
 {"trade_ratios",reinterpret_cast<PyCFunction>(core_ratios),METH_VARARGS,nullptr},
 {"pay",reinterpret_cast<PyCFunction>(core_pay),METH_VARARGS,"Economy primitive; controller must separately apply public ledger diff."},
 {"bank_trade",reinterpret_cast<PyCFunction>(core_bank_trade),METH_VARARGS,"Economy primitive; controller must separately apply public ledger diff."},
 {"production",reinterpret_cast<PyCFunction>(core_production),METH_VARARGS,"Flat gross production, or per-seat gold claims when gold=True."},
 {"distribute",reinterpret_cast<PyCFunction>(core_distribute),METH_VARARGS,"Economy primitive with exact bank-shortage rule; returns flat grants."},
 {"public_hand_diff",reinterpret_cast<PyCFunction>(core_public_diff),METH_VARARGS,"Public changes only: never use to record a hidden steal."},
 {"apply_simple",reinterpret_cast<PyCFunction>(core_apply_simple),METH_VARARGS,"Chance-free atomic action entry point."},
 {"road_queries",reinterpret_cast<PyCFunction>(core_road_queries),METH_VARARGS,"Native graph queries and cached reach/award distances."},
 {"refresh_awards",reinterpret_cast<PyCFunction>(core_refresh_awards),METH_VARARGS,"Refresh native road lengths and both integer award holders."},
 {"apply_placement",reinterpret_cast<PyCFunction>(core_apply_placement),METH_VARARGS,"Setup and build placement entry point."},
 {"apply_cards",reinterpret_cast<PyCFunction>(core_apply_cards),METH_VARARGS,"Complete native concrete-deck buy/knight/roadbuilding/monopoly/plenty actions."},
 {"control",reinterpret_cast<PyCFunction>(core_control),METH_NOARGS,"Small control tuple only; contains no private hands, cards or referee labels."},
 {"phase",reinterpret_cast<PyCFunction>(core_phase),METH_NOARGS,"Current public phase, without constructing the full control tuple."},
 {"apply_atomic",reinterpret_cast<PyCFunction>(core_apply_atomic),METH_VARARGS,"One native dispatcher for all15 atomic no-trade actions; no Game reconstruction."},
 {"apply_chance",reinterpret_cast<PyCFunction>(core_apply_chance),METH_VARARGS,"Complete native roll, robber and discard with one original Chance callback per draw."},
 {"legal_actions",reinterpret_cast<PyCFunction>(core_legal_actions),METH_VARARGS,"Ordered native atomic legality; debug copy of reusable worker queries."},
 {"decision",reinterpret_cast<PyCFunction>(core_decision),METH_VARARGS,"Exact native flat/setup/road/discard decision arrays; debug copies."},
 {"encode9",reinterpret_cast<PyCFunction>(core_encode9),METH_VARARGS,"Persistent native visible9/9d rows; no Python feature or View callbacks."},
 {"encode8",reinterpret_cast<PyCFunction>(core_encode8),METH_VARARGS,"Native contract-8/8.1 rows: hexes, vertices, edges and globals packed, as encoding.encode."},
 {"record8",reinterpret_cast<PyCFunction>(core_record8),METH_VARARGS,"One position's information-set record (onnx_record), every field but the mask, as raw bytes."},
 {nullptr,nullptr,0,nullptr}
};
PyType_Slot slots[]={
 {Py_tp_new,reinterpret_cast<void*>(core_new)}, {Py_tp_init,reinterpret_cast<void*>(core_init)},
 {Py_tp_dealloc,reinterpret_cast<void*>(core_free)}, {Py_tp_methods,reinterpret_cast<void*>(methods)},
 {0,nullptr}
};
PyType_Spec spec={"hexset._native_core.ArenaState",sizeof(Core),0,Py_TPFLAGS_DEFAULT,slots};
PyModuleDef module={PyModuleDef_HEAD_INIT,"_native_core",nullptr,-1,nullptr,nullptr,nullptr,nullptr,nullptr};
}
PyMODINIT_FUNC PyInit__native_core() {
    Object result(PyModule_Create(&module)); if (!result.value) return nullptr;
    Object type(PyType_FromSpec(&spec)); if (!type.value) return nullptr;
    if (PyModule_AddObject(result.value,"ArenaState",type.value)<0) return nullptr;
    type.release();
    if (!add_request_batch(result.value)) return nullptr;
    return result.release();
}
