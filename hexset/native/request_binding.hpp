// SPDX-License-Identifier: GPL-3.0-only
// Included after Core/checked/Object/error; call add_request_batch at module init.
#pragma once
#include "request_core.hpp"
struct NativeRequestBatch {
    PyObject_HEAD
    PyObject* cores=nullptr;
    RequestBuffers* buffers=nullptr;
};
const ArenaState& request_lane(NativeRequestBatch* self,Int lane) {
    ensure(self->cores && self->buffers,"uninitialized native request batch");
    ensure(lane>=0 && lane<PyTuple_GET_SIZE(self->cores),"invalid native request lane");
    return checked(reinterpret_cast<Core*>(PyTuple_GET_ITEM(self->cores,static_cast<Py_ssize_t>(lane))));
}
std::vector<const ArenaState*> request_states(NativeRequestBatch* self) {
    ensure(self->cores && self->buffers,"uninitialized native request batch");
    std::vector<const ArenaState*> states;
    auto n=PyTuple_GET_SIZE(self->cores);states.reserve(static_cast<std::size_t>(n));
    for(Py_ssize_t row=0;row<n;++row) states.push_back(&checked(reinterpret_cast<Core*>(PyTuple_GET_ITEM(self->cores,row))));
    return states;
}
PyObject* request_new(PyTypeObject* type,PyObject*,PyObject*) {
    auto* self=reinterpret_cast<NativeRequestBatch*>(type->tp_alloc(type,0));
    if(self) {self->cores=nullptr;self->buffers=nullptr;} return reinterpret_cast<PyObject*>(self);
}
void request_free(NativeRequestBatch* self) {
    delete self->buffers;Py_XDECREF(self->cores);Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}
int request_init(NativeRequestBatch* self,PyObject* args,PyObject* kwargs) {
    PyObject* sequence;const char* contract="9";
    static const char* names[]={"cores","contract",nullptr};
    if(!PyArg_ParseTupleAndKeywords(args,kwargs,"O|s",const_cast<char**>(names),&sequence,&contract)) return -1;
    try {
        ensure(!self->buffers,"native request batch cannot be reinitialized");
        Object cores(PySequence_Tuple(sequence));if(!cores.value) throw PythonError{};
        Object module(PyImport_ImportModule("hexset._native_core"));if(!module.value) throw PythonError{};
        Object core_type(PyObject_GetAttrString(module.value,"ArenaState"));if(!core_type.value) throw PythonError{};
        std::vector<const ArenaState*> states;
        for(Py_ssize_t row=0;row<PyTuple_GET_SIZE(cores.value);++row) {
            auto* item=PyTuple_GET_ITEM(cores.value,row);int ok=PyObject_IsInstance(item,core_type.value);
            if(ok<0) throw PythonError{};
            ensure(ok!=0,"request batch requires native ArenaState lanes");
            states.push_back(&checked(reinterpret_cast<Core*>(item)));
        }
        auto prepared=std::make_unique<RequestBuffers>(states,contract);
        self->cores=cores.release();self->buffers=prepared.release();return 0;
    } catch(...) {error();return -1;}
}
template<class T> PyObject* request_bytes(const std::vector<T>& values) {
    return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(values.data()),static_cast<Py_ssize_t>(values.size()*sizeof(T)));
}
PyObject* request_fill(NativeRequestBatch* self,PyObject* object) {
    try {
        auto states=request_states(self);Object rows(PySequence_Fast(object,"native identities must be rows"));
        if(!rows.value) throw PythonError{};
        std::vector<Int> ids;
        for(Py_ssize_t row=0;row<PySequence_Fast_GET_SIZE(rows.value);++row) {
            auto values=integers(PySequence_Fast_GET_ITEM(rows.value,row));ensure(values.size()==7,"native identity needs seven columns");
            ids.insert(ids.end(),values.begin(),values.end());
        }
        auto& b=*self->buffers;b.fill(states,ids);
        Object result(PyDict_New());if(!result.value) throw PythonError{};
        set(result.value,"epoch",PyLong_FromUnsignedLongLong(b.epoch));
        set(result.value,"contract",PyUnicode_FromString(b.contract.c_str()));
        set(result.value,"shape",tuple({static_cast<Int>(b.rows()),static_cast<Int>(b.width),static_cast<Int>(b.flat_width),static_cast<Int>(b.setup_width)}));
        set(result.value,"ids",request_bytes(b.ids));set(result.value,"observation",request_bytes(b.observations));
        set(result.value,"flat",request_bytes(b.flat));set(result.value,"setup",request_bytes(b.setup));
        set(result.value,"rb",request_bytes(b.roads));set(result.value,"rb_offsets",request_bytes(b.road_offsets));
        set(result.value,"discard",request_bytes(b.discards));set(result.value,"discard_offsets",request_bytes(b.discard_offsets));
        return result.release();
    } catch(...) {error();return nullptr;}
}
PyObject* request_check(NativeRequestBatch* self,PyObject* object) {
    try {auto states=request_states(self);self->buffers->check(states,static_cast<std::uint64_t>(integer(object)));Py_RETURN_NONE;}
    catch(...) {error();return nullptr;}
}
PyObject* request_finish(NativeRequestBatch* self,PyObject* object) {
    try {
        request_states(self);auto epoch=integer(object);auto& b=*self->buffers;
        ensure(b.outstanding && epoch>=0 && static_cast<std::uint64_t>(epoch)==b.epoch,"stale native request epoch");
        b.outstanding=false;Py_RETURN_NONE;
    } catch(...) {error();return nullptr;}
}
PyObject* request_targets(NativeRequestBatch* self,PyObject* object) {
    try {
        auto states=request_states(self);auto& b=*self->buffers;b.check(states,static_cast<std::uint64_t>(integer(object)));
        std::vector<Int> points,turns;
        for(std::size_t row=0;row<b.rows();++row) {
            const auto& state=*states[b.ids[row*7+1]];turns.push_back(state.get(Block::game_control)[TurnsColumn]);
            for(std::size_t seat=0;seat<b.players;++seat) points.push_back(true_points(state,static_cast<int>(seat)));
        }
        Object result(PyDict_New());if(!result.value) throw PythonError{};
        set(result.value,"points",request_bytes(points));set(result.value,"turns",request_bytes(turns));return result.release();
    } catch(...) {error();return nullptr;}
}
PyObject* actor_tuple(const ArenaState& state) {
    const auto& game=state.get(Block::game_control);std::vector<Int> actors;
    if(game[PhaseColumn]!=6) {
        if(game[PhaseColumn]==3) {
            const auto& quota=state.get(Block::discard_quota);
            for(std::size_t seat=0;seat<quota.size();++seat) if(quota[seat]>0) actors.push_back(static_cast<Int>(seat));
        }
        if(actors.empty()) actors.push_back(request_actor(state));
    }
    return tuple(actors);
}
PyObject* request_actors_at(NativeRequestBatch* self,PyObject* object) {
    try {return actor_tuple(request_lane(self,integer(object)));}
    catch(...) {error();return nullptr;}
}
PyObject* request_actors(NativeRequestBatch* self,PyObject*) {
    try {
        auto states=request_states(self);Object result(PyTuple_New(static_cast<Py_ssize_t>(states.size())));
        if(!result.value) throw PythonError{};
        for(std::size_t lane=0;lane<states.size();++lane) {
            PyTuple_SET_ITEM(result.value,static_cast<Py_ssize_t>(lane),actor_tuple(*states[lane]));
        }
        return result.release();
    } catch(...) {error();return nullptr;}
}
PyObject* request_closes(NativeRequestBatch* self,PyObject* args) {
    long long row;int seat;PyObject* types;
    if(!PyArg_ParseTuple(args,"LiO",&row,&seat,&types)) return nullptr;
    try {
        ensure(self->cores && self->buffers,"uninitialized native request batch");
        auto& b=*self->buffers;
        ensure(b.outstanding && row>=0 && static_cast<std::size_t>(row)<b.rows(),"invalid native closure row");
        return PyBool_FromLong(option_closes(request_lane(self,b.ids[row*7+1]),seat,integers(types)));
    } catch(...) {error();return nullptr;}
}
PyObject* request_replace(NativeRequestBatch* self,PyObject* args) {
    long long lane;PyObject* core;
    if(!PyArg_ParseTuple(args,"LO",&lane,&core)) return nullptr;
    try {
        auto states=request_states(self);auto& b=*self->buffers;
        ensure(!b.outstanding,"cannot replace a lane with unanswered native requests");
        ensure(lane>=0 && static_cast<std::size_t>(lane)<states.size(),"native replacement lane out of range");
        ensure(PyObject_TypeCheck(core,Py_TYPE(PyTuple_GET_ITEM(self->cores,0))),"native replacement requires ArenaState");
        const auto& state=checked(reinterpret_cast<Core*>(core));
        RequestBuffers validate({states.front(),&state},b.contract);
        Object cores(PyTuple_New(static_cast<Py_ssize_t>(states.size())));if(!cores.value) throw PythonError{};
        for(std::size_t row=0;row<states.size();++row) {
            auto* item=row==static_cast<std::size_t>(lane) ? core : PyTuple_GET_ITEM(self->cores,static_cast<Py_ssize_t>(row));
            Py_INCREF(item);PyTuple_SET_ITEM(cores.value,static_cast<Py_ssize_t>(row),item);
        }
        b.imported_states[lane]=&state;
        Py_DECREF(self->cores);self->cores=cores.release();Py_RETURN_NONE;
    } catch(...) {error();return nullptr;}
}
PyMethodDef request_methods[]={
 {"fill",reinterpret_cast<PyCFunction>(request_fill),METH_O,"Pack visible rows; immutable returned buffers own their lifetimes."},
 {"check",reinterpret_cast<PyCFunction>(request_check),METH_O,"Validate request epoch and every lane's native state generation."},
 {"finish",reinterpret_cast<PyCFunction>(request_finish),METH_O,"Release answered requests; the next fill reuses native capacities."},
 {"targets",reinterpret_cast<PyCFunction>(request_targets),METH_O,"Privileged labels only; never in the visible request packet."},
 {"actors",reinterpret_cast<PyCFunction>(request_actors),METH_NOARGS,"All presently acting seats; simultaneous discard seats retain ascending order."},
 {"actors_at",reinterpret_cast<PyCFunction>(request_actors_at),METH_O,"Presently acting seats for one lane, without scanning the batch."},
 {"option_closes",reinterpret_cast<PyCFunction>(request_closes),METH_VARARGS,"Native decision-run closure after successful atomic parts."},
 {"replace",reinterpret_cast<PyCFunction>(request_replace),METH_VARARGS,"Replace a retired lane, preserving native request capacities and epoch."},
 {nullptr,nullptr,0,nullptr}
};
PyType_Slot request_slots[]={
 {Py_tp_new,reinterpret_cast<void*>(request_new)},{Py_tp_init,reinterpret_cast<void*>(request_init)},
 {Py_tp_dealloc,reinterpret_cast<void*>(request_free)},{Py_tp_methods,reinterpret_cast<void*>(request_methods)},{0,nullptr}
};
PyType_Spec request_spec={"hexset._native_core.RequestBatch",sizeof(NativeRequestBatch),0,Py_TPFLAGS_DEFAULT,request_slots};
bool add_request_batch(PyObject* module) {
    Object type(PyType_FromSpec(&request_spec));if(!type.value) return false;
    if(PyModule_AddObject(module,"RequestBatch",type.value)<0) return false;
    type.release();return true;
}
