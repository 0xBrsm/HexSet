// SPDX-License-Identifier: GPL-3.0-only
// Include after Core/checked/error inside state_native.cpp. These are owned
// debug copies; native request batches consume legal_core.hpp directly.
#pragma once
inline PyObject* core_legal_actions(Core* self,PyObject* args) {
    int seat=-1;if(!PyArg_ParseTuple(args,"|i",&seat)) return nullptr;
    try {
        auto actions=hexset_native::native_legal_actions(checked(self),seat);
        Object out(PyList_New(static_cast<Py_ssize_t>(actions.size())));
        if(!out.value) throw PythonError{};
        for(std::size_t i=0;i<actions.size();++i) {
            const auto& action=actions[i];
            auto* row=Py_BuildValue("(iLL)",action.type,static_cast<long long>(action.a),static_cast<long long>(action.b));
            if(!row) throw PythonError{};
            PyList_SET_ITEM(out.value,static_cast<Py_ssize_t>(i),row);
        }
        return out.release();
    } catch(...) {error();return nullptr;}
}
inline PyObject* core_decision(Core* self,PyObject* args) {
    int seat;if(!PyArg_ParseTuple(args,"i",&seat)) return nullptr;
    try {
        auto decision=hexset_native::native_decision(checked(self),seat);
        Object out(PyDict_New());if(!out.value) throw PythonError{};
        set(out.value,"flat",PyBytes_FromStringAndSize(reinterpret_cast<const char*>(decision.flat.data()),static_cast<Py_ssize_t>(decision.flat.size())));
        set(out.value,"setup",PyBytes_FromStringAndSize(reinterpret_cast<const char*>(decision.setup.data()),static_cast<Py_ssize_t>(decision.setup.size())));
        std::vector<std::int16_t> roads;roads.reserve(decision.roads.size()*2);
        for(const auto& row:decision.roads) for(auto value:row) roads.push_back(static_cast<std::int16_t>(value));
        set(out.value,"roads",PyBytes_FromStringAndSize(reinterpret_cast<const char*>(roads.data()),static_cast<Py_ssize_t>(roads.size()*sizeof(std::int16_t))));
        std::vector<std::int8_t> discards;discards.reserve(decision.discards.size()*10);
        for(const auto& row:decision.discards) for(auto value:row) discards.push_back(static_cast<std::int8_t>(value));
        set(out.value,"discard",PyBytes_FromStringAndSize(reinterpret_cast<const char*>(discards.data()),static_cast<Py_ssize_t>(discards.size())));
        return out.release();
    } catch(...) {error();return nullptr;}
}
