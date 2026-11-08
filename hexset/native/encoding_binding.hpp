// SPDX-License-Identifier: GPL-3.0-only
// Include inside state_native.cpp after Core, checked() and error().
// Add core_encode9 and core_encode8 to that module's method table; there is no Python callback.
#pragma once
inline PyObject* core_encode9(Core* self,PyObject* args) {
    int perspective;const char* contract="9";
    if (!PyArg_ParseTuple(args,"i|s",&perspective,&contract)) return nullptr;
    try {
        auto row=hexset_native::encode9(checked(self),perspective,contract);
        return PyByteArray_FromStringAndSize(reinterpret_cast<const char*>(row.data()),
                                             static_cast<Py_ssize_t>(row.size()*sizeof(float)));
    } catch (...) {error();return nullptr;}
}
inline PyObject* core_encode8(Core* self,PyObject* args) {
    int perspective;const char* contract="8";
    if (!PyArg_ParseTuple(args,"i|s",&perspective,&contract)) return nullptr;
    try {
        auto row=hexset_native::encode8(checked(self),perspective,contract);
        return PyByteArray_FromStringAndSize(reinterpret_cast<const char*>(row.data()),
                                             static_cast<Py_ssize_t>(row.size()*sizeof(float)));
    } catch (...) {error();return nullptr;}
}
inline PyObject* core_record8(Core* self,PyObject* args) {
    int perspective;if (!PyArg_ParseTuple(args,"i",&perspective)) return nullptr;
    try {
        auto record=hexset_native::record8(checked(self),perspective);
        Object out(PyDict_New());if (!out.value) throw PythonError{};
        for (const auto& [name,values]:record.ints) {
            Object bytes(PyBytes_FromStringAndSize(reinterpret_cast<const char*>(values.data()),static_cast<Py_ssize_t>(values.size()*sizeof(hexset_native::Int))));
            if (!bytes.value || PyDict_SetItemString(out.value,name.c_str(),bytes.value)<0) throw PythonError{};
        }
        for (const auto& [name,values]:record.floats) {
            Object bytes(PyBytes_FromStringAndSize(reinterpret_cast<const char*>(values.data()),static_cast<Py_ssize_t>(values.size()*sizeof(float))));
            if (!bytes.value || PyDict_SetItemString(out.value,name.c_str(),bytes.value)<0) throw PythonError{};
        }
        return out.release();
    } catch (...) {error();return nullptr;}
}
