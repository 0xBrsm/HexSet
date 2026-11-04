// SPDX-License-Identifier: GPL-3.0-only
// Include inside state_native.cpp after Core, checked() and error().
// Add core_encode9 to that module's method table; there is no Python callback.
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
