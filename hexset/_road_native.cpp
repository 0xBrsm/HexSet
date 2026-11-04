// SPDX-License-Identifier: GPL-3.0-only
// Integer-only graph queries. No hidden game state crosses this boundary.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <climits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using Ints = std::vector<int>;
using Rows = std::vector<Ints>;

struct Ref {
    PyObject* p;
    explicit Ref(PyObject* value) : p(value) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
};

// Unwind back into the binding without replacing the Python exception.
struct PythonError {};

Ints integers(PyObject* object) {
    Ref seq(PySequence_Fast(object, "expected a sequence of integers"));
    if (!seq.p) throw PythonError();
    Py_ssize_t size = PySequence_Fast_GET_SIZE(seq.p);
    if (size > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "graph sequence is too large");
        throw PythonError();
    }
    Ints out;
    out.reserve(size);
    for (Py_ssize_t i = 0; i < size; ++i) {
        long value = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq.p, i));
        if ((value == -1 && PyErr_Occurred()) || value > INT_MAX || value < INT_MIN) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_OverflowError, "graph integer is too large");
            throw PythonError();
        }
        out.push_back(static_cast<int>(value));
    }
    return out;
}

struct Topology {
    std::vector<std::pair<int, int>> edges;
    Rows incident;

    Topology(PyObject* pairs, int vertices) : incident(vertices) {
        Ref seq(PySequence_Fast(pairs, "expected edge endpoint pairs"));
        if (!seq.p) throw PythonError();
        Py_ssize_t size = PySequence_Fast_GET_SIZE(seq.p);
        if (size > INT_MAX) {
            PyErr_SetString(PyExc_OverflowError, "too many edges");
            throw PythonError();
        }
        for (Py_ssize_t e = 0; e < size; ++e) {
            Ints pair = integers(PySequence_Fast_GET_ITEM(seq.p, e));
            if (pair.size() != 2 || pair[0] < 0 || pair[1] < 0 ||
                pair[0] >= vertices || pair[1] >= vertices) {
                PyErr_SetString(PyExc_ValueError, "invalid edge endpoints");
                throw PythonError();
            }
            edges.emplace_back(pair[0], pair[1]);
            incident[pair[0]].push_back(e);
            incident[pair[1]].push_back(e);
        }
    }
};

constexpr const char* TOPOLOGY = "hexset.road_topology";

const Topology& get_topology(PyObject* object) {
    auto* result = static_cast<Topology*>(PyCapsule_GetPointer(object, TOPOLOGY));
    if (!result) throw PythonError();
    return *result;
}

struct Graph {
    const std::vector<std::pair<int, int>>& edges;
    const Rows& incident;
    Ints owners, buildings;

    Graph(PyObject* capsule, PyObject* edge_owner, PyObject* vertex_owner)
        : edges(get_topology(capsule).edges), incident(get_topology(capsule).incident),
          owners(integers(edge_owner)), buildings(integers(vertex_owner)) {
        if (owners.size() != edges.size() || buildings.size() != incident.size()) {
            PyErr_SetString(PyExc_ValueError, "ownership does not match topology");
            throw PythonError();
        }
    }

    bool passable(int vertex, int seat) const {
        return buildings[vertex] == -1 || buildings[vertex] == seat;
    }
};

struct Link {
    int index, far, cost;
    bool onward, joined;
};

struct Search {
    const Graph& graph;
    int seat, budget;
    int target = INT_MAX;
    bool keep_paths, found = false;
    Ints usable, starts, best;
    std::vector<std::vector<Link>> adjacent;
    std::vector<unsigned char> joins, used;
    std::vector<std::vector<unsigned char>> paths;
    size_t owned;

    Search(const Graph& g, int s, int roads, bool save_paths)
        : graph(g), seat(s), budget(roads), keep_paths(save_paths),
          best(roads + 1, 0), adjacent(g.buildings.size()), joins(g.buildings.size(), 0) {
        for (size_t e = 0; e < g.edges.size(); ++e) {
            if (g.owners[e] == s) {
                usable.push_back(e);
                const auto ends = g.edges[e];
                if (g.buildings[ends.first] == -1) joins[ends.first] = 1;
                if (g.buildings[ends.second] == -1) joins[ends.second] = 1;
            }
        }
        owned = usable.size();
        Ints frontier;
        for (size_t v = 0; v < joins.size(); ++v) {
            if (g.buildings[v] == s) joins[v] = 1;
            if (joins[v]) frontier.push_back(v);
        }
        std::vector<unsigned char> near(g.edges.size(), 0);
        for (int depth = 0; depth < roads && !frontier.empty(); ++depth) {
            Ints grown;
            std::vector<unsigned char> reached(g.buildings.size(), 0);
            for (int v : frontier) {
                for (int e : g.incident[v]) {
                    if (g.owners[e] != -1 || near[e]) continue;
                    near[e] = 1;
                    const auto ends = g.edges[e];
                    for (int w : {ends.first, ends.second}) {
                        if (g.passable(w, s) && !reached[w]) {
                            grown.push_back(w);
                            reached[w] = 1;
                        }
                    }
                }
            }
            frontier.swap(grown);
        }
        for (size_t e = 0; e < near.size(); ++e) if (near[e]) usable.push_back(e);
        used.resize(usable.size(), 0);
        if (keep_paths) paths.assign(roads + 1, used);
        // Edge and first-seen vertex order reproduce the Python DFS, including ties.
        for (size_t i = 0; i < usable.size(); ++i) {
            const auto ends = g.edges[usable[i]];
            int cost = i < owned ? 0 : 1;
            add(ends.first, ends.second, i, cost);
            add(ends.second, ends.first, i, cost);
        }
    }

    void add(int v, int w, int index, int cost) {
        if (adjacent[v].empty()) starts.push_back(v);
        adjacent[v].push_back({index, w, cost, graph.passable(w, seat), cost == 0 || joins[w] != 0});
    }

    void improve(int spent, int length) {
        if (length >= target) found = true;
        if (length > best[spent]) {
            best[spent] = length;
            if (keep_paths) paths[spent] = used;
        }
    }

    void walk(int v, int spent, int length, bool touched) {
        if (touched) improve(spent, length);
        for (const Link& link : adjacent[v]) {
            if (found) return;
            if (used[link.index] || spent + link.cost > budget) continue;
            used[link.index] = 1;
            bool reached = touched || link.joined;
            if (link.onward) walk(link.far, spent + link.cost, length + 1, reached);
            else if (reached) improve(spent + link.cost, length + 1);
            used[link.index] = 0;
        }
    }

    void run(int stop_at = INT_MAX) {
        target = stop_at;
        for (int v : starts) {
            walk(v, 0, 0, joins[v] != 0);
            if (found) return;
        }
        for (int j = 1; j <= budget; ++j) {
            if (best[j - 1] >= best[j]) {
                best[j] = best[j - 1];
                if (keep_paths) paths[j] = paths[j - 1];
            }
        }
    }
};

struct AllowThreads {
    PyThreadState* state;
    AllowThreads() : state(PyEval_SaveThread()) {}
    ~AllowThreads() { PyEval_RestoreThread(state); }
};

PyObject* integer_list(const Ints& values) {
    Ref out(PyList_New(values.size()));
    if (!out.p) throw PythonError();
    for (size_t i = 0; i < values.size(); ++i) {
        PyObject* value = PyLong_FromLong(values[i]);
        if (!value) throw PythonError();
        PyList_SET_ITEM(out.p, i, value);
    }
    Py_INCREF(out.p);
    return out.p;
}

// Translate failures at every entrypoint; C++ exceptions never cross Python.
template <class Call> PyObject* checked(Call call) {
    try { return call(); }
    catch (const PythonError&) { return nullptr; }
    catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    catch (const std::exception& e) { PyErr_SetString(PyExc_RuntimeError, e.what()); return nullptr; }
}

void delete_topology(PyObject* capsule) {
    delete static_cast<Topology*>(PyCapsule_GetPointer(capsule, TOPOLOGY));
}

PyObject* topology(PyObject*, PyObject* args) {
    return checked([&]() -> PyObject* {
        PyObject* pairs;
        int vertices;
        if (!PyArg_ParseTuple(args, "Oi", &pairs, &vertices)) return nullptr;
        if (vertices < 0) { PyErr_SetString(PyExc_ValueError, "negative vertex count"); return nullptr; }
        std::unique_ptr<Topology> graph(new Topology(pairs, vertices));
        PyObject* capsule = PyCapsule_New(graph.get(), TOPOLOGY, delete_topology);
        if (capsule) graph.release();
        return capsule;
    });
}

PyObject* longest(PyObject*, PyObject* args) {
    return checked([&]() -> PyObject* {
        PyObject *pairs, *owners, *buildings, *extra;
        int seat;
        if (!PyArg_ParseTuple(args, "OOOiO", &pairs, &owners, &buildings, &seat, &extra)) return nullptr;
        Graph graph(pairs, owners, buildings);
        if (extra != Py_None) {
            long edge = PyLong_AsLong(extra);
            if (edge == -1 && PyErr_Occurred()) return nullptr;
            if (edge < 0 || static_cast<size_t>(edge) >= graph.edges.size()) {
                PyErr_SetString(PyExc_IndexError, "hypothetical edge is out of range");
                return nullptr;
            }
            graph.owners[edge] = seat;
        }
        Search search(graph, seat, 0, false);
        { AllowThreads unlocked; search.run(); }
        if (PyErr_CheckSignals() < 0) return nullptr;
        return PyLong_FromLong(search.best[0]);
    });
}

PyObject* with_new(PyObject*, PyObject* args) {
    return checked([&]() -> PyObject* {
        PyObject *pairs, *owners, *buildings;
        int seat, roads, keep;
        if (!PyArg_ParseTuple(args, "OOOiip", &pairs, &owners, &buildings, &seat, &roads, &keep)) return nullptr;
        if (roads < 0) {
            PyErr_SetString(PyExc_ValueError, "road budget cannot be negative");
            return nullptr;
        }
        Graph graph(pairs, owners, buildings);
        Search search(graph, seat, roads, keep != 0);
        { AllowThreads unlocked; search.run(); }
        if (PyErr_CheckSignals() < 0) return nullptr;
        Ref best(integer_list(search.best));
        if (!keep) { Py_INCREF(best.p); return best.p; }
        Ref paths(PyList_New(roads + 1));
        if (!paths.p) throw PythonError();
        for (int j = 0; j <= roads; ++j) {
            Ints path;
            for (size_t i = search.owned; i < search.usable.size(); ++i) {
                if (search.paths[j][i]) path.push_back(search.usable[i]);
            }
            PyList_SET_ITEM(paths.p, j, integer_list(path));
        }
        return PyTuple_Pack(2, best.p, paths.p);
    });
}

PyObject* to_take(PyObject*, PyObject* args) {
    return checked([&]() -> PyObject* {
        PyObject *pairs, *owners, *buildings;
        int seat, beat, budget, cap;
        if (!PyArg_ParseTuple(args, "OOOiiii", &pairs, &owners, &buildings, &seat, &beat, &budget, &cap)) return nullptr;
        Graph graph(pairs, owners, buildings);
        int owned = std::count(graph.owners.begin(), graph.owners.end(), seat);
        int result = cap;
        {
            AllowThreads unlocked;
            for (int k = std::max(1, beat - owned); k <= budget; ++k) {
                Search search(graph, seat, k, false);
                search.run(beat);
                if (search.found) { result = k; break; }
            }
        }
        if (PyErr_CheckSignals() < 0) return nullptr;
        return PyLong_FromLong(result);
    });
}

PyObject* reach(PyObject*, PyObject* args) {
    return checked([&]() -> PyObject* {
        PyObject *pairs, *owners, *buildings;
        int players, max_roads;
        if (!PyArg_ParseTuple(args, "OOOii", &pairs, &owners, &buildings, &players, &max_roads)) return nullptr;
        if (players < 0) { PyErr_SetString(PyExc_ValueError, "negative player count"); return nullptr; }
        Graph graph(pairs, owners, buildings);
        Rows distances(players, Ints(graph.buildings.size(), -1));
        {
            AllowThreads unlocked;
            for (int seat = 0; seat < players; ++seat) {
                if (std::find(graph.buildings.begin(), graph.buildings.end(), seat) == graph.buildings.end()) continue;
                Ints frontier;
                Ints& dist = distances[seat];
                for (size_t v = 0; v < graph.buildings.size(); ++v) {
                    if (graph.buildings[v] == seat) frontier.push_back(v);
                }
                int left = max_roads;
                for (size_t e = 0; e < graph.edges.size(); ++e) {
                    if (graph.owners[e] == seat) {
                        --left;
                        frontier.push_back(graph.edges[e].first);
                        frontier.push_back(graph.edges[e].second);
                    }
                }
                for (int v : frontier) dist[v] = 0;
                for (int step = 1; !frontier.empty() && step <= left; ++step) {
                    Ints grown;
                    for (int v : frontier) {
                        if (!graph.passable(v, seat)) continue;
                        for (int e : graph.incident[v]) {
                            if (graph.owners[e] != -1) continue;
                            const auto ends = graph.edges[e];
                            int w = ends.first == v ? ends.second : ends.first;
                            if (dist[w] < 0) { dist[w] = step; grown.push_back(w); }
                        }
                    }
                    frontier.swap(grown);
                }
            }
        }
        if (PyErr_CheckSignals() < 0) return nullptr;
        Ref out(PyTuple_New(players));
        if (!out.p) throw PythonError();
        for (int seat = 0; seat < players; ++seat) {
            Ref row(integer_list(distances[seat]));
            PyObject* tuple = PyList_AsTuple(row.p);
            if (!tuple) throw PythonError();
            PyTuple_SET_ITEM(out.p, seat, tuple);
        }
        Py_INCREF(out.p);
        return out.p;
    });
}

PyMethodDef methods[] = {
    {"topology", topology, METH_VARARGS, "Pack immutable graph adjacency once."},
    {"longest", longest, METH_VARARGS, "Longest trail with an optional hypothetical road."},
    {"with_new", with_new, METH_VARARGS, "Longest trails within each new-road budget."},
    {"to_take", to_take, METH_VARARGS, "Minimum new-road budget reaching a target trail length."},
    {"reach", reach, METH_VARARGS, "Minimum road distances from every seat's network."},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_road_native", "Native integer road queries.", -1, methods};
} // namespace

PyMODINIT_FUNC PyInit__road_native() { return PyModule_Create(&module); }
