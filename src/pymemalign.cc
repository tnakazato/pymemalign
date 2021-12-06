#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <unistd.h>
#include <memory>

/*
 * Refer to following documents:
 * https://docs.python.org/3/extending/extending.html
 * https://docs.python.org/3/c-api/capsule.html
 * https://docs.python.org/3/c-api/arg.html
 *
 */

#define MODULE_NAME "_pymemalign"

#undef KEEP_GIL

#ifdef KEEP_GIL
# define BEGIN_ALLOW_THREADS {
# define END_ALLOW_THREADS }
#else
# define BEGIN_ALLOW_THREADS Py_BEGIN_ALLOW_THREADS
# define END_ALLOW_THREADS Py_END_ALLOW_THREADS
#endif

namespace {



constexpr char const kAlignedPyArrayName[] =
        MODULE_NAME ".AlignedPyArray";
}

extern "C" {

}

namespace {

struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

#define LONG_FROM_SSIZE_T(s) PyLong_FromSsize_t(s)
#define ASLONG(s) PyLong_AsLong(s)

void DecrementRef(PyObject *obj) {
    Py_XDECREF(obj);
}

class RefHolder: public std::unique_ptr<PyObject, decltype(&DecrementRef)> {
public:
    explicit RefHolder(PyObject *obj) :
            std::unique_ptr<PyObject, decltype(&DecrementRef)>(obj,
                    DecrementRef) {
    }
};

bool isAligned(void *p, size_t alignment) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    // printf("addr = %lu, align = %lu mod = %lu\n", addr, alignment, addr % alignment);
    return addr % alignment == 0;
}

// For numpy array holding aligned pointer
// destructor for PyCapsule
void DestructPyCapsuleForNP(PyObject *obj) {
    if (!PyCapsule_IsValid(obj, kAlignedPyArrayName)) {
        return;
    }
    void *ptr = PyCapsule_GetPointer(obj, kAlignedPyArrayName);
    //printf("LOG: Deallocate aligned buffer for numpy address is %p\n", ptr);
    free(ptr);
}

bool getTypeInfo(PyObject *type_obj, int &nptype, size_t &element_size) {
    if (!(PyType_Check(type_obj))) {
        return false;
    }
    PyTypeObject *typeObj = (PyTypeObject *)type_obj;
    auto const type_name = typeObj->tp_name;
    if (strcmp(type_name, "numpy.bool") == 0) {
        nptype = NPY_BOOL;
        element_size = 8;
    } else if (strcmp(type_name, "numpy.int8") == 0) {
        nptype = NPY_INT8;
        element_size = 8;
    } else if (strcmp(type_name, "numpy.int16") == 0) {
        nptype = NPY_INT16;
        element_size = 16;
    } else if (strcmp(type_name, "numpy.int32") == 0) {
        nptype = NPY_INT32;
        element_size = 32;
    } else if (strcmp(type_name, "numpy.int64") == 0) {
        nptype = NPY_INT64;
        element_size = 64;
    } else if (strcmp(type_name, "numpy.uint8") == 0) {
        nptype = NPY_UINT8;
        element_size = 8;
    } else if (strcmp(type_name, "numpy.uint16") == 0) {
        nptype = NPY_UINT16;
        element_size = 16;
    } else if (strcmp(type_name, "numpy.uint32") == 0) {
        nptype = NPY_UINT32;
        element_size = 32;
    } else if (strcmp(type_name, "numpy.uint64") == 0) {
        nptype = NPY_UINT64;
        element_size = 64;
    } else if (strcmp(type_name, "numpy.float16") == 0) {
        nptype = NPY_FLOAT16;
        element_size = 16;
    } else if (strcmp(type_name, "numpy.float32") == 0) {
        nptype = NPY_FLOAT32;
        element_size = 32;
    } else if (strcmp(type_name, "numpy.float64") == 0) {
        nptype = NPY_FLOAT64;
        element_size = 64;
    } else if (strcmp(type_name, "numpy.complex64") == 0) {
        nptype = NPY_COMPLEX64;
        element_size = 64;
    } else if (strcmp(type_name, "numpy.complex128") == 0) {
        nptype = NPY_COMPLEX128;
        element_size = 128;
    } else {
        return false;
    }
    return true;
}

PyObject *NewUninitializedAlignedPyArray(PyObject *self, PyObject *args) {
    PyObject *shape = nullptr;
    PyObject *type = nullptr;
    // TODO: this must be obtained from function arg
    size_t alignment = 32u;
//    printf("REFCOUNT: 0 shape %ld\n", (long)Py_REFCNT(&args[1]));
    if (!PyArg_ParseTuple(args, "OOl", &type, &shape, &alignment)) {
        return nullptr;
    }

    // type should be PyTypeObject
    if (!PyType_Check(type)) {
        PyErr_SetString(PyExc_ValueError, "First argument should be a data type.");
    }

    // shape should be a tuple
//    printf("REFCOUNT: 1 shape %ld\n", (long)Py_REFCNT(shape));
    int is_tuple = PyTuple_Check(shape);
    if (!is_tuple) {
        PyErr_SetString(PyExc_ValueError, "Second argument should be a shape tuple.");
        return nullptr;
    }

    // numpy shape
    Py_ssize_t len = PyTuple_Size(shape);
//    printf("LOG: tuple size %ld\n", (long)len);
    std::unique_ptr<npy_intp[]> dims(new npy_intp[len]);
    for (Py_ssize_t i = 0; i < len; ++i) {
        auto item = PyTuple_GetItem(shape, i);
        dims[i] = (npy_intp)PyLong_AsLong(item);
//        printf("LOG: tuple item (%ld) %ld\n", i, dims[i]);
    }

    // create numpy array from alinged pointer
    // assume type is float at this moment
    ssize_t num_elements = 1;
    for (Py_ssize_t i = 0; i < len; ++i) {
        num_elements *= dims[i];
    }

    // type mapping
    size_t element_size = 0;
    int nptype = NPY_FLOAT;
    if (!(getTypeInfo(type, nptype, element_size))) {
        PyErr_SetString(PyExc_ValueError, "Failed to get type information from dtype. Maybe unsupported type.");
        return nullptr;
    }

    // allocate memory with aligned pointer
    void *p = nullptr;
    int status = posix_memalign(&p, alignment, num_elements * element_size);
    //printf("LOG: allocate memory address is %p\n", p);
    if (status != 0 || !isAligned(p, alignment)) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory.");
        return nullptr;
    }
    // encapsulate pointer
    RefHolder capsule(PyCapsule_New(p, kAlignedPyArrayName, DestructPyCapsuleForNP));
//    printf("REFCOUNT: capsule %ld\n", (long)Py_REFCNT(capsule));

    // create array
    RefHolder arr(PyArray_SimpleNewFromData(len, dims.get(), nptype, p));
//    printf("REFCOUNT: arr %ld\n", (long)Py_REFCNT(arr));
    if (!PyArray_Check(arr.get())) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create ndarray.");
        return nullptr;
    }

    // set capsule object as a base object of the array
    int s = PyArray_SetBaseObject((PyArrayObject *)(arr.get()), capsule.get());
    if (s != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to set base for ndarray.");
        return nullptr;
    }

    // give ownership of capsule object to arr
    capsule.release();

//    printf("REFCOUNT: capsule %ld\n", (long)Py_REFCNT(capsule));
//
//    printf("REFCOUNT: 2 shape %ld\n", (long)Py_REFCNT(shape));

    return arr.release();
}
//





PyMethodDef module_methods[] =
        {

                { "allocate", NewUninitializedAlignedPyArray,
                        METH_VARARGS, "Creates a new aligned numpy ndarray."},

                { NULL, NULL, 0, NULL } /* Sentinel */
        };

PyDoc_STRVAR(module_doc, "Python wrapper for posix_memalign.");

}

extern "C" {

static int module_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int module_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    MODULE_NAME,
    module_doc,
    sizeof(struct module_state),
    module_methods,
    NULL,
    module_traverse,
    module_clear,
    NULL
};

#define INITERROR return NULL

PyMODINIT_FUNC
PyInit__pymemalign(void) {
    PyObject *mod = PyModule_Create(&moduledef);
    if (mod == nullptr) {
        INITERROR;
    }

    // to use NumPy C-API
    import_array();

    struct module_state *st = GETSTATE(mod);

    static char excep_name[] = MODULE_NAME ".error";
    static char excep_doc[] = "error on invoking pymemalign functions";
    auto py_error = st->error;
    py_error = PyErr_NewExceptionWithDoc(excep_name, excep_doc, nullptr,
            nullptr);
    if (py_error == NULL) {
        DecrementRef(mod);
        INITERROR;
    }

    return mod;
}

}
