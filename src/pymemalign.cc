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
    // TODO: must be implemented
    return true;
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

PyObject *NewUninitializedAlignedPyArray(PyObject *self, PyObject *args) {
    PyObject *shape = nullptr;
    PyObject *type = nullptr;
    // TODO: this must be obtained from function arg
    size_t alignment = 32u;
//    printf("REFCOUNT: 0 shape %ld\n", (long)Py_REFCNT(&args[1]));
    if (!PyArg_ParseTuple(args, "OO", &type, &shape)) {
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
    switch(nptype) {
    case(NPY_BOOL): {
        element_size = sizeof(bool);
    }
    break;
    case(NPY_INT8): {
        element_size = sizeof(int8_t);
    }
    break;
    case(NPY_INT32): {
        element_size = sizeof(int32_t);
    }
    break;
    case(NPY_INT64): {
        element_size = sizeof(int64_t);
    }
    break;
    case(NPY_UINT8): {
        element_size = sizeof(uint8_t);
    }
    break;
    case(NPY_UINT32): {
        element_size = sizeof(uint32_t);
    }
    break;
    case(NPY_FLOAT): {
        element_size = sizeof(float);
    }
    break;
    case(NPY_DOUBLE): {
        element_size = sizeof(double);
    }
    break;
    case(NPY_LONGDOUBLE): {
        element_size = sizeof(long double);
    }
    break;
    default:
        // unsupported type
        PyErr_SetString(PyExc_ValueError, "Unsupported data type.");
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
