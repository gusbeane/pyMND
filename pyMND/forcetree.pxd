cdef struct dd:
    float[3] s 
    float mass
    int sibling
    int nextnode

cdef union uu:
    int [8] suns
    dd d

cdef struct NODE:
    float length
    float[3] center
    uu u

cdef class TREE(object):
    cdef NODE* Nodes
    cdef NODE* Nodes_base
    cdef int* Nextnode
    cdef int MaxNodes, MaxPart, NumPart
    cdef double[:,:] Pos
    cdef double[:] Mass
    cdef int last
    cdef double Theta, Softening
    cdef int empty_tree
    cdef _allocate(self)
