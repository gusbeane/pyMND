import numpy as np
cimport numpy as np
import sys

from libc.math cimport sqrt,fabs

from libc.stdlib cimport malloc, free
import cython
from cython.parallel import prange

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

# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.cdivision(True)
cdef class TREE(object):
    def __init__(self, MaxNodes, MaxPart, Pos, Mass, Theta, Softening):
        self.MaxNodes = MaxNodes
        self.MaxPart = MaxPart

        self._allocate()

        #self.Pos = Pos
        #self.Mass = Mass
        self.NumPart = len(Pos)
        
        self.Mass = <double *> malloc(self.NumPart*sizeof(double))
        for i in range(self.NumPart):
            self.Mass[i] = Mass[i]

        self.Pos = <double **> malloc(self.NumPart*sizeof(double*))
        for i in range(self.NumPart):
            self.Pos[i] = <double *> malloc(3*sizeof(double))
            for j in range(3):
                self.Pos[i][j] = Pos[i][j]

        if self.NumPart > 0:
            self.empty_tree = 0
        else:
            self.empty_tree = 1

        self.Theta = Theta
        self.Softening = Softening

        self.last = 0
    
    cdef _allocate(self):
        self.Nodes_base = <NODE *> malloc((self.MaxNodes + 1) * sizeof(NODE))
        
        self.Nextnode = <int *> malloc(self.MaxPart * sizeof(int))

cdef force_treebuild(TREE tree):
    cdef int i, j, subnode = 0, parent = -1, numnodes
    cdef int nfree, th, nn
    cdef double lenhalf
    cdef NODE* nfreep
    cdef double length
    cdef double xmin[3], xmax[3]

    tree.Nodes = tree.Nodes_base - tree.MaxPart

    # select first node
    nfree = tree.MaxPart
    nfreep = &tree.Nodes[nfree]

    # /* create an empty  root node  */

    for j in range(3):	#/* find enclosing rectangle */
        xmin[j] = tree.Pos[0][j]
        xmax[j] = xmin[j]


    for i in range(1, tree.NumPart):
        for j in range(3):
            if tree.Pos[i][j] > xmax[j]:
                xmax[j] = tree.Pos[i][j]
            if (tree.Pos[i][j] < xmin[j]):
                xmin[j] = tree.Pos[i][j]
    
    # determine maximum extension
    length = xmax[0] - xmin[0]
    for j in range(1, 3):
        if xmax[j] - xmin[j] > length:
            length = xmax[j] - xmin[j]

    for j in range(3):
        nfreep.center[j] = (xmin[j] + xmax[j]) / 2.
    nfreep.length = length

    for i in range(8):
        nfreep.u.suns[i] = -1

    numnodes = 1
    nfreep += 1
    nfree += 1

    nfreep = &tree.Nodes[nfree]

    # Insert all particles
    for i in range(tree.NumPart):
        th = tree.MaxPart

        while True:
            if (th >= tree.MaxPart):	# /* we are dealing with an
                         #     * internal node */
                subnode = 0
                if tree.Pos[i][0] > tree.Nodes[th].center[0]:
                    subnode += 1
                if tree.Pos[i][1] > tree.Nodes[th].center[1]:
                    subnode += 2
                if tree.Pos[i][2] > tree.Nodes[th].center[2]:
                    subnode += 4

                nn = tree.Nodes[th].u.suns[subnode]

                if (nn >= 0):	#/* ok, something is in the
                         #* daughter slot already,
                         #* need to continue */
                    parent = th;	#/* note: subnode can
                            # * still be used in the
                            # * next step of the walk */
                    th = nn
                else:
    #                 /*
    #                  * here we have found an empty slot
    #                  * where we can attach the new
    #                  * particle as a leaf
    #                  */
                    tree.Nodes[th].u.suns[subnode] = i
                    break
            else:
    #             /*
    #              * we try to insert into a leaf with a single
    #              * particle need to generate a new internal
    #              * node at this point
    #              */
                tree.Nodes[parent].u.suns[subnode] = nfree

                nfreep.length = 0.5 * tree.Nodes[parent].length
                lenhalf = 0.25 * tree.Nodes[parent].length

                if subnode & 1:
                    nfreep.center[0] = tree.Nodes[parent].center[0] + lenhalf
                else:
                    nfreep.center[0] = tree.Nodes[parent].center[0] - lenhalf

                if subnode & 2:
                    nfreep.center[1] = tree.Nodes[parent].center[1] + lenhalf
                else:
                    nfreep.center[1] = tree.Nodes[parent].center[1] - lenhalf

                if subnode & 4:
                    nfreep.center[2] = tree.Nodes[parent].center[2] + lenhalf
                else:
                    nfreep.center[2] = tree.Nodes[parent].center[2] - lenhalf

                for j in range(8):
                    nfreep.u.suns[j] = -1

                subnode = 0
                if (tree.Pos[th][0] > nfreep.center[0]):
                    subnode += 1
                if (tree.Pos[th][1] > nfreep.center[1]):
                    subnode += 2
                if (tree.Pos[th][2] > nfreep.center[2]):
                    subnode += 4

                nfreep.u.suns[subnode] = th

                th = nfree;	#/* resume trying to insert
    #                      * the new particle at the
    #                      * newly created internal
    #                      * node */

                numnodes += 1
                nfree += 1
                nfreep += 1

                if ((numnodes) >= tree.MaxNodes):
                    print("maximum number {} of tree-nodes reached.".format(tree.MaxNodes))
                    print("for particle {}".format(i))
                    sys.exit(1)


    # /* now compute the multipole moments recursively */
    tree.last = -1
    tree = force_update_node_recursive(tree.MaxPart, -1, tree)

    if (tree.last >= tree.MaxPart):
        tree.Nodes[tree.last].u.d.nextnode = -1
    else:
        tree.Nextnode[tree.last] = -1

    return tree

cdef force_update_node_recursive(int no, int sib, TREE tree):
    cdef int j, jj, p, pp = 0, nextsib
    cdef int suns[8]
    cdef double	s[3], mass
    cdef int found_sibling

    if no >= tree.MaxPart:	# /* internal node */
        for j in range(8):
            suns[j] = tree.Nodes[no].u.suns[j]	#/* this "backup" is
                             #* necessary because the
                             #* nextnode entry will
                             #* overwrite one element
                             #* (union!) */
        if tree.last >= 0:
            if tree.last >= tree.MaxPart:
                tree.Nodes[tree.last].u.d.nextnode = no
            else:
                tree.Nextnode[tree.last] = no
        tree.last = no


        mass = 0
        s[0] = 0
        s[1] = 0
        s[2] = 0

        for j in range(8):
            p = suns[j]
            if (p >= 0):
                # /*
                #  * check if we have a sibling on the same
                #  * level
                #  */
                found_sibling = 0
                for jj in range(j+1, 8):
                    pp = suns[jj]
                    if (pp >= 0):
                        found_sibling = 1
                        break

                if found_sibling == 1:	# /* yes, we do */
                    nextsib = pp
                else:
                    nextsib = sib

                tree = force_update_node_recursive(p, nextsib, tree)

                if (p >= tree.MaxPart): # {	/* an internal node or
                             #* pseudo particle */
                    mass += tree.Nodes[p].u.d.mass #	/* we assume a fixed
                                     # * particle mass */
                    s[0] += tree.Nodes[p].u.d.mass * tree.Nodes[p].u.d.s[0]
                    s[1] += tree.Nodes[p].u.d.mass * tree.Nodes[p].u.d.s[1]
                    s[2] += tree.Nodes[p].u.d.mass * tree.Nodes[p].u.d.s[2]
                else:	# /* a particle */
                    mass += tree.Mass[p]
                    s[0] += tree.Mass[p] * tree.Pos[p][0]
                    s[1] += tree.Mass[p] * tree.Pos[p][1]
                    s[2] += tree.Mass[p] * tree.Pos[p][2]

        if mass > 0.0:
            s[0] /= mass
            s[1] /= mass
            s[2] /= mass
        else:
            s[0] = tree.Nodes[no].center[0]
            s[1] = tree.Nodes[no].center[1]
            s[2] = tree.Nodes[no].center[2]

        tree.Nodes[no].u.d.s[0] = s[0]
        tree.Nodes[no].u.d.s[1] = s[1]
        tree.Nodes[no].u.d.s[2] = s[2]
        tree.Nodes[no].u.d.mass = mass

        tree.Nodes[no].u.d.sibling = sib
    else:		#/* single particle */
        if tree.last >= 0:
            if (tree.last >= tree.MaxPart):
                tree.Nodes[tree.last].u.d.nextnode = no
            else:
                tree.Nextnode[tree.last] = no
        tree.last = no
    
    return tree

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double * _force_treeevaluate(double* pos, TREE tree) nogil:
    cdef NODE *nop
    cdef int no, ninteractions
    cdef double r2, dx, dy, dz, mass, r, fac, u, h, h_inv, h3_inv
    cdef double acc_x, acc_y, acc_z, pos_x, pos_y, pos_z
    cdef double * acc

    acc = <double *> malloc(3*sizeof(double)) 

    if tree.empty_tree == 1:
        acc[0] = 0.0
        acc[1] = 0.0
        acc[2] = 0.0
        return acc

    acc_x = 0
    acc_y = 0
    acc_z = 0
    ninteractions = 0

    pos_x = pos[0]
    pos_y = pos[1]
    pos_z = pos[2]

    h = 2.8 * tree.Softening
    h_inv = 1.0 / h
    h3_inv = h_inv * h_inv * h_inv

    no = tree.MaxPart		#/* first node */

    while no >= 0:
        if no < tree.MaxPart:
            # /* the index of the node is the index of the particle */
            dx = tree.Pos[no][0] - pos_x
            dy = tree.Pos[no][1] - pos_y
            dz = tree.Pos[no][2] - pos_z

            r2 = dx * dx + dy * dy + dz * dz

            mass = tree.Mass[no]

            no = tree.Nextnode[no]
        else:	#/* we have an  internal node */
            nop = &tree.Nodes[no]

            dx = nop.u.d.s[0] - pos_x
            dy = nop.u.d.s[1] - pos_y
            dz = nop.u.d.s[2] - pos_z

            r2 = dx * dx + dy * dy + dz * dz

            mass = nop.u.d.mass

            if (nop.length * nop.length > r2 * tree.Theta * tree.Theta):
                # /* open cell */
                no = nop.u.d.nextnode
                continue
            # /* check in addition whether we lie inside the cell */

            if (fabs(nop.center[0] - pos_x) < 0.60 * nop.length):
                if (fabs(nop.center[1] - pos_y) < 0.60 * nop.length):
                    if (fabs(nop.center[2] - pos_z) < 0.60 * nop.length):
                        no = nop.u.d.nextnode
                        continue

            no = nop.u.d.sibling #	/* ok, node can be used */

        r = sqrt(r2)

        if (r >= h):
            fac = mass / (r2 * r)
        else:
            u = r * h_inv
            if (u < 0.5):
                fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4))
            else:
                fac = mass * h3_inv * (21.333333333333 - 48.0 * u +
                             38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u))

        acc_x += dx * fac
        acc_y += dy * fac
        acc_z += dz * fac

        ninteractions += 1

    # /* store result at the proper place */

    acc[0] = acc_x
    acc[1] = acc_y
    acc[2] = acc_z

    return acc

cpdef force_treeevaluate(double [:] pos, TREE tree):
    cdef double *posc, *acc 
    cdef int i
    posc = <double *> malloc(3*sizeof(double))
    acc = <double *> malloc(3*sizeof(double))

    for i in range(3):
        posc[i] = pos[i]
    
    acc = _force_treeevaluate(posc, tree)

    cdef double [3] acc_out
    
    for i in range(3):
        acc_out[i] = acc[i]
    return acc_out

@cython.boundscheck(False)
cpdef _force_treeevaluate_loop(double[:,:] pos, int N, TREE tree, int num_threads):
    cdef double ** acc, **posc
    cdef double * acci, *posi
    cdef int i, j
    cdef double [:,:] acc_out

    # coerce python memory views into c arrays
    # needed in order to release GIL for prange
    acc = <double **> malloc(N * sizeof(double*))
    posc = <double **> malloc(N * sizeof(double*))
    for i in range(N):
        acc[i] = <double *> malloc(3 * sizeof(double))
        posc[i] = <double *> malloc(3 * sizeof(double))
        for j in range(3):
            posc[i][j] = pos[i][j]
    
    acci = <double *> malloc(3 * sizeof(double))
    posi = <double *> malloc(3 * sizeof(double))


    with nogil:
        for i in prange(N, num_threads=num_threads):
            posi[0] = posc[i][0]
            posi[1] = posc[i][1]
            posi[2] = posc[i][2]

            acci = _force_treeevaluate(posi, tree)
            acc[i][0] = acci[0]
            acc[i][1] = acci[1]
            acc[i][2] = acci[2]

    acc_out = np.zeros((N, 3))
    for i in range(N):
        for j in range(3):
            acc_out[i][j] = acc[i][j]

    return acc_out


cpdef force_treeevaluate_loop(pos, tree, num_threads=1):
    N = len(pos)
    return _force_treeevaluate_loop(pos, N, tree, num_threads)

cpdef construct_tree(pos, mass, theta, softening):
    maxpart = pos.shape[0]
    NumPart = maxpart
    maxnodes = int(1.5 * maxpart)
    
    tree = TREE(maxnodes, maxpart, pos, mass, theta, softening)

    tree = force_treebuild(tree)
    return tree

cpdef construct_empty_tree():
    maxnodes = 0
    maxpart = 0
    pos = np.array([]).reshape((0,0))
    mass = np.array([])
    theta = 1.0
    softening = 1.0
    tree = TREE(maxnodes, maxpart, pos, mass, theta, softening)
    return tree
