import numpy as np
cimport numpy as np
import sys

from libc.stdlib cimport malloc, free
import cython

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
    cdef NODE* Nodes
    cdef NODE* Nodes_base
    cdef int* Nextnode
    cdef int MaxNodes, MaxPart, NumPart
    cdef double[:,:] Pos
    cdef double[:] Mass
    cdef int last

    def __init__(self, MaxNodes, MaxPart, Pos, Mass):
        self.MaxNodes = MaxNodes
        self.MaxPart = MaxPart

        self._allocate()

        self.Pos = Pos
        self.Mass = Mass
        self.NumPart = len(Pos)

        self.last = 0
    
    cdef _allocate(self):
        self.Nodes_base = <NODE *> malloc((self.MaxNodes + 1) * sizeof(NODE))
        
        self.Nextnode = <int *> malloc(self.MaxPart * sizeof(int))

# @cython.wraparound(False)
# @cython.boundscheck(False)
# @cython.cdivision(True)
cdef force_treebuild(TREE tree):
    cdef int i, j, subnode = 0, parent = -1, numnodes
    cdef int nfree, th, nn
    cdef double lenhalf
    cdef NODE* nfreep
    cdef double length
    cdef double xmin[3], xmax[3]

    # global last, Nodes, Nodes_base, MaxNodes, MaxPart, NumPart, Nextwenode

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
    
    for j in range(3):
        print(xmin[j], xmax[j])

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
    force_update_node_recursive(tree.MaxPart, -1, tree)

    if (tree.last >= tree.MaxPart):
        tree.Nodes[tree.last].u.d.nextnode = -1
    else:
        tree.Nextnode[tree.last] = -1

    return tree

cdef force_update_node_recursive(int no, int sib, TREE tree):
    cdef int j, jj, p, pp = 0, nextsib
    cdef int suns[8]
    cdef double	s[3], mass

    # global last, Nodes, Nodes_base, MaxNodes, MaxPart, NumPart, Nextnode

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
                for jj in range(j+1, 8):
                    pp = suns[jj]
                    if (pp >= 0):
                        break

                if (jj < 8):	# /* yes, we do */
                    nextsib = pp
                else:
                    nextsib = sib

                force_update_node_recursive(p, nextsib, tree)

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

        if mass > 0:
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

# cdef force_treeevaluate(double[:] pos, double softening, double[:] acc):
#     struct NODE    *nop = 0
#     int no, ninteractions;
#     double r2, dx, dy, dz, mass, r, fac, u, h, h_inv, h3_inv
#     double acc_x, acc_y, acc_z, pos_x, pos_y, pos_z

#     acc_x = 0
#     acc_y = 0
#     acc_z = 0
#     ninteractions = 0

#     pos_x = pos[0]
#     pos_y = pos[1]
#     pos_z = pos[2]

#     h = 2.8 * softening
#     h_inv = 1.0 / h
#     h3_inv = h_inv * h_inv * h_inv

#     no = MaxPart		#/* first node */

#     while (no >= 0) {
#         if (no < MaxPart) {
#             # /* the index of the node is the index of the particle */
#             dx = P[no].Pos[0] - pos_x
#             dy = P[no].Pos[1] - pos_y
#             dz = P[no].Pos[2] - pos_z

#             r2 = dx * dx + dy * dy + dz * dz;

#             mass = P[no].Mass;

#             no = Nextnode[no];
#         } else {	/* we have an  internal node */
#             nop = &Nodes[no];

#             dx = nop->u.d.s[0] - pos_x;
#             dy = nop->u.d.s[1] - pos_y;
#             dz = nop->u.d.s[2] - pos_z;

#             r2 = dx * dx + dy * dy + dz * dz;

#             mass = nop->u.d.mass;

#             /*
#              * printf("nop->len=%g  r=%g  %g %g\n", nop->len,
#              * sqrt(r2), nop->len * nop->len , r2 * ErrTolTheta *
#              * ErrTolTheta);
#              */

#             if (nop->len * nop->len > r2 * ErrTolTheta * ErrTolTheta) {
#                 /* open cell */
#                 no = nop->u.d.nextnode;
#                 continue;
#             }
#             /* check in addition whether we lie inside the cell */

#             if (fabs(nop->center[0] - pos_x) < 0.60 * nop->len) {
#                 if (fabs(nop->center[1] - pos_y) < 0.60 * nop->len) {
#                     if (fabs(nop->center[2] - pos_z) < 0.60 * nop->len) {
#                         no = nop->u.d.nextnode;
#                         continue;
#                     }
#                 }
#             }
#             no = nop->u.d.sibling;	/* ok, node can be used */
#         }

#         r = sqrt(r2);

#         if (r >= h)
#             fac = mass / (r2 * r);
#         else {
#             u = r * h_inv;
#             if (u < 0.5)
#                 fac = mass * h3_inv * (10.666666666667 + u * u * (32.0 * u - 38.4));
#             else
#                 fac =
#                     mass * h3_inv * (21.333333333333 - 48.0 * u +
#                              38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u));
#         }

#         acc_x += dx * fac;
#         acc_y += dy * fac;
#         acc_z += dz * fac;

#         ninteractions++;
#     }

#     /* store result at the proper place */

#     acc[0] = acc_x;
#     acc[1] = acc_y;
#     acc[2] = acc_z;

#     return ninteractions;

# cdef force_treeallocate(int maxnodes, int maxpart):
#     global MaxNodes, MaxPart, Nextnode, Nodes_base
#     cdef int allbytes = 0
    
#     MaxNodes = maxnodes
#     MaxPart = maxpart

#     Nodes_base = <NODE *> malloc((MaxNodes + 1) * sizeof(NODE))
#     allbytes += (MaxNodes + 1) * sizeof(NODE)

#     Nextnode = <int *> malloc(MaxPart * sizeof(int))
#     allbytes += MaxPart * sizeof(int)

#     print("Use {} MByte for BH-tree.".format(allbytes / (1024.0 * 1024.0)))

# cdef force_treefree():
#     global Nextnode, Nodes_base
#     free(Nextnode)
#     free(Nodes_base)


def construct_tree(pos, mass):
    global NumPart
    maxpart = pos.shape[0]
    NumPart = maxpart
    maxnodes = int(1.5 * maxpart)
    
    tree = TREE(maxnodes, maxpart, pos, mass)

    tree = force_treebuild(tree)
    return tree