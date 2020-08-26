import numpy as np
from math import sqrt

class Node(object):
    def __init__(self, center, length):
        self.suns = [-1, -1, -1, -1, -1, -1, -1, -1]
        self.center = center
        self.length = length

        self.com = np.array([0, 0, 0])
        self.mass = 0.0

class Tree(object):
    def __init__(self, center, length, points, soft):
        self.center = center
        self.length = length
        self.points = points
        
        self.Npart = len(points)

        self.nodes = [Node(self.center, self.length)]
        self.Nnodes = 0

        self.soft = soft
        self.h = 2.8 * soft

        for p_idx in range(self.Npart):
            self._add_point(p_idx)
    
    def _add_point(self, p_idx):
        p = self.points[p_idx]
        # Start at parent node.
        node_idx = 0
        node = self.nodes[node_idx]

        while True:
            oct_idx, new_cen, new_len = self._determine_oct_idx(p, node.center, node.length)
            
            node.com = (node.mass * node.com + p) / (node.mass + 1.0)
            node.mass += 1.0
            
            if node.suns[oct_idx] == -1:
                # Sun is empty, we can add the point.
                node.suns[oct_idx] = p_idx
                print('nice', p_idx)
                break
            elif node.suns[oct_idx] >= self.Npart:
                # Sun points to another node, repeat procedure.
                node = self.nodes[node.suns[oct_idx] - self.Npart]
                continue
            elif node.suns[oct_idx] < self.Npart and node.suns[oct_idx] >= 0:
                # Sun points to a single particle, need to replace with a new node.
                new_node = Node(new_cen, new_len)

                self.nodes.append(new_node)
                self.Nnodes += 1

                old_pidx = node.suns[oct_idx]
                old_p = self.points[old_pidx]

                new_node.mass = 1.0
                new_node.com = old_p

                node.suns[oct_idx] = self.Npart + self.Nnodes
                new_oct_idx, _, _ = self._determine_oct_idx(old_p, new_node.center, new_node.length)
                new_node.suns[new_oct_idx] = old_pidx
                node = new_node
                continue
            else:
                raise ValueError

    def _determine_oct_idx(self, pos, cen, length):
        oct_idx = 0
        half_length = length/2.0
        new_cen = cen - np.array([length, length, length])/4.0
        if pos[0] > cen[0]:
            new_cen[0] += half_length
            oct_idx += 1
        if pos[1] > cen[1]:
            new_cen[1] += half_length
            oct_idx += 2
        if pos[2] > cen[2]:
            new_cen[2] += half_length
            oct_idx += 4
        
        return oct_idx, new_cen, half_length
    
    def force_evaluate(self, pos, theta=1.0):
        return self._eval(self.nodes[0], pos, self.h, theta, self._force_kernel)

    def _eval(self, node, p, h, theta, kernel):
        force = np.array([0.0, 0.0, 0.0])

        if node.mass == 0.0:
            return force
        
        dx = node.com[0] - p[0]
        dy = node.com[1] - p[1]
        dz = node.com[2] - p[2]

        r = sqrt(dx*dx + dy*dy + dz*dz)
        u = r/h

        #If we satisfy the opening angle criterion, we end here
        if r > node.length/theta and u > 1.0:
            force += node.mass * kernel(r, h, dx, dy, dz)
        else:
            # Otherwise, we open the node. We check for leaves and evaluate, otherwise we restart force calc.
            for i in range(8):
                if node.suns[i] == -1:
                    continue
                elif node.suns[i] < self.Npart:
                    leaf_p = self.points[node.suns[i]]

                    dx = leaf_p[0] - p[0]
                    dy = leaf_p[1] - p[1]
                    dz = leaf_p[2] - p[2]
                    r = sqrt(dx*dx + dy*dy + dz*dz)
                    force += 1.0 * kernel(r, h, dx, dy, dz)
                else:
                    new_node = self.nodes[node.suns[i] - self.Npart]
                    force += self._eval(new_node, p, h, theta, kernel)
        
        return force

    def _force_kernel(self, r, h, dx, dy, dz):
        u = r/h

        if u > 1.0:
            fac = 1. / (r*r*r)
        elif u > 0.5:
            fac =  (1./(h*h*h)) * (21.333333333333 - 48.0 * u +
							 38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u))
        else:
            fac = (1./(h*h*h)) * (10.666666666667 + u * u * (32.0 * u - 38.4))

        return np.array([fac * dx, fac * dy, fac * dz])


if __name__ == '__main__':
    np.random.seed(10)

    N = 1000
    soft = 0.3/2.8
    h = 2.8 * soft
    points = np.random.rand(N, 3)
    t = Tree(np.array([0.5, 0.5, 0.5]), 1.0, points, soft)
    
    eval_pos = np.array([0.5, 0.5, 0.5])

    tree_force = t.force_evaluate(eval_pos)

    force = np.zeros(3)
    # do direct evaluation
    for p in points:
        dx = p[0] - eval_pos[0]
        dy = p[1] - eval_pos[1]
        dz = p[2] - eval_pos[2]
        r = np.sqrt(dx*dx + dy*dy + dz*dz)

        u = r/h
        if u > 1.0:
            fac = 1. / (r*r*r)
        elif u > 0.5:
            fac =  (1./(h*h*h)) * (21.333333333333 - 48.0 * u +
							 38.4 * u * u - 10.666666666667 * u * u * u - 0.066666666667 / (u * u * u))
        else:
            fac = (1./(h*h*h)) * (10.666666666667 + u * u * (32.0 * u - 38.4))

        force[0] += dx * fac
        force[1] += dy * fac
        force[2] += dz * fac

    print(force)
    print(tree_force)
