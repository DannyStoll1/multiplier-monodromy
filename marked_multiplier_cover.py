#! /usr/bin/env python

import numpy as np
from lamination import Lamination
from collections import namedtuple
import sys
import networkx as nx
import matplotlib.pyplot as plt
from math import sin, cos, pi
from cmath import exp


_Face = namedtuple("Face", ["vertices", "degree"])


class Face(_Face):
    def edges(self):
        return list(zip(
            self.vertices,
            self.vertices[1:] + (self.vertices[0],)
        ))

    def __len__(self):
        return len(self.vertices)


class PlanarCycle:
    def __init__(self, face,
                 *args,
                 **kwargs,
                 ):
        self.face = face
        self.pos, self.v_labels = self.place(*args, **kwargs)

    def place(self,
              base_edge=(0, 1),
              base_loc=0,
              base_dir=-1j,
              edge_length=1,
              ):

        n = len(self.face.vertices)

        t0, t1 = base_edge
        r = 1/(2*sin(pi/edge_length))

        if t0 == (t1-1) % n:
            t0, t1 = t1, t0

#         pos = {
#                 t: (x0 + r*cos(((t-t0)*2-1)*pi/n),
#                     y0 - r*sin(((t-t0)*2-1)*pi/n))
#                 for t,v in enumerate(self.face.vertices)
#                 }
        pos = {}
        v_labels = {}

        dir = base_dir
        loc = base_loc - dir*edge_length/2

        for t in tuple(range(t0, n)) + tuple(range(t0)):
            pos[t] = (loc.real, loc.imag)
            dir *= exp(-2j*pi/n)
            loc += dir

            v_labels[t] = self.face.vertices[t]

        return pos, v_labels

    def __len__(self):
        return len(self.face.vertices)


class Tessellation:
    def __init__(faces=[], edges=[], vertices=[]):
        self.faces = faces
        self.edges = edges
        self.vertices = vertices

    def euler_characteristic(self):
        chi = \
            len(self.vertices) - \
            len(self.edges) + \
            len(self.faces)

        # It had better be even!
        assert chi % 2 == 0

        return chi

    def genus(self):
        return 1 - self.euler_characteristic()//2

    def face_sizes(self):
        return [len(f.vertices) for f in self.faces.values()]

    def num_odd_faces(self):
        return len([s for s in self.face_sizes() if s % 2])

    def show(self):
        fig, ax = plt.subplots()
        ax.set_aspect(1)

        x = 0j

        seen_real_edges = {}

        for name, face in self.faces.items():
            for i, edge in enumerate(face.edges):
                a, b = edge.endpoints
                if (a, b) in self.real_edges:
                    if (a, b) in seen_real_edges.keys():
                        base_face, base_edge = seen_real_edges[(a, b)]
                    else:
                        seen_real_edges[(a, b)] = (name, i)
                        base_face = None
                        base_edge = None

            planar_face = PlanarCycle(face, base_loc=x)
            self.draw_face(name, planar_face, fig, ax,
                           )
            x += 2.5

        plt.show()

    def draw_face(self, name, planar_face, fig, ax,
                  ):
        pos, vertex_labels = planar_face.pos, planar_face.v_labels
        n = len(planar_face)

        G = nx.Graph()

        G.add_nodes_from(range(n))

        G.add_edges_from(
            zip(tuple(range(n)),
                tuple(range(1, n)) + (0,))
        )

        nx.draw_networkx_nodes(G, pos,
                               node_size=120,
                               node_color='#ffffff',
                               )
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos,
                                labels=vertex_labels,
                                font_size=10,
                                font_family='sans-serif')


class Edge:
    def __init__(self, angles, cycle_classes, period, degree=2):
        self.endpoints = cycle_classes
        self.angles = angles
        self.period = period
        self.degree = degree
        self.is_real = (sum(self.angles) == self.degree ** self.period - 1)

    def __str__(self):
        n = self.period
        m = len(str(self.degree**self.period))
        t = self.angles[0]
        a, b = self.endpoints
        return f"{t:>{n}b} = {a:>{m}d} -- {b:<{m}d}"

    def __repr__(self):
        a, b = self.endpoints
        s, t = self.angles
        return f"Edge ({a}, {b}) with representatives ({s}, {t})"


class MarkedMultCover(Lamination, Tessellation):
    """
    Represents the combinatorics of the branched cover
        (c,lambda) -> c
    sending a unicritical polynomial f(z) = z^d + c
    with marked n-cycle multiplier lambda to the
    unmarked polynomial.

    The branched cover is described by a tessellation of
    a Riemann surface, whose vertices and faces
    may be labeled according to the marked cycles.

    `period`:
        The period of the cycles whose multipliers
        we follow

    `degree`:
        The degree d of the unicritical polynomial
        f(z) = z^d + c.

        TODO: UNTESTED FOR DEGREES GREATER THAN 2
    """

    def __init__(self, period=4, degree=2):
        period = int(period)
        degree = int(degree)

        super().__init__(period, degree)
        self.period = period
        self.max_angle = self.degree**self.period-1

        if self.period == 1 or self.degree > 2:
            raise NotImplementedError

        # Leaves of period p in Multibrot lamination
        self.ray_sets = sorted([
            tuple(map(
                lambda x: int(x*(self.max_angle)),
                angles))
            for angles in self.arcs_of_period(period)
        ])

        # Map each angle to the minimum angle in its cycle
        self.cycles = {
            angle: min(cycle)
            for angle in range(self.max_angle)
            if len(cycle := self.orbit(angle)) == self.period
        }

        # Map each angle to the minimum angle in its cycle class
        self.cycle_classes = {
            angle: min(
                self.cycles[angle],
                self.cycles.get(self.max_angle - angle, 0))
            for angle in self.cycles.keys()
        }

        # Leaves of lamination, labeled by minimum cycle representative
        self.cycle_pairs = [
            (
                angles,
                tuple(map(
                    lambda x: self.cycles[x],
                    angles))
            )
            for angles in self.ray_sets
        ]

        # Vertices, labeled by minimum cycle representative
        self.vertices = sorted(set(self.cycles.values()))

        # Primitive leaves of lamination,
        # labeled by minimum cycle representative
        self.edges = [
            Edge(t, (a, b),
                 period=self.period,
                 degree=self.degree,
                 )
            for (t, (a, b)) in self.cycle_pairs
            if a != b
        ]

        # Faces, labeled by cycle class representative
        self.faces = {
            angle:
            self.get_face(angle)
            for angle in set(self.cycle_classes.values())
        }

        self.real_edges = {
            endpoints
            for edge in self.edges
            for endpoints in [
                edge.endpoints,
                tuple(reversed(edge.endpoints))
            ]
            if edge.is_real
        }

    def orbit(self, angle):
        """
        Compute the orbit of an angle under
        multiplication by the degree
        """

        return {
            roll(angle, self.period, dist, self.degree)
            for dist in range(self.period)
        }

    def get_face(self, angle):
        start = self.cycles[angle]

        node = start
        nodes = [node]

        deg = 1

        while True:
            for edge in self.edges:
                a, b = edge.endpoints
                if node == a:
                    node = b
                    nodes.append(node)
                elif node == b:
                    node = a
                    nodes.append(node)

            if node == start:
                nodes = tuple(nodes)
                if len(nodes) > 1:
                    return Face(nodes[:-1], deg)
                return Face(nodes, deg)

            deg += 1

    def summarize(self, indent=4):
        indent_str = ' '*indent

        print(f"\n{len(self.vertices)} vertices:")
        n = self.period
        m = len(str(self.degree**n))
        for v in self.vertices:
            print(f"{indent_str}{v:>{m}d}")
            # print(f"{indent_str}{v:0>{n}b}")

        print(f"\n{len(self.edges)} edges:")
        for edge in self.edges:
            # print(f"{indent_str}{a:0>{n}b} - {b:0>{n}b}")
            print(f"{indent_str}{edge}")

        print(f"\n{len(self.faces)} faces:")
        for p, face in self.faces.items():
            print(f"{indent_str}[{p:0>{n}b}] = {face}")

        print("\nFace sizes:")
        print(f"{indent_str}{self.face_sizes()}")

        print(f"\nSmallest face: {min(self.face_sizes())}")
        print(f"\nLargest face: {max(self.face_sizes())}")
        print(f"\nGenus is {self.genus()}")


def binary_roll(x, period, dist=1):
    max_angle = (1 << period) - 1
    x %= max_angle
    dist %= period
    return ((x << dist) | (x >> (max_angle - dist))) % max_angle


def roll(x, period, dist=1, d=2):
    if dist == 0:
        return x
    if d == 2:
        return binary_roll(x, period, dist)

    max_angle = (d ** period) - 1
    x %= max_angle
    dist %= period

    left_shift = (x * d**dist) % max_angle
    right_shift = x // (d**(max_angle - dist))
    return left_shift + right_shift


if __name__ == "__main__":
    try:
        period = int(sys.argv[1])
    except:
        period = 7

    print((
        f"Computing combinatorics of "
        f"(c,lambda) -> c cover "
        f"for period {period}"
    ))

    cov = MarkedMultCover(period)

    cov.summarize()

    # cov.show()
