#! /usr/bin/env python

import numpy as np
from lamination import Lamination
from collections import namedtuple
import sys

Face = namedtuple("Face", ["vertices", "degree"])

class MarkedMultCover(Lamination):
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
                if len(cycle:=self.orbit(angle)) == self.period
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
                tuple(map(
                    lambda x: self.cycles[x],
                    angles))
                for angles in self.ray_sets
                ]

        # Vertices, labeled by minimum cycle representative
        self.vertices = sorted(set(self.cycles.values()))

        # Primitive leaves of lamination,
        # labeled by minimum cycle representative
        self.edges = [
                (a,b) for a,b in self.cycle_pairs
                if a != b
                ]

        # Faces, labeled by cycle class representative
        self.faces = {
                angle:
                    self.get_face(angle)
                    for angle in set(self.cycle_classes.values())
                }

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
        return {len(f.vertices) for f in self.faces.values()}

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
            for (a,b) in self.edges:
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
        for v in self.vertices:
            print(f"{indent_str}{v}")

        print(f"\n{len(self.edges)} edges:")
        for e in self.edges:
            print(f"{indent_str}{e}")

        print(f"\n{len(self.faces)} faces:")
        for p, face in self.faces.items():
            print(f"{indent_str}[{p:0>{self.period}b}] = {face}")

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
