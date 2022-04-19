#!/usr/bin/env python3


"""
The Ramer-Douglas-Peucker algorithm roughly ported from the pseudo-code provided
by http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm

The code is taken from
https://github.com/mitroadmaps/roadtracer/blob/master/lib/discoverlib/rdp.py
"""

import math


def distP2Segment(P, A, B):
    """
    Compute the minimum distance between P and segment AB
    @param P: P(w, h)
    @param A: A(ax, ay)
    @param B: B(bx, by)
    @return: Return the distance
    """
    
    w, h = P[0], P[1]
    ax, ay = A[0], A[1]
    bx, by = B[0], B[1]
    px = w*1.0 - ax*1.0
    py = h*1.0 - ay*1.0 #AP
    bax = bx*1.0 - ax*1.0
    bay = by*1.0 - ay*1.0 #AB
    if A==B:
        return math.sqrt(px*px + py*py)
    r = (px*bax + py*bay)/(bax*bax + bay*bay)
    if r<=0:
        return math.sqrt(px*px + py*py)
    elif r>=1:
        pbx = w*1.0 - bx*1.0
        pby = h*1.0 - by*1.0 #BP
        return math.sqrt(pbx*pbx + pby*pby)
    else:
        norm = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9
        bax /= norm
        bay /= norm
        return abs(bax * py - bay * px)


def rdp(points, epsilon):
    """
    Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.

    @param points: Series of points for a line geometry represnted in graph.
    @param epsilon: Tolerance required for RDP algorithm to aproximate the
                    line geometry.

    @return: Approximate series of points for approximate line geometry
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = distP2Segment(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d
    if dmax >= epsilon:
        results = rdp(points[: index + 1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]
    return results
