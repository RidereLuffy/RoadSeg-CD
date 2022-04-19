#!/usr/bin/env python3

"""
Codes are modified based on https://github.com/anilbatra2185/road_connectivity
"""

import math
import numpy as np
from skimage.morphology import skeletonize
import data_utils.graph_utils as graph_utils
import data_utils.sknw as sknw


def getKeypoints(mask, thresh=0.8, is_gaussian=True, is_skeleton=False, smooth_dist=4):
    """
    Generate keypoints for binary prediction mask.

    @param mask: Binary road probability mask
    @param thresh: Probability threshold used to cnvert the mask to binary 0/1 mask
    @param gaussian: Flag to check if the given mask is gaussian/probability mask
                    from prediction
    @param is_skeleton: Flag to perform opencv skeletonization on the binarized
                        road mask
    @param smooth_dist: Tolerance parameter used to smooth the graph using
                        RDP algorithm

    @return: return ndarray of road keypoints
    """

    if is_gaussian:
        mask /= 255.0
        mask[mask < thresh] = 0
        mask[mask >= thresh] = 1

    h, w = mask.shape
    if is_skeleton:
        ske = mask
    else:
        ske = skeletonize(mask).astype(np.uint16)
    graph = sknw.build_sknw(ske, multi=True)

    length_th = 10

    for i_edge in list(graph.edges): # remove the edges with same endpoints
        if i_edge[0]==i_edge[1]:
            graph.remove_edge(i_edge[0],i_edge[1])

    for i_edge in list(graph.edges): # remove the short edges as the burrs
        if graph.edges[i_edge]['weight'] < length_th:
            if len(graph.edges(i_edge[0]))==1 or len(graph.edges(i_edge[1]))==1:
                graph.remove_edge(i_edge[0],i_edge[1])

    for i_node in list(graph.nodes): # remove the nodes with no edge
        if not graph.edges(i_node):
            graph.remove_node(i_node)

    i_edge_wait_remove = []
    for i_edge in list(graph.edges): #remove the short edges in the crossroads
        if graph.edges[i_edge]['weight'] < length_th:
            if len(graph.edges(i_edge[0]))>2 and len(graph.edges(i_edge[1]))>2:
                i_edge_wait_remove.append(i_edge)
    for i_edge in i_edge_wait_remove:
        graph.remove_edge(i_edge[0],i_edge[1])

    segments = graph_utils.simplify_graph(graph, smooth_dist)
    linestrings_1 = graph_utils.segmets_to_linestrings(segments)
    linestrings = graph_utils.unique(linestrings_1)

    keypoints = []
    for line in linestrings:
        linestring = line.rstrip("\n").split("LINESTRING ")[-1]
        points_str = linestring.lstrip("(").rstrip(")").split(", ")
        ## If there is no road present
        if "EMPTY" in points_str:
            return keypoints
        points = []
        for pt_st in points_str:
            x, y = pt_st.split(" ")
            x, y = float(x), float(y)
            points.append([x, y])

        x1, y1 = points[0]
        x2, y2 = points[-1]

        if x2 < x1:
            keypoints.append(points[::-1])
        else:
            keypoints.append(points)
    return keypoints


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
    

def getVectorMapsAngles(road_mask, keypoints, theta=5, bin_size=10):
    """
    Convert Road keypoints obtained from road mask to orientation angle mask.
    Reference: Section 3.1
        https://anilbatra2185.github.io/papers/RoadConnectivityCVPR2019.pdf

    @param shape: Road Label/PIL image shape i.e. H x W
    @param keypoints: road keypoints generated from Road mask using
                        function getKeypoints()
    @param theta: thickness width for orientation vectors, it is similar to
                    thicknes of road width with which mask is generated.
    @param bin_size: Bin size to quantize the Orientation angles.

    @return: Return ndarray of shape H x W, containing orientation angles per pixel.
    """

    im_h, im_w = road_mask.shape
    vecmap = np.zeros((im_h, im_w, 2), dtype=np.float32)
    vecmap_angles = np.zeros((im_h, im_w), dtype=np.float32)
    vecmap_angles.fill(180)
    dismap = np.zeros((im_h, im_w), dtype=np.float32)
    dismap.fill(1e9)
    height, width, channel = vecmap.shape
    for j in range(len(keypoints)):
        for i in range(1, len(keypoints[j])):
            a = keypoints[j][i - 1]
            b = keypoints[j][i]
            ax, ay = a[0], a[1]
            bx, by = b[0], b[1]
            bax = bx - ax
            bay = by - ay
            if bay < 0:
                b = keypoints[j][i - 1]
                a = keypoints[j][i]
                ax, ay = a[0], a[1]
                bx, by = b[0], b[1]
                bax = bx - ax
                bay = by - ay
            norm = math.sqrt(1.0 * bax * bax + bay * bay)
            if norm == 0:
                norm = norm + 1e-9
            bax /= norm
            bay /= norm

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    if not road_mask[h,w]:
                        continue
                    if dismap[h, w]==0:
                        continue
                    dis = distP2Segment([w, h], [ax, ay], [bx, by])
                    if dismap[h, w]!=1e9:
                        if dismap[h, w] <= dis+1e-4:
                            continue
                    vecmap[h, w, 0] = bax
                    vecmap[h, w, 1] = bay
                    _theta = math.degrees(math.atan2(bay, bax))
                    vecmap_angles[h, w] = (_theta + 180) % 180
                    if vecmap_angles[h, w]>=180:
                        print('vecmap_angles[h, w]',vecmap_angles[h, w])
                        print('_theta', _theta)
                        print('h,w',h,w)
                        print('bay, bax',bay, bax)
                        print('ax', ax)
                        print('bx', bx)
                        print('a', a)
                        print('b', b)
                    dismap[h, w] = dis

    vecmap_angles = (vecmap_angles / bin_size).astype(int)
    print('angle labels: ', set(vecmap_angles.flatten().tolist()))
    return vecmap, vecmap_angles


def convertAngles2VecMap(shape, vecmapAngles):
    """
    Helper method to convert Orientation angles mask to Orientation vectors.

    @params shape: Road mask shape i.e. H x W
    @params vecmapAngles: Orientation agles mask of shape H x W
    @param bin_size: Bin size to quantize the Orientation angles.

    @return: ndarray of shape H x W x 2, containing x and y values of vector
    """

    h, w = shape
    vecmap = np.zeros((h, w, 2), dtype=np.float)

    for h1 in range(h):
        for w1 in range(w):
            angle = vecmapAngles[h1, w1]
            if angle < 36.0:
                angle *= 10.0
                if angle >= 180.0:
                    angle -= 360.0
                vecmap[h1, w1, 0] = math.cos(math.radians(angle))
                vecmap[h1, w1, 1] = math.sin(math.radians(angle))

    return vecmap


def convertVecMap2Angles(shape, vecmap, bin_size=10):
    """
    Helper method to convert Orientation vectors to Orientation angles.

    @params shape: Road mask shape i.e. H x W
    @params vecmap: Orientation vectors of shape H x W x 2

    @return: ndarray of shape H x W, containing orientation angles per pixel.
    """

    im_h, im_w = shape
    angles = np.zeros((im_h, im_w), dtype=np.float)
    angles.fill(360)

    for h in range(im_h):
        for w in range(im_w):
            x = vecmap[h, w, 0]
            y = vecmap[h, w, 1]
            angles[h, w] = (math.degrees(math.atan2(y, x)) + 360) % 360

    angles = (angles / bin_size).astype(int)
    return angles
