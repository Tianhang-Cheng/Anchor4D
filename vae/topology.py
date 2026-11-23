"""Mesh topology for the anchor VAE: connected components, geodesic landmarks, anchors and
vertex-to-anchor geodesic distances.

All distances are exact shortest paths on the edge graph of the welded frame-0 mesh,
computed per connected component. Only connectivity is cached across calls; edge lengths,
landmarks, anchors and distances are recomputed for every geometry.
"""
import hashlib
import os

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

_CACHE = None

# Share of the anchor budget used to give every (large enough) component one anchor first.
COMPONENT_FIRST_FRAC = 0.5


def connectivity(n, faces):
    """Unique undirected edges, components and per-component CSR layout (cached on the face bytes)."""
    global _CACHE
    faces = np.ascontiguousarray(faces)
    key = (n, faces.shape, faces.dtype.str, hashlib.sha256(faces).digest())
    if _CACHE is not None and _CACHE[0] == key:
        return _CACHE[1]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    keys = np.unique(edges[:, 0].astype(np.int64) * n + edges[:, 1])
    edges = np.stack([keys // n, keys % n], axis=1).astype(edges.dtype)
    edges = edges[edges[:, 0] != edges[:, 1]]
    directed = np.concatenate([edges, edges[:, ::-1]])
    directed = directed[np.lexsort((directed[:, 1], directed[:, 0]))]
    adj = coo_matrix((np.ones(len(directed)), (directed[:, 0], directed[:, 1])), shape=(n, n)).tocsr()
    nc, component = connected_components(adj, directed=False)
    order = np.argsort(component, kind='stable')
    starts = np.searchsorted(component[order], np.arange(nc + 1))
    members = [order[starts[c]:starts[c + 1]] for c in range(nc)]
    local_id = np.empty(n, np.int64)
    for m in members:
        local_id[m] = np.arange(len(m))
    edge_order = np.argsort(component[directed[:, 0]], kind='stable')
    edge_starts = np.searchsorted(component[directed[edge_order, 0]], np.arange(nc + 1))
    degree = np.diff(adj.indptr)
    parts = []
    for c, m in enumerate(members):
        ei = edge_order[edge_starts[c]:edge_starts[c + 1]]
        ptr = np.r_[0, np.cumsum(degree[m])].astype(adj.indptr.dtype)
        idx = local_id[directed[ei, 1]].astype(adj.indices.dtype)
        parts.append((ei, idx, ptr))
    result = dict(directed=directed, component=component, members=members, parts=parts)
    _CACHE = key, result
    return result


def weighted_graphs(vertices, structure):
    """One edge-length CSR graph per component (component-local vertex ids)."""
    directed = structure['directed']
    lengths = np.maximum(np.linalg.norm(vertices[directed[:, 0]] - vertices[directed[:, 1]], axis=1), 1e-10)
    return [csr_matrix((lengths[ei].astype(np.float64), idx, ptr), shape=(len(ptr) - 1,) * 2)
            for ei, idx, ptr in structure['parts']]


def landmark_rows(graph, count):
    """Geodesic farthest-point sampling of `count` landmarks inside one component."""
    nearest = np.full(graph.shape[0], np.inf)
    nxt = 0
    ids = []
    geo = np.empty((count, graph.shape[0]), np.float32)
    raw = np.empty((count, graph.shape[0]), np.float16)
    for i in range(count):
        dist = dijkstra(graph, indices=nxt, directed=True)
        ids.append(nxt)
        geo[i] = np.where(np.isfinite(dist), dist, 0)
        raw[i] = dist
        nearest = np.minimum(nearest, dist)
        nxt = int(np.argmax(nearest))
    return ids, geo, raw


def _distance_block(graph, sources):
    return dijkstra(graph, indices=sources, directed=True).astype(np.float16)


def _jobs(tasks, n):
    """Run independent SciPy jobs, in a small CPU pool for large workloads (results keep task order).

    ANCHOR4D_GEO_WORKERS=1 disables multiprocessing."""
    import multiprocessing
    workers = max(1, int(os.environ.get('ANCHOR4D_GEO_WORKERS', '4')))
    work = sum(args[0].shape[0] * (int(args[1]) if np.isscalar(args[1]) else len(args[1]))
               for _, args in tasks)
    if multiprocessing.current_process().daemon or work < 500_000:
        workers = 1
    if workers == 1 or len(tasks) < 2:
        for fn, args in tasks:
            yield fn(*args)
        return
    from joblib import Parallel, delayed, parallel_config
    with parallel_config(backend='loky', inner_max_num_threads=1):
        blocks = Parallel(n_jobs=min(workers, len(tasks)), return_as='generator',
                          max_nbytes='1M', mmap_mode='r')(
            delayed(fn)(*args) for fn, args in tasks)
        yield from blocks


def component_landmarks(graphs, count, n):
    return _jobs([(landmark_rows, (g, k)) for g, k in zip(graphs, count)], n)


def anchor_blocks(graphs, sources, n):
    """Geodesic rows from the given sources, in component-local blocks of at most 64 MiB."""
    tasks, indices = [], []
    for c, (graph, src) in enumerate(zip(graphs, sources)):
        block = max(1, min(32, (64 << 20) // (8 * max(1, graph.shape[0]))))
        for s in range(0, len(src), block):
            chunk = src[s:s + block]
            indices.append((c, chunk))
            tasks.append((_distance_block, (graph, chunk)))
    for (c, src), rows in zip(indices, _jobs(tasks, n)):
        yield c, src, rows


def landmark_budget(vertices, faces, component, landmarks):
    """Largest-remainder split of the landmarks over components by surface area."""
    area = 0.5 * np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                         vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
    weight = np.bincount(component[faces[:, 0]], area, minlength=component.max() + 1)
    if weight.sum() <= 0:
        weight = np.bincount(component).astype(np.float64)
    size = np.bincount(component, minlength=len(weight))
    total = min(landmarks, len(vertices))
    share = total * weight / weight.sum()
    count = np.minimum(np.floor(share).astype(np.int64), size)
    while count.sum() < total:
        open_ = count < size
        count[np.argmax(np.where(open_, share - count, -np.inf))] += 1
    return count


def component_fps(vertices, faces, component, K, seeds, members):
    """K anchors: the landmarks (`seeds`) first, then one anchor per uncovered component in decreasing area order
    (up to COMPONENT_FIRST_FRAC * K of them), then Euclidean farthest-point sampling within components.
    May return fewer than K ids when the mesh has fewer vertices."""
    nc = int(component.max()) + 1
    area = 0.5 * np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                         vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1)
    carea = np.bincount(component[faces[:, 0]], area, minlength=nc)
    points = [vertices[m] for m in members]
    distances = [np.full(len(m), np.inf) for m in members]
    has = np.zeros(nc, bool)
    picked = []
    best_distance = np.full(nc, -1.)
    best_vertex = np.array([m[0] for m in members], dtype=np.int64)

    def update_candidate(c):
        local = np.where(np.isinf(distances[c]), -1., distances[c])
        idx = int(np.argmax(local))
        best_distance[c], best_vertex[c] = local[idx], members[c][idx]

    def add(v, refresh=True):
        c = component[v]
        distances[c] = np.minimum(distances[c], np.linalg.norm(points[c] - vertices[v], axis=1))
        if refresh:
            update_candidate(c)
        has[c] = True
        picked.append(int(v))

    for v in seeds:
        add(v, refresh=False)
    for c in np.flatnonzero(has):
        update_candidate(c)
    room, used, ptr = int(COMPONENT_FIRST_FRAC * K), 0, 0
    new = [c for c in np.argsort(-carea, kind='stable') if not has[c]]
    while len(picked) < K:
        while ptr < len(new) and has[new[ptr]]:
            ptr += 1
        if ptr < len(new) and used < room:
            m = members[new[ptr]]
            p = points[new[ptr]]
            add(m[np.argmin(np.linalg.norm(p - p.mean(0), axis=1))])
            used += 1
            continue
        best = np.max(best_distance)
        tied = np.isnan(best_distance) if np.isnan(best) else best_distance == best
        v = int(np.min(best_vertex[tied]))
        if best <= 0:   # every vertex is already an anchor
            break
        add(v)
    return picked


def topology(vertices, faces, landmarks=128, anchors=512):
    """Frame-0 mesh (normalized coordinates) -> dict of torch tensors:
    edges [2, E] (directed), component [V], geo [V, landmarks] (landmark geodesics),
    anchors [anchors] (vertex ids, possibly repeated on tiny meshes),
    geo_anchor [V, anchors] fp16 (exact in-component geodesic vertex -> anchor, inf across components)."""
    n = len(vertices)
    structure = connectivity(n, faces)
    component, members = structure['component'], structure['members']
    graphs = weighted_graphs(vertices, structure)
    count = landmark_budget(vertices, faces, component, landmarks)
    geo = np.zeros((n, int(count.sum())), np.float32)
    landmark_ids, raw = [], {}
    offset = 0
    for c, (picked, rows, half_rows) in enumerate(component_landmarks(graphs, count, n)):
        m = members[c]
        chosen = m[picked]
        landmark_ids.extend(chosen.tolist())
        geo[m, offset:offset + len(picked)] = rows.T
        offset += len(picked)
        raw[c] = (half_rows, {int(a): i for i, a in enumerate(chosen)})
    result = dict(edges=torch.from_numpy(structure['directed'].T.copy()).long(),
                  component=torch.from_numpy(component.astype(np.int64)), geo=torch.from_numpy(geo))
    ids = component_fps(vertices, faces, component, anchors, landmark_ids, members=members)
    if len(ids) < anchors:   # fewer vertices than anchors: repeat slots (the model masks repeated slots)
        ids += [ids[i % len(ids)] for i in range(anchors - len(ids))]
    ids = np.asarray(ids[:anchors], np.int64)
    result['anchors'] = torch.from_numpy(ids)

    # vertex -> anchor geodesics: reuse the landmark rows, solve the remaining sources
    output = np.full((n, anchors), np.inf, np.float16)
    todo_by_component = [np.empty(0, np.int64) for _ in members]
    packed = {}
    for c in np.unique(component[ids]):
        m = members[c]
        slots = np.flatnonzero(component[ids] == c)
        if len(m) * len(slots) < output.size // 2:
            packed[c] = (slots, np.empty((len(m), len(slots)), np.float16))
        sources = np.unique(ids[slots])
        half_rows, lookup = raw[c]
        todo = np.array([a for a in sources if int(a) not in lookup], np.int64)
        todo_by_component[c] = np.searchsorted(m, todo)
        known_slots = np.array([s for s in slots if int(ids[s]) in lookup], np.int64)
        block = max(1, (32 << 20) // (2 * len(m)))
        for start in range(0, len(known_slots), block):
            cols = known_slots[start:start + block]
            row_ids = [lookup[int(ids[s])] for s in cols]
            if c in packed:
                packed[c][1][:, np.searchsorted(slots, cols)] = half_rows[row_ids].T
            else:
                output[np.ix_(m, cols)] = half_rows[row_ids].T
    for c, src, rows in anchor_blocks(graphs, todo_by_component, n):
        m = members[c]
        sources = m[src]
        cols = np.flatnonzero(np.isin(ids, sources))
        values = rows[np.searchsorted(sources, ids[cols])].T
        if c in packed:
            slots, target = packed[c]
            target[:, np.searchsorted(slots, cols)] = values
        else:
            output[np.ix_(m, cols)] = values
    for c, (slots, target) in packed.items():
        output[np.ix_(members[c], slots)] = target
    result['geo_anchor'] = torch.from_numpy(output)
    return result
