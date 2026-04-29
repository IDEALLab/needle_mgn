import numpy as np
import meshio
from odbAccess import openOdb
from multiprocessing import Pool
from scipy.spatial import cKDTree
import glob
import os
import time

# -----------------------------
# CONFIG
# -----------------------------
ODB_DIR = "H:/6-inps"     # directory to glob for .odb files
OUTPUT_DIR = "H:/output"  # directory for .vtu output files
STEP_NAME = None  # None = first step
INSTANCES = ["NDL-1", "EUL-1"]

# Fields to extract
NODE_FIELDS = ["COORD", "U", "V", "A", "CPRESS"]
ELEMENT_FIELDS = ["S", "P", "EVF_VOID"]
# Source fields (in the ODB) to merge into a single output field
ELEMENT_FIELD_ALIASES = {
    "S": ["S", "SVAVG"],
}

EDGE_RADIUS = 1.2  # max distance (model units) to connect needle nodes to tissue nodes
INP_DIR = "H:/6-inps"  # directory containing .inp files named to match the ODB

# -----------------------------
# HELPERS
# -----------------------------
def build_index_map(nodes):
    labels = [node.label for node in nodes]
    label_to_idx = {label: i for i, label in enumerate(labels)}
    coords = np.array([node.coordinates for node in nodes])
    return label_to_idx, coords

def extract_connectivity(elements, label_to_idx):
    cells = []
    for el in elements:
        conn = [label_to_idx[n] for n in el.connectivity]
        cells.append(conn)
    return np.array(cells)

def extract_field(frame, field_name, label_to_idx, is_node=True):
    field = frame.fieldOutputs[field_name]
    n = len(label_to_idx)
    
    # Determine shape dynamically
    sample = next(iter(field.values))
    data_shape = np.shape(sample.data)
    
    if len(data_shape) == 0:
        out = np.zeros((n,))
    else:
        out = np.zeros((n, data_shape[0]))

    for val in field.values:
        label = val.nodeLabel if is_node else val.elementLabel
        if label in label_to_idx:
            out[label_to_idx[label]] = val.data

    return out

def build_global_maps(odb, instances):
    global_node_offset = {}
    global_elem_offset = {}

    node_maps = {}
    elem_maps = {}

    total_nodes = 0
    total_elems = 0

    for inst_name in instances:
        inst = odb.rootAssembly.instances[inst_name]

        # --- Node map ---
        node_labels = np.array([n.label for n in inst.nodes], dtype=np.int32)
        node_map = {label: i for i, label in enumerate(node_labels)}

        node_maps[inst_name] = node_map
        global_node_offset[inst_name] = total_nodes
        total_nodes += len(node_labels)

        # --- Element map ---
        elem_labels = np.array([e.label for e in inst.elements], dtype=np.int32)
        elem_map = {label: i for i, label in enumerate(elem_labels)}

        elem_maps[inst_name] = elem_map
        global_elem_offset[inst_name] = total_elems
        total_elems += len(elem_labels)

    return node_maps, elem_maps, global_node_offset, global_elem_offset, total_nodes, total_elems

def build_field_block_cache(frame, field_name, label_maps, global_offsets, total_size, is_node):
    """One-time pass on the first frame: record valid block indices and pre-compute global_idx arrays."""
    field = frame.fieldOutputs[field_name]
    block_indices = []
    global_idxs = []
    output_shape = None

    for i, bdb in enumerate(field.bulkDataBlocks):
        if bdb.instance is None:
            continue
        inst_name = bdb.instance.name
        if inst_name not in label_maps:
            continue

        labels = bdb.nodeLabels if is_node else bdb.elementLabels
        data = bdb.data
        if labels is None or data is None:
            continue

        label_map = label_maps[inst_name]
        offset = global_offsets[inst_name]
        local_idx = np.array([label_map[l] for l in labels], dtype=np.int32)
        global_idxs.append(local_idx + offset)
        block_indices.append(i)

        if output_shape is None:
            output_shape = (total_size,) if data.ndim == 1 else (total_size, data.shape[1])

    return block_indices, global_idxs, output_shape

def extract_field_cached(frame, field_name, block_indices, global_idxs, output_shape):
    """Extract a field using pre-built block cache — no label mapping or instance checks per frame."""
    field = frame.fieldOutputs[field_name]
    out = np.zeros(output_shape)
    blocks = field.bulkDataBlocks
    for idx, global_idx in zip(block_indices, global_idxs):
        out[global_idx] = blocks[idx].data
    return out

# -----------------------------
# INP PARSER
# -----------------------------
def parse_inp_materials(inp_path):
    """Parse material properties and orientations from an Abaqus INP file.

    Returns:
        materials   – dict of {name: {density, E, nu, c10, k1, k2, kappa, d}}
        orientations – dict of {name: {axis1: ndarray(3), axis2: ndarray(3)}}
    """
    materials = {}
    orientations = {}
    current_mat = None
    expect = None

    with open(inp_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('**'):
                continue

            if line.startswith('*'):
                upper = line.upper()
                if '*MATERIAL' in upper:
                    current_mat = line.split('=')[-1].strip().split(',')[0].strip()
                    materials[current_mat] = {}
                    expect = None
                elif '*DENSITY' in upper and current_mat is not None:
                    expect = 'density'
                elif 'HYPERELASTIC' in upper and current_mat is not None:
                    expect = 'hyperelastic'
                elif '*ELASTIC' in upper and current_mat is not None:
                    expect = 'elastic'
                elif '*ORIENTATION' in upper:
                    ori_name = line.split('=')[-1].strip().split(',')[0].strip()
                    expect = ('orientation', ori_name)
                    current_mat = None
                else:
                    expect = None
            else:
                try:
                    vals = [float(v) for v in line.replace(',', ' ').split() if v.strip()]
                except ValueError:
                    continue
                if not vals:
                    continue
                if expect == 'density':
                    materials[current_mat]['density'] = vals[0]
                    expect = None
                elif expect == 'elastic' and len(vals) >= 2:
                    materials[current_mat]['E'] = vals[0]
                    materials[current_mat]['nu'] = vals[1]
                    expect = None
                elif expect == 'hyperelastic' and len(vals) >= 5:
                    materials[current_mat]['c10'] = vals[0]
                    materials[current_mat]['k1']  = vals[1]
                    materials[current_mat]['k2']  = vals[2]
                    materials[current_mat]['kappa'] = vals[3]
                    materials[current_mat]['d']   = vals[4]
                    expect = None
                elif isinstance(expect, tuple) and expect[0] == 'orientation' and len(vals) >= 6:
                    orientations[expect[1]] = {
                        'axis1': np.array(vals[0:3]),
                        'axis2': np.array(vals[3:6]),
                    }
                    expect = None

    return materials, orientations

# -----------------------------
# WORKER (module-level so multiprocessing can pickle it on Windows)
# -----------------------------
def extract_and_write_worker(args):
    frame_ids, node_cache, elem_cache, points, all_cells, global_elem_offset, instance_elem_counts, \
        eul_tree, ndl_start, ndl_end, eul_start, static_cell_type, static_point_data, \
        odb_path = args

    import traceback
    try:
      odb = openOdb(odb_path, readOnly=True)
    except Exception:
        return {"node_fields": 0.0, "element_fields": 0.0, "mesh_write": 0.0,
                "error": traceback.format_exc()}

    output_name = odb_path.split('/')[-1].split('.')[0].split('\\')[-1]
    try:
      step = odb.steps[STEP_NAME] if STEP_NAME else list(odb.steps.values())[0]
      frames = step.frames

      t_node = t_elem = t_write = 0.0

      for frame_id in frame_ids:
        frame = frames[frame_id]

        point_data = dict(static_point_data)
        t0 = time.perf_counter()
        for field_name, (block_indices, global_idxs, output_shape) in node_cache.items():
            try:
                ndata = extract_field_cached(frame, field_name, block_indices, global_idxs, output_shape)
                if ndata.ndim == 1:
                    ndata = ndata[:, None]
                point_data[field_name] = ndata
            except Exception:
                continue
        t_node += time.perf_counter() - t0

        cell_data = {}
        t0 = time.perf_counter()
        for field_name, (sources, output_shape) in elem_cache.items():
            try:
                edata = np.zeros(output_shape)
                for source_name, block_indices, global_idxs in sources:
                    blocks = frame.fieldOutputs[source_name].bulkDataBlocks
                    for idx, gidx in zip(block_indices, global_idxs):
                        edata[gidx] = blocks[idx].data
                block_values = []
                for inst_name in INSTANCES:
                    start = global_elem_offset[inst_name]
                    end = start + instance_elem_counts[inst_name]
                    block_values.append(edata[start:end])
                cell_data[field_name] = block_values
            except Exception:
                continue
        t_elem += time.perf_counter() - t0

        t0 = time.perf_counter()
        frame_points = point_data.pop("COORD", points)

        # Recompute edges from deformed NDL positions each frame
        neighbor_lists = eul_tree.query_ball_point(frame_points[ndl_start:ndl_end], r=EDGE_RADIUS)
        edge_list = [
            [ndl_start + local_ndl, eul_start + local_eul]
            for local_ndl, neighbors in enumerate(neighbor_lists)
            for local_eul in neighbors
        ]
        n_edges = len(edge_list)
        frame_cells = all_cells + ([("line", np.array(edge_list, dtype=np.int32))] if n_edges else [])

        cell_type_data = list(static_cell_type)
        if n_edges:
            cell_type_data.append(np.full(n_edges, 2, dtype=np.int32))
        cell_data["element_type"] = cell_type_data

        # Pad element fields with zeros for the edge block
        if n_edges:
            for field_name in list(cell_data.keys()):
                if field_name == "element_type":
                    continue
                bv = cell_data[field_name]
                ref = bv[0]
                bv.append(np.zeros((n_edges,) + ref.shape[1:]))

        mesh = meshio.Mesh(
            points=frame_points,
            cells=frame_cells,
            point_data=point_data,
            cell_data=cell_data
        )
        filename = os.path.join(OUTPUT_DIR, f"{output_name}_{frame_id:04d}.vtu")
        mesh.write(filename)
        t_write += time.perf_counter() - t0

      odb.close()
      return {"node_fields": t_node, "element_fields": t_elem, "mesh_write": t_write, "error": None}
    except Exception:
        odb.close()
        return {"node_fields": 0.0, "element_fields": 0.0, "mesh_write": 0.0,
                "error": traceback.format_exc()}

# -----------------------------
# MAIN EXPORT
# -----------------------------
def export_vtu(odb_path):
    timings = {}

    t0 = time.perf_counter()
    odb = openOdb(odb_path, readOnly=True)
    timings["open_odb"] = time.perf_counter() - t0

    odb_name = odb.name

    step = odb.steps[STEP_NAME] if STEP_NAME else list(odb.steps.values())[0]
    frames = step.frames

    # Build global maps for fast extraction
    t0 = time.perf_counter()
    node_maps, elem_maps, global_node_offset, global_elem_offset, total_nodes, total_elems = build_global_maps(odb, INSTANCES)
    timings["build_global_maps"] = time.perf_counter() - t0

    # Build points and connectivity once (static across frames)
    t0 = time.perf_counter()
    all_points = []
    all_cells = []
    offset = 0

    for inst_name in INSTANCES:
        inst = odb.rootAssembly.instances[inst_name]
        _, coords = build_index_map(inst.nodes)
        n_local = coords.shape[0]

        all_points.append(coords)

        conn = extract_connectivity(inst.elements, {n.label: i for i, n in enumerate(inst.nodes)})
        conn = conn + offset
        all_cells.append(("hexahedron", conn))

        offset += n_local

    points = np.vstack(all_points)

    # Build needle→tissue edges: connect each NDL-1 node to EUL-1 nodes within EDGE_RADIUS
    ndl_start = global_node_offset["NDL-1"]
    ndl_end   = ndl_start + len(node_maps["NDL-1"])
    eul_start = global_node_offset["EUL-1"]
    eul_end   = eul_start + len(node_maps["EUL-1"])

    # EUL mesh is fixed; build KD-tree once from reference positions.
    # NDL moves, so edges are recomputed per frame in the worker using deformed COORD.
    eul_tree = cKDTree(points[eul_start:eul_end])

    timings["build_geometry"] = time.perf_counter() - t0

    # Per-instance element counts for mapping block data in the same order as all_cells
    instance_elem_counts = {
        inst_name: len(odb.rootAssembly.instances[inst_name].elements)
        for inst_name in INSTANCES
    }

    # Per-node material features from INP file
    materials, orientations = parse_inp_materials(os.path.join(INP_DIR, odb_name.split('.')[0] + ".inp"))

    mat_density = np.zeros((total_nodes, 1))
    mat_E       = np.zeros((total_nodes, 1))
    mat_nu      = np.zeros((total_nodes, 1))
    mat_c10     = np.zeros((total_nodes, 1))
    mat_k1      = np.zeros((total_nodes, 1))
    mat_k2      = np.zeros((total_nodes, 1))
    mat_kappa   = np.zeros((total_nodes, 1))
    mat_fiber   = np.zeros((total_nodes, 3))

    ndl_sl = slice(ndl_start, ndl_end)
    eul_sl = slice(eul_start, eul_end)

    m = materials.get("NEDLE", {})
    mat_density[ndl_sl, 0] = m.get('density', 0.0)
    mat_E[ndl_sl, 0]       = m.get('E', 0.0)
    mat_nu[ndl_sl, 0]      = m.get('nu', 0.0)

    m = materials.get("EUL_ANISO", {})
    mat_density[eul_sl, 0] = m.get('density', 0.0)
    mat_c10[eul_sl, 0]     = m.get('c10', 0.0)
    mat_k1[eul_sl, 0]      = m.get('k1', 0.0)
    mat_k2[eul_sl, 0]      = m.get('k2', 0.0)
    mat_kappa[eul_sl, 0]   = m.get('kappa', 0.0)

    if orientations:
        mat_fiber[eul_sl] = next(iter(orientations.values()))['axis1']

    static_point_data = {
        "mat_density": mat_density,
        "mat_E":       mat_E,
        "mat_nu":      mat_nu,
        "mat_c10":     mat_c10,
        "mat_k1":      mat_k1,
        "mat_k2":      mat_k2,
        "mat_kappa":   mat_kappa,
        "mat_fiber":   mat_fiber,
    }

    # Hex-block cell type labels (edge block appended per frame in worker)
    static_cell_type = [
        np.zeros(instance_elem_counts["NDL-1"], dtype=np.int32),
        np.ones(instance_elem_counts["EUL-1"], dtype=np.int32),
    ]

    # Build block caches from the first frame (block order/labels are fixed across frames)
    t0 = time.perf_counter()
    node_cache = {}
    for field_name in NODE_FIELDS:
        try:
            node_cache[field_name] = build_field_block_cache(
                frames[0], field_name, node_maps, global_node_offset, total_nodes, is_node=True)
        except Exception:
            pass

    elem_cache = {}
    for output_name in ELEMENT_FIELDS:
        sources = []
        output_shape = None
        for source_name in ELEMENT_FIELD_ALIASES.get(output_name, [output_name]):
            try:
                bidx, gidx, oshape = build_field_block_cache(
                    frames[0], source_name, elem_maps, global_elem_offset, total_elems, is_node=False)
                sources.append((source_name, bidx, gidx))
                if output_shape is None:
                    output_shape = oshape
            except Exception:
                pass
        if sources:
            elem_cache[output_name] = (sources, output_shape)
    timings["build_block_cache"] = time.perf_counter() - t0

    n_frames = len(frames)
    odb.close()  # each worker opens its own connection

    N_EXTRACT_WORKERS = 16
    frame_ids = list(range(n_frames))
    chunks = [frame_ids[i::N_EXTRACT_WORKERS] for i in range(N_EXTRACT_WORKERS) if frame_ids[i::N_EXTRACT_WORKERS]]
    args_list = [
        (chunk, node_cache, elem_cache, points, all_cells, global_elem_offset, instance_elem_counts,
         eul_tree, ndl_start, ndl_end, eul_start, static_cell_type, static_point_data,
         odb_path)
        for chunk in chunks
    ]

    t0 = time.perf_counter()
    with Pool(processes=N_EXTRACT_WORKERS) as pool:
        worker_results = pool.map(extract_and_write_worker, args_list)
    timings["wall_extract_and_write"] = time.perf_counter() - t0

    # Sum per-worker CPU timings; collect any errors
    frame_timings = {"node_fields": 0.0, "element_fields": 0.0, "mesh_write": 0.0}
    errors = [r["error"] for r in worker_results if r.get("error")]
    for r in worker_results:
        if not r["error"]:
            for k in frame_timings:
                frame_timings[k] += r[k]

    if errors:
        print(f"\n  {len(errors)} worker(s) failed:")
        for err in errors:
            print(err)
        return  # don't write .done marker

    timings.update(frame_timings)
    total = sum(timings.values())
    print("\n--- Timing Report ---")
    for name, elapsed in timings.items():
        print(f"  {name:<20s}: {elapsed:7.2f}s  ({100*elapsed/total:.1f}%)")
    print(f"  {'TOTAL':<20s}: {total:7.2f}s")
    print(f"  (node_fields/element_fields/mesh_write are summed CPU time across {N_EXTRACT_WORKERS} workers)")

    open(os.path.join(OUTPUT_DIR, f"{odb_name}.done"), 'w').close()

# -----------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    odb_files = sorted(glob.glob(os.path.join(ODB_DIR, "*.odb")))
    print(f"Found {len(odb_files)} ODB files in {ODB_DIR}")
    for i, odb_path in enumerate(odb_files, 1):
        run_name = os.path.splitext(os.path.basename(odb_path))[0]
        if os.path.exists(os.path.join(OUTPUT_DIR, f"{run_name}.done")):
            print(f"[{i}/{len(odb_files)}] Skipping {run_name} (already complete)")
            continue
        print(f"\n[{i}/{len(odb_files)}] Processing {run_name}")
        export_vtu(odb_path)