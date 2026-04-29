import sys
from odbAccess import *
from abaqusConstants import *
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def save_node_element_data(odb):
    NDL_nodes = np.zeros((len(odb.rootAssembly.instances['NDL-1'].nodes)+1, 3))
    NDL_elements = np.zeros((len(odb.rootAssembly.instances['NDL-1'].elements)+1, 8))
    EUL_nodes = np.zeros((len(odb.rootAssembly.instances['EUL-1'].nodes)+1, 3))
    EUL_elements = np.zeros((len(odb.rootAssembly.instances['EUL-1'].elements)+1, 8))
    for node in tqdm(odb.rootAssembly.instances['NDL-1'].nodes):
        NDL_nodes[node.label, 0] = node.coordinates[0]
        NDL_nodes[node.label, 1] = node.coordinates[1]
        NDL_nodes[node.label, 2] = node.coordinates[2]
    for element in tqdm(odb.rootAssembly.instances['NDL-1'].elements):
        NDL_elements[element.label, 0] = element.connectivity[0]
        NDL_elements[element.label, 1] = element.connectivity[1]
        NDL_elements[element.label, 2] = element.connectivity[2]
        NDL_elements[element.label, 3] = element.connectivity[3]
        NDL_elements[element.label, 4] = element.connectivity[4]
        NDL_elements[element.label, 5] = element.connectivity[5]
        NDL_elements[element.label, 6] = element.connectivity[6]
        NDL_elements[element.label, 7] = element.connectivity[7]
    for node in tqdm(odb.rootAssembly.instances['EUL-1'].nodes):
        EUL_nodes[node.label, 0] = node.coordinates[0]
        EUL_nodes[node.label, 1] = node.coordinates[1]
        EUL_nodes[node.label, 2] = node.coordinates[2]
    for element in tqdm(odb.rootAssembly.instances['EUL-1'].elements):
        EUL_elements[element.label, 0] = element.connectivity[0]
        EUL_elements[element.label, 1] = element.connectivity[1]
        EUL_elements[element.label, 2] = element.connectivity[2]
        EUL_elements[element.label, 3] = element.connectivity[3]
        EUL_elements[element.label, 4] = element.connectivity[4]
        EUL_elements[element.label, 5] = element.connectivity[5]
        EUL_elements[element.label, 6] = element.connectivity[6]
        EUL_elements[element.label, 7] = element.connectivity[7]

    np.save('NDL_nodes.npy', NDL_nodes)
    np.save('NDL_elements.npy', NDL_elements)
    np.save('EUL_nodes.npy', EUL_nodes)
    np.save('EUL_elements.npy', EUL_elements)

# Field outputs in the odb:
# ['CF', 'CM', 'COORD', 'EVF_ASSEMBLY_EUL-1_EUL_ANISO', 'EVF_VOID', 
#  'LE', 'LOCALDIR1', 'LOCALDIR2', 'NFORC1', 'NFORC2', 'NFORC3', 'RF', 
#  'RM', 'RT', 'S', 'STATUS', 'SVAVG', 'U', 'UR', 'UT']

def save_needle_coords(odb, job_num):
    currentstepname = odb.steps.keys()[0]

    coord_data = np.zeros((len(odb.steps[currentstepname].frames), 5682, 3))
    for i in range(len(odb.steps[currentstepname].frames)):
        frame  = odb.steps[currentstepname].frames[i]
        u_field = frame.fieldOutputs['COORD']
        for bdb in u_field.bulkDataBlocks:
            # print("  nodeLabels:", len(bdb.nodeLabels))  # may be None
            # print(bdb.elementLabels)
            # print(bdb.instance.name)  # may be None
            # print("  Shape:", bdb.data.shape)  # may be None
            coord_data[i, bdb.elementLabels, 0] = bdb.data[:, 0]
            coord_data[i, bdb.elementLabels, 1] = bdb.data[:, 1]
            coord_data[i, bdb.elementLabels, 2] = bdb.data[:, 2]
    np.save('needle_coords_9-ANISO-{}.npy'.format(job_num), coord_data)
    
def pool_wrap(job_num):
    odb_dir=r'H:/aniso-9'
    odb = openOdb('{}/9-ANISO-RUN-{}.odb'.format(odb_dir, job_num), readOnly=True)
    save_needle_coords(odb, job_num)
    odb.close()

if __name__ == '__main__':
    job_num = 1
    odb_dir=r'H:/aniso-9'
    name      = '9-ANISO-RUN-{}'.format(job_num)
    odb_path  = '{}/{}.odb'.format(odb_dir, name)
    odb       = openOdb(odb_path, readOnly=True)
    node_ids  = range(1, 33)
    node_ids  = np.asarray(node_ids, dtype=np.int32)

    step      = odb.steps[odb.steps.keys()[0]]
    frames    = step.frames
    n_frames  = len(frames)
    node_ids  = range(1, 33)
    node_ids  = np.asarray(node_ids, dtype=np.int32)
    n_nodes   = len(node_ids)
    # save_node_element_data(odb)
    # save_needle_coords(odb, job_num)
    odb.close()
    
    # global NDL_elements_shape 
    # NDL_elements = np.load('NDL_elements.npy')
    # NDL_elements_shape = NDL_elements.shape
    # print(NDL_elements_shape)

    odb_dir=r'H:/aniso-9'
    # 103 bad?
    job_nums = range(1, 10)

    # Try to check whether stdout is valid
    try:
        is_tty = sys.stdout.isatty()
    except Exception:
        is_tty = False

    # Disable tqdm if not running in terminal
    use_tqdm = is_tty

    # 24 processes seems to saturate the NAS bandwidth
    pool = Pool(processes=24)
    results = list(pool.imap_unordered(pool_wrap, job_nums))  # disables if not TTY
    pool.close()
    pool.join()
