import scipy as sp
import networkx as nx
import random
from quippy.ringstat import distance_map
from itertools import combinations


def qm_at_distancemap(at, silica=False, nneightol=1.3):
    qm_list = (at.hybrid_mark == 1).nonzero()[0]

    qm_at = at.select(list=qm_list)
    qm_at.set_cutoff(nneightol * qm_at.cutoff)
    qm_at.calc_connect()

    if silica:
        print 'Converting topology from Si-O-Si to Si-Si'
        si_si, si_o, si_si_cutoff = tetrahedra_to_bonds(qm_at)
        qm_list = sp.logical_and(at.hybrid_mark == 1,
                                 at.z == 14).nonzero()[0]
        dm = distance_map(at=qm_at, n0=qm_at.n, n1=qm_at.n)
        dm = dm[qm_at.z == 14, :][:, qm_at.z == 14] # keep just the Si-Si distances
        qm_at = at.select(list=qm_list)
    else:
        dm = distance_map(qm_at, qm_at.n, qm_at.n)
    return qm_at, dm


def neighbours_to_nxgraph(at, nneightol=1.3):
    """
    Generate a NetworkX graph from a quippy Atoms object. Uses quippy neighbour property. Python indexing: graph node i is atom i+1.
    """
    at.set_cutoff(at.cutoff * nneightol)
    at.calc_connect()
    try:
        # fast way: use quip Fortran function
        adjacency_matrix = sp.array(distance_map(at, at.n, at.n) == 1).astype(int)
    except Exception as err:
        # slower way: go Python but using Fortran structures
        print err, "neighbours_to_nxgraph falling back to slow mode"
        adjacency_matrix = sp.zeros([at.n, at.n])
        for i in range(at.n):
            for neighbour in at.neighbours[i+1]:
                adjacency_matrix[i,neighbour.j-1] = 1
    graph = nx.from_numpy_matrix(sp.array(adjacency_matrix))
    return graph


def distancemap_to_nxgraph(dm):
    adj = sp.array(dm == 1)
    return nx.from_numpy_matrix(adj)


def find_connected_parts(graph, k):

    """
    Given a graph and the desired number of partitions, return list of connected components
    and list of number optimal partitions for each.
    """
    conn_graphs = graph.connected_components()
    ks = sp.array([float(g.number_of_nodes() * k) / graph.number_of_nodes() for g in conn_graphs])
    int_ks = sp.array(ks, dtype='int32')
    while int_ks.sum() < k:
        # add 1 part to the subgraph that would be divided in the highest Y of X.Y parts
        idx = (ks - int_ks).argmax()
        int_ks[idx] +=1
        # exclude that graph from possible future additions
        ks[idx] = int_ks[idx]
    return conn_graphs, int_ks


def merge_2_small_subregions(graph, assign):
    """
    assign is list of integers between 1 and N. Try to merge two small contiguous regions in graph and return the new_assign.
    """
    sizes = [sp.array([m == i for m in assign]).sum() for i in sp.unique(assign)]
    likely_to_merge = sp.argsort(sizes) + 1 # assign starts from 1
    # take first 3 smallest regions, find the best merging for each
    for i, j in combinations(likely_to_merge, 2):
        select1 = sp.where(sp.array(assign) == i)[0]
        select2 = sp.where(sp.array(assign) == j)[0]
        if (graph.subgraph(select1).number_of_edges() + graph.subgraph(select2).number_of_edges() 
            < graph.subgraph(sp.concatenate((select1, select2))).number_of_edges()):
            # found two smallest neighbouring regions to merge.
            # update new_assign list
            print "Merging %d & %d" % (i,j)
            new_assign = [i if old_region == j else old_region for old_region in assign]
            break   
    # now shift new_assign from [1,..., N] to [1,..., N-1]
    # assign[idx] > j must be reduced by 1
    new_assign = [region - 1 if region > j else region for region in new_assign]
    return new_assign


def spectral_bipartition_cluster(at, nneightol=1.3, median_split=True):
    # WARNING: OLD, avoid using it
    """
    Given an atomic neighbourhood, return a list of assingments to either region 0 or 1.
    The two regions are guaranteed to be connected and convex, and are usually fairly balanced.
    
    Fiedler, Miroslav. "A property of eigenvectors of nonnegative symmetric matrices and its application to graph theory." 
    Czechoslovak Mathematical Journal 25, no. 4 (1975): 619-633.
    """
    eigvals, eigvecs = sp.linalg.eigh(nx.laplacian_matrix(neighbours_to_nxgraph(at)).todense(), eigvals=(0,1))
    # Fiedler's vector: eigenvector associated with second lowest eigenvalue   
    x2 = sp.array(eigvecs[:,1])  
    return sp.array([int(i) for i in (x2 < 0)])


def spectral_bipartition_graph(graph, verbose=False):
    """
    Given a graph, return list of nodes of the two subgraphs obtained by spectral bipartitioning.
    Find best connected partition
    """
    accept_threshold = 0.2
    sparseLA_minN = 200


    def loop_for_connected(x2, graph, accept_threshold):
        for split in sp.linspace(sp.median(x2), 0.0, 10):
            mask = sp.array([int(i) for i in x2 > split])
            nodelist1 = [n for i,n in enumerate(graph.nodes()) if mask[i]]
            nodelist2 = [n for i,n in enumerate(graph.nodes()) if not mask[i]]
            split_ok = nx.is_connected(graph.subgraph(nodelist1)) and nx.is_connected(graph.subgraph(nodelist2))
            # quit if either (A) a valid split is found (B) we're getting too big size differences
            if split_ok or (sp.absolute(len(nodelist1)-len(nodelist2))/float(len(mask)) > accept_threshold): break
        return nodelist1, nodelist2, split_ok

    
    split_ok = False
    if graph.number_of_nodes() > sparseLA_minN:
        # Only use sparse LA for big enough matrices
        x2 = nx.fiedler_vector(graph)
        nodelist1, nodelist2, split_ok = loop_for_connected(x2, graph, accept_threshold=accept_threshold)
        if not split_ok and verbose: print "Switching to dense LA, sparse eigsh method failed to converge"

    if not split_ok:
        eig, x2 = sp.linalg.eigh(nx.laplacian_matrix(graph).todense(), eigvals=(1,1))
        x2 = x2.flatten()
        # round off values that are close to machine zero. HAPPENS IN HIGHLY SYMMETRICAL GRAPHS
        x2[sp.absolute(x2) < 1e-6] = 0.0
        nodelist1, nodelist2, split_ok = loop_for_connected(x2, graph, accept_threshold = 1.0)
    if verbose: print "spectral bisection success: %s. Split: %d - %d" % (split_ok, len(nodelist1), len(nodelist2))
    if not split_ok: print "ERROR: Spectral bipartition failed to produce connected splits. Fiedler's vector:", x2
    return nodelist1, nodelist2


def spectral_graph_partitioning_seq(graph, nregions, merge=True, verbose=False) :
    """
    Given a graph, usually representing an atomic environment, 
    iterate spectral bipartitioning and return a list of assignments to regions in [1, ..., nregions].
    If nregions != 2**N, merge=True will merge small contiguous regions to go from 2**N > nregions --> nregions;
    if merge=False, the method will return 2**N regions.
    """
    if int(sp.log2(nregions)) != sp.log2(nregions):
        K = 2**(int(sp.log2(nregions)) + 1)
        if not merge:
            print "WARNING: spectral_graph_partitioning_seq will divide the system into %d instead of %d" % (K, nregions)
    else:
        K = nregions
    subgraphs = [graph] # first: whole graph
    for i in range(int(sp.log2(K))):
        subgraphs_temp = []
        for g in subgraphs:
            # split each subgraph in two
            mask1, mask2 = spectral_bipartition_graph(g, verbose=verbose)
            subgraphs_temp += [g.subgraph(mask1), g.subgraph(mask2)]
        subgraphs = subgraphs_temp # update subgraphs
    assignments = []
    if verbose: print "spectral partitioning divided system in parts of sizes:", [x.number_of_nodes() for x in subgraphs]
    dummy = [[assignments.append((node, i)) for node in region.nodes()] for i, region in enumerate(subgraphs)]
    # returns an integer assign mask
    assign = [assign + 1 for i, assign in sorted(assignments)]

    if K != nregions and merge:
        # merge smallest regions to obtain desired nregions instead of 2**N
        for nmerge in range((K - nregions)):
            assign_temp = merge_2_small_subregions(graph, assign)
            assign = assign_temp
    return assign


def find_medoids_partitioned_graph(graph, assign):
    """
    Given a graph and a list of assignments to N > 0 partitions, return list of medoids of such graph (centre elements of each subgraph).
    WARNING: node counting starts from 0, so qm_at[1] will be graph.nodes()[0]
    The centres of a graph are the set of nodes such that their eccentricity is equal to
    the graph radius. http://en.wikipedia.org/wiki/Graph_center
    """
    assign = sp.array(assign)
    medoids = []
    for region in sp.unique(assign):
        subgraph = graph.subgraph(sp.where(assign == region)[0])
        radius = nx.radius(subgraph)
        centres = [node for node in subgraph.nodes() if nx.eccentricity(subgraph, v=node) == radius]
        medoids.append(random.choice(centres))
    return medoids


def atpos_to_nxpos(at, subtract_centre_of_mass=False, project_2d=False):
    """
    Generate NetworkX positions from quippy Atoms object. Python indexing: pos[i] is at.pos[i+1]
    """
    pos = {}
    positions = sp.array(at.pos).T
    if subtract_centre_of_mass or project_2d:
        positions = positions - positions.mean(axis=0)
    if project_2d:
        u, s, v = sp.linalg.svd(positions)
        positions = sp.dot(positions, v[:2].T) # 2D projection on principal components of coordinates
    for atom in range(at.n):
        pos[atom] = positions[atom]
    return pos


def unconnected_atoms(at, nneightol=1.3):
    graph = neighbours_to_nxgraph(at, nneightol=nneightol)
    neighbours_to_nxgraph(at, nneightol=1.3)
    d = nx.degree(graph)
    return [k+1 for (k,v) in d.iteritems() if v == 0 ]


def heal_hybrid_atoms(at, silica=False, nneightol=1.3):
    qm_at, dm = qm_at_distancemap(at, silica, nneightol)
    de_mark = qm_at.orig_index[unconnected_atoms(qm_at)]
    at.hybrid_mark[de_mark] = 0
    if at.has_property('hybrid'):
        at.hybrid[:] = at.hybrid_mark
    return 

