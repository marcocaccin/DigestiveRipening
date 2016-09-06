# Portions of this code were written by
# Marco Caccin, James R. Kermode

import numpy as np

from quippy.ringstat import distance_map
from quippy.clusters import HYBRID_ACTIVE_MARK
from quippy.farray import farray, fzeros, FortranArray, fenumerate
from quippy import AtomsWriter
import random
import atomsgraph as atg
from digestive_ripening import digestive_ripening
"""
de-quippyfication status:
- distance_map can be replaced with an equivalent matscipy version
- HYBRID_ACTIVE_MARK has to be changed somehow
- farray is only used for the assign array for compatibility with higher level bgqscript
- FortranArray appears in k-medoids algo
- fzeros appears in initialisation of distance matrix
- fenumerate appears only in enumerating medoids, which are not useful quantities anyway
- AtomsWriter is just for writing the movie: nice to look at but has no real use.
"""
try:
    import metis
    is_metis = True
except:
    is_metis = False
try:
    from sklearn import cluster
    is_sklearn = True
except:
    is_sklearn = False

    
def tetrahedra_to_bonds(q):
    si_si = []
    si_o = []

    max_si_si_dist = 0.0
    
    for i in q.indices:
        if q.z[i] != 14:
            continue

        for nj in q.neighbours[i]:
            j = nj.j

            if q.z[j] != 8:
                continue

            # i and j are an Si-O neighbour pair
            si_o.append((i, j, tuple(nj.shift), nj.distance))

            for nk in q.neighbours[j]:
                k = nk.j
                
                if not i < k:
                    continue

                shift = nj.shift + nk.shift

                if q.z[k] != 14 or (k == i and all(shift == 0)):
                    continue

                # i and k are an Si-Si pair with a common O neighbour
                r_ij = float(np.sqrt(((nj.diff + nk.diff)**2).sum()))
                if r_ij > max_si_si_dist:
                    max_si_si_dist = r_ij
                    
                si_si.append((i, k, tuple(shift), r_ij))

    print '%d Si-Si, %d Si-O' % (len(si_si), len(si_o))

    q.connect.wipe()
    for (i, j, shift, r_ij) in si_si:
        d = 1.0 # all Si-Si bonds interpreted as being of length 1 ang
        q.connect.add_bond(q.pos, q.lattice, i, j, shift, d)

    return si_si, si_o, max_si_si_dist


def atspecies_to_speciesstr(at_species):
    """
    Feed in at.species, it's a crap FortranArray of characters --> get out a nice list of strings.
    """
    return [(''.join(string)).strip() for string in at_species]


def bulk_coordination_of_species(species):
    if species == 'Si':
        return 4
    elif species == 'O':
        return 2
    elif species == 'H':
        return 1


def initialise_qm_region(at, nneightol, silica):
    """
    From Atoms - at - return an Atoms object containing the QM atoms only with related orig_index - qm_list - 
    and the distance map of qm_at
    """
    qm_list = (at.hybrid_mark == HYBRID_ACTIVE_MARK).nonzero()[0]
    qm_at = at.select(list=qm_list)
    qm_at.set_cutoff_factor(nneightol)
    qm_at.calc_connect()

    if silica:
        print 'Converting topology from Si-O-Si to Si-Si'
        si_si, si_o, si_si_cutoff = tetrahedra_to_bonds(qm_at)
        qm_list = np.logical_and(at.hybrid_mark == HYBRID_ACTIVE_MARK,
                                 at.z == 14).nonzero()[0]
        dm = distance_map(at=qm_at, n0=qm_at.n, n1=qm_at.n)
        dm = dm[qm_at.z == 14, :][:, qm_at.z == 14] # keep just the Si-Si distances
        qm_at = at.select(list=qm_list)
    else:
        dm = distance_map(qm_at, qm_at.n, qm_at.n)
                                 
    print 'len(qm_list) = ', len(qm_list)
    return qm_at, qm_list, dm


def initialise_medoids(at, qm_at, dm, K, verbose):
    """
    Provided an Atoms object - at - and its QM subset - qm_at -, and a distance map of qm_at, 
    select K atoms in qm_at as initial medoids of such graph.
    """
    medoids = []
    qm_list = qm_at.orig_index
    
    if hasattr(at, 'qm_medoids') and (at.qm_medoids != 0).sum() > 0:
        prev_medoid_indices = at.qm_medoids.nonzero()[0]
        prev_medoid_order = at.qm_medoids[prev_medoid_indices]

        for i, m in sorted(zip(prev_medoid_order, prev_medoid_indices)):
            if m in qm_list and len(medoids) < K:
                medoids.append((qm_at.orig_index == m).nonzero()[0][0])
            else:
                print 'prev QM medoid %d (atom %d) no longer in QM region' % (i, m)
    else:
        # initialise first medoid to be QM atom furthest away from everything else
        medoids = [ dm.sum(axis=0).argmax() ]

    # add remaining K-1 medoids as atoms furthest from other medoids
    while len(medoids) < K:
        dist = fzeros(qm_at.n, dtype=np.int32)
        for m in medoids:
            dist += dm[m, :]
        dist[medoids] = 0 # ignore medoids already selected
        medoids.append(dist.argmax())
    
    if verbose:
        print 'initial medoids', medoids, [qm_at.orig_index[m] for m in medoids]
    return medoids


def k_medoids(dm, medoids, max_step):
    """
    Standard k-medoids. Provide a distance map - dm - and an initial set of set of nodes as medoids. Number of partitions = len(medoids)
    Result is a list of assignments of atom i to medoid j

    Returns
    ______
    medoids : list 
    assign : rank-1 FortranArray of length len(dm), with entries in frange(len(medoids))
    """
    step = 0
    while step < max_step:
        # assign each atom to the closest medoid
        assign = dm[:, medoids].argmin(axis=2)

        # re-determine the medoid of each cluster
        new_medoids = []
        for i, m in fenumerate(medoids):
            sub_dm = dm.view(np.ndarray)[np.ix_(assign == i, assign == i)].sum(axis=0).view(FortranArray)
            if sub_dm.size == 0:
                # delete empty medoid
                continue
            new_i = sub_dm.argmin()
            new_m = ((assign == i).nonzero()[0]).view(FortranArray)[new_i]
            new_medoids.append(new_m)
            
        # check for convergence
        if medoids == new_medoids:
            break
        medoids = new_medoids
        step += 1
        
    else:
        raise RuntimeError('exceeded maximum number of steps')
    return medoids, assign


def partition_qm_list(at, K, init_method=None, ripening_algorithm='none', nneightol=1.3, max_step=100, quick_exit=False, make_movie=False, silica=False, verbose=False, assign=None, ripening=True, spectral_partitioning=True, use_metis=False):
    """
    Given at Atoms object with a single QM zone, split it into K convex subzones of approximately same size.
    If nothing is specified, generate partition by k-means without digestive ripening (extremely fast, especially if sklearn is installed, and decent quality).
    """

    qm_at, qm_list, dm = initialise_qm_region(at, nneightol, silica)

    # if there is not a specified method, use the one specified in the extra (old) keywords
    if init_method is None:
        if use_metis:
            init_method = 'metis'
        elif spectral_partitioning:
            init_method = 'spectral'
        else:
            init_method = 'k-medoids'

    # if there is an initial assign, do not try to pre-partition the graph but rather use that guess
    if assign is not None:
        assign = np.asarray(assign)
        if len(assign) == qm_at.n and len(np.unique(assign)) == K:
            init_method = 'skip'

    if not ripening:
        # if ripening flag is not specified, use newer keyword ripening_algorithm
        ripening_algorithm = 'none'
    elif ripening_algorithm == 'none':
        ripening_algorithm = 'fullrand'
        
    if make_movie:
        moviewriter = AtomsWriter('medoid_movie.%03d.nc' % K)
        at_movie = qm_at.copy()
        assign_movie = 'assign_movie.%03d.csv' % K
    else:
        movie = None
        at_movie = None
        assign_movie = 'none'


    if verbose:
        print("### Init: %s, ripening: %s | metis=%i, sklearn=%i ###" %(init_method, ripening_algorithm, is_metis, is_sklearn))
    qm_graph = atg.distancemap_to_nxgraph(dm)
    
    # generate K partitions
    if init_method == 'spectral':
        try:
            if is_sklearn:
                clustering_algorithm = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='nearest_neighbors')# np.asarray(dm == 1))
                clustering_algorithm.fit(qm_at.positions)
                assign = clustering_algorithm.labels_.astype(np.int)
            else:
                assign = atg.spectral_graph_partitioning_seq(qm_graph, K, merge=True, verbose=verbose)
            # find medoid of each partition. COMMENT: this line is ESSENTIAL. medoids fail if graph is not connected.
            medoids = atg.find_medoids_partitioned_graph(qm_graph, assign)

        except Exception as exc:
            print "spectral partitioning got exception: %s. Switching to k-medoids." % exc
            # Fall back to k-medoids
            init_method = 'k-medoids'
            
    elif init_method == 'metis':
        try:
            assign = farray(metis.part_graph(qm_graph, K)[1]) + 1
            medoids = atg.find_medoids_partitioned_graph(qm_graph, assign)
        except Exception as exc:
            print "Metis partitioning got exception: %s. Switching to k-medoids." % exc
            # Fall back to k-medoids
            init_method = 'k-medoids'
            
    if init_method == 'k-medoids':
        if is_sklearn:
            try:
                clustering_algorithm = cluster.MiniBatchKMeans(n_clusters=K)
                clustering_algorithm.fit(qm_at.positions)
                assign = clustering_algorithm.labels_.astype(np.int)
                medoids = atg.find_medoids_partitioned_graph(qm_graph, assign)
            except Exception as exc:
                print "Sklearn k-means partitioning got exception: %s. Switching to builtin k-medoids." % exc
                # Fall back to k-medoids
                medoids = initialise_medoids(at, qm_at, dm, K, verbose)
                medoids, assign = k_medoids(dm, medoids, max_step)
        else:
            ###### STANDARD K-MEDOIDS ######
            medoids = initialise_medoids(at, qm_at, dm, K, verbose)
            medoids, assign = k_medoids(dm, medoids, max_step)
  
    ##### DIGESTIVE RIPENING TO EQUALISE CLUSTER SIZES AND OPTIMISE CONVEXITY ######
    assign = np.asarray(assign)
    try:
        assign = digestive_ripening(dm, assign, algorithm=ripening_algorithm, verbose=verbose, movie_name=assign_movie)
    except Exception as err:
        print("Ripening algorithm %s failed. Digestive ripening skipped." % ripening_algorithm)
        print("ERROR - digestive ripening: %s" % err)

    ###### FINAL MEDOID ASSIGNMENT ######
    
    # transform assign into farray, with values starting from 1. Painful but needed for compatibility
    assign = farray(np.asarray(assign) - int(np.min(assign)) + 1)

    medoids = atg.find_medoids_partitioned_graph(qm_graph, assign)
    # transform medoids from graph node index Atoms index (python indexing to fortran indexing)
    medoids = [m + 1 for m in medoids]

    at.add_property('qm_cluster', 0, overwrite=True)
    at.qm_cluster[qm_list] = assign
    
    if silica:
        qm_o = np.logical_and(at.hybrid_mark == HYBRID_ACTIVE_MARK,
                              at.z == 8).nonzero()[0]
        for i in qm_o:
            # see which subgraph our nearest neighbours are assigned to
            nj = [n.j for n in at.neighbours[i]]
            cs = at.qm_cluster[nj]
            count_regions = sorted([(((cs == c).sum()).item(), c) for c in set(cs)])
            counts, regions = zip(*count_regions)
            if len(counts) == 1:
                # all neighbours are in the same region, so it's easy, we go there too
                at.qm_cluster[i] = regions[0]
            else:
                # pick the most popular region
                at.qm_cluster[i] = regions[-1]
                
    for i, m in fenumerate(medoids):
        print 'QM cluster %d: medoid %d size %d' % (i, m, (at.qm_cluster == i).sum())

    at.add_property('qm_medoids', 0, overwrite=True)
    for i, m in fenumerate(medoids):
        at.qm_medoids[qm_at.orig_index[m]] = i

    # Generate AtomsList movie from the lightweight assign movie.
    if make_movie:
        with open(assign_movie, 'r') as f:
            assigns = np.loadtxt(f, dtype='int', delimiter=',')
            for assign in assigns:
                at_movie.add_property('qm_cluster', assign, overwrite=True)
                moviewriter.write(at_movie)
        moviewriter.close()
    return medoids
