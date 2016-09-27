import scipy as sp
import networkx as nx
from networkx.algorithms.shortest_paths import single_source_shortest_path_length as splength

try:
    from joblib import Parallel, delayed
    import multiprocessing
    is_parallel = True
except:
    is_parallel = False



def append_partitioning(filename, assign):
    with open(filename, 'ab') as f:
        sp.savetxt(f, assign[None], fmt='%d', delimiter=',')
    return

    
def distancemap_to_nxgraph(dm):
    return nx.from_numpy_matrix(dm == 1)

    
def G(dm, assign, part, k=1.0, consistent=False):
    """
    Calculate free energy of region i using distance matrix dm and assignment assign
    k measures the relative importance of surface and 1/diameter components of G (arbitrarily)

    WARNING: 
    consistent=False: quick but questionable. Slices distance map of the whole graph, while distance between nodes in 
    actual subgraphs should be recalculated because of cut edges.
    consistent=True: fully consistent definition of G but slower to compute.
    """

    if consistent:
        subgraph = graph.subgraph(sp.where(assign == part)[0])
        sub_dm = sp.zeros((subgraph.number_of_nodes(), subgraph.number_of_nodes()))
        try:
            for i, node in enumerate(subgraph.nodes()):
                sub_dm[i,:] = sp.array([w for k, w in splength(subgraph, node).iteritems()])
        except Exception as err:
            # no path connecting a couple of nodes: subgraph is not connected, so G must be infinite
            return sp.inf
    else:
        sub_dm = dm.view(sp.ndarray)[sp.ix_(assign == part, assign == part)]

    # surface energy alone already does a very good job
    # since it is an extensive property it favours same size subgraphs
    surface_energy = (sub_dm**2).sum()
    if sp.size(sub_dm) > 1:
        r = sub_dm.max() / float((assign == part).sum())**2
    else:
        r = 1e-6
    return float(surface_energy + k/float(r))


def G_whole(dm, assign, k=1.0, consistent=False):
    return sp.sum([G(dm, assign, part, k=k, consistent=consistent) for part in sp.unique(assign)])


def region_neighbouring_atoms(graph, assign, i):
    """
    Find inter-regional neighbours of the members of region i
    graph is the complete graph, assign is list of which atom belongs to whom
    (r, (j, k)) are indices of atom pair (i,j) crossing border between regions i--r
    """
    subgraph = graph.subgraph((assign == i).nonzero()[0])
    restgraph = graph.subgraph((assign != i).nonzero()[0])
    missing_edges = list(set(graph.edges()).difference(set(subgraph.edges()+restgraph.edges())))

    neighbours = [((lambda j, k: assign[k] if assign[j] == i else assign[j])(j,k), (j,k))
                  for j, k in missing_edges if assign[j] != assign[k]]
    return neighbours


def partitions_are_connected(dm, assign, affected_regions, verbose=False):
    # type(affected_regions) = list
    # make edge matrices from the distance map provided. edge = bond between atom i and j
    # create n graphs from adjacency matrices
    graph = distancemap_to_nxgraph(dm)
    graphs = [ graph.subgraph((assign == i).nonzero()[0]) for i in affected_regions ]
    # boolean results
    results = [nx.is_connected(graph) for graph in graphs]
     # print out non-connected regions
    for region, res in zip(affected_regions, results):
        if not res and verbose:
            print "QM region %d is not connected" % region
            print connectivity_matrices[:]
    # True if all checked regions are connected
    all_connected = sp.alltrue(results)
    return all_connected, results


def remove_node_is_connected(graph, assign, i, removed_node):
    """
    Given a graph object partitioned according to the array-like assign mask,
    take away node removed_node from subgraph i and check if i is still connected.
    Necessary and sufficient condition for subgraph i connectivity is that
    all old neighbours of i have a finite path length to all nodes in i.
    """
    assign = sp.asarray(assign)
    subgraph = graph.subgraph((assign == i).nonzero()[0])
    neighbours = subgraph.neighbors(removed_node)
    subgraph.remove_node(removed_node)
    # return nx.is_connected(subgraph)
    for node in neighbours:
        path_lengths_to_node = [l for l in splength(subgraph, node)]
        if len(path_lengths_to_node) != subgraph.number_of_nodes():
            return False
    return True


def dG_atom_swap(dm, assign, i, neighbours, G_cur, quick_exit):
    """
    Generate list of all possible moves involving region i
    'in' means taking jj from region j into region i
    'out' means moving ii out of region i into region j
    dm = total distance map; assign = lists current owner of atom; i = sub-cluster considered for exchange;
    neighbours = list of boundary atoms of sub-cluster i; G_cur = energy of system before swap
    """

    moves = []
    for j, (ii, jj) in neighbours:
        # try gaining one atom if j has > 1 atom
        if int((assign == r).sum()) > 1:
            new_assign = assign.copy()
            new_assign[jj] = i
            G_in = G(dm, new_assign, i) + G(dm, new_assign, j)
            dG = G_in - (G_cur[i] + G_cur[j])
            if dG < 0.0:
                moves.append((dG, 'in', j, ii, jj))
                if quick_exit: break
        # try losing one atom if i has > 1 atom
        if int((assign == i).sum()) > 1:
            new_assign = assign.copy()
            new_assign[ii] = j
            G_out = G(dm, new_assign, i) + G(dm, new_assign, j)
            dG = G_out - (G_cur[i] + G_cur[j])
            if dG < 0.:
                moves.append((dG, 'out', j, ii, jj))
                if quick_exit: break
    return moves
        

def digestive_ripening_move(graph, dm, moves, i, assign, verbose=False):
    """
    Given a set of possible moves for a single subgraph - i -, do the most rewarding that still conserves connectivity of subgraphs.
    """
    found_good_move = False
    if len(moves) > 0:
        # better move has more negative \Delta G
        moves = sorted(moves)
        for attempt in range(len((moves))): #3
            # node ii is in subgraph i, node jj is in subgraph j
            dG, in_out, j, ii, jj = moves[attempt]
            # try move that lowers total G the most. If move creates disconnected clusters, try next best one and so on.
            new_assign = assign.copy()
            if in_out == 'in':
                new_assign[jj] = i
                no_broken_cluster = remove_node_is_connected(graph, assign, j, jj)
            elif in_out == 'out':
                new_assign[ii] = j
                no_broken_cluster = remove_node_is_connected(graph, assign, i, ii)
            else:
                raise ValueError('bad in_out %r' % in_out)

            if no_broken_cluster and dG < 0.: 
                if verbose:
                    print "MOVE: region %02d | %s | possible moves: %02d | successful: %02d" % (i, in_out, len(moves), attempt+1)
                assign[:] = new_assign
                found_good_move = True		
                break #3  jump out of list of moves for cluster i
            else:
                # look for next move for cluster i
                continue
    return found_good_move, assign


def one_swap(graph, edge_idx, edges_temp, assign, G_cur, dm):
    
    edge = edges_temp[edge_idx]
    part_0, part_1 = assign[edge]
    if part_0 != part_1:
        assign_new = assign.copy()
        assign_new[edge[1]] = part_0
        G_new =  G_whole(dm, assign_new)
        if G_new < G_cur:
            if remove_node_is_connected(graph, assign, part_1, edge[1]):
                return G_new, assign_new, edge_idx
    # in all other cases, return the old stuff
    return G_cur, assign, edge_idx


def digestive_ripening_steepest(dm, assign, quick_exit=False, verbose=False, movie='none'):

    graph = distancemap_to_nxgraph(dm)
    found_good_move = True
    while found_good_move:
        G_cur = sp.array([G(dm, assign, i) for i in sp.unique(assign)])
        sorted_indices = (abs(G_cur - G_cur.mean())).argsort()[::-1]
        if verbose:
            print G_cur.round(1), ' '.join(['%3d' % (assign == i).sum() for i in sp.unique(assign) ])
            print "DEBUG: region %02d goes first" % sorted_indices[1]
        for i in sorted_indices: #2
            # find inter-regional neighbours of the members of region i
            neighbours = region_neighbouring_atoms(graph, assign, i)        
            # generate list of \Delta G of all possible single-atom swaps involving region i
            moves = dG_atom_swap(dm, assign, i, neighbours, G_cur, quick_exit)
            found_good_move, assign = digestive_ripening_move(graph, dm, moves, i, assign, verbose=verbose)
            if found_good_move:
                if movie != 'none':
                    append_partitioning(movie, assign)
                break #2 jump out of list of subgraphs. New assign is accepted, recalculate G.
            else:
                # look for a move with another subgraph
                continue
    return assign

    
def digestive_ripening_MC(dm, assign, verbose=False, movie='none'):

    def edgy_nodes(edges):
        # from a list of edges return a list of unique nodes at those edges
        return list(set(sp.array(edges).flatten()))
    
    whole_graph =  distancemap_to_nxgraph(dm)
    subgraphs = [whole_graph.subgraph(sp.where(assign == i)[0]) for i in sp.unique(assign)]
    kept_edges = []; [kept_edges.extend(s.edges()) for s in subgraphs]
    cut_edges = list(set(whole_graph.edges()) - set(kept_edges))
    adj_list = whole_graph.adjacency_list()
    
    G_cur = sp.sum([G(dm, assign, part) for part in sp.unique(assign)])
    if verbose:
        print("Current G: %e" % G_cur)
    nothing_new = 0
    while True:
        if nothing_new > whole_graph.number_of_nodes():
            break
        node = sp.random.choice(edgy_nodes(cut_edges))
        node_assign_old = assign[node]
        assign_new = assign.copy()
        # try assigning node to graph of one of its neighbours if neighbour is not of the same subgraph
        try:
            node_assign_new = assign[sp.random.choice(list(set(adj_list[node]) - set(subgraphs[node_assign_old].nodes())))]
        except IndexError:
            # there are no neighbouring nodes
            continue
        assign_new[node] = node_assign_new
        if sp.sum([G(dm, assign_new, part) for part in sp.unique(assign_new)]) < G_cur:
            nothing_new = 0
            # accept move: update assign list, refresh subgraphs, refresh edges lists
            assign[:] = assign_new
            subgraphs[node_assign_old] = whole_graph.subgraph(sp.where(assign == node_assign_old)[0])
            subgraphs[node_assign_new] = whole_graph.subgraph(sp.where(assign == node_assign_new)[0])
            kept_edges = []; [kept_edges.extend(s.edges()) for s in subgraphs]
            cut_edges = list(set(whole_graph.edges()) - set(kept_edges))
            G_cur = sp.sum([G(dm, assign, part) for part in sp.unique(assign)])
            if verbose:
                print("Current G: %e" % G_cur)
                if movie != 'none':
                    append_partitioning(movie, assign)
        else: # maybe add temperature at some point
            nothing_new += 1
            continue
    return assign


def digestive_ripening_fullrand(dm, assign, verbose=False, movie='none', parallel=False):


    """
    'fullrand' scheme:
    Generate list of all edges of the graph "edges" and a copy of it "edges_temp"--- repeated: list contains both (i,j) and (j,i) for nodes i, j
    (1) Pick one edge in "edges_temp" at random
    (2) If it is on the boundary between one subgraph and another, try moving second node of the edge into the subgraph of the first node.
        If new assignment conserves connectivity AND is advantageous, accept it and reset edges_temp = edges, go to (1)
        Else take away that edge from "edges_temp" and go to (1)
    (3) if edges_temp becomes empty, we ran out of subgraph swaps possibilities, algorithm is converged. Quit.
    """
    graph = distancemap_to_nxgraph(dm) 
    # generate list of all edges in the graph
    edges = sp.array(sp.where(dm == 1)).T
    # Calculate free energy of starting partitioned graph
    G_cur = G_whole(dm, assign)

    nothing_happened = 0
    threshold = len(edges)
    edges_temp = list(edges.copy())
    num_cores = multiprocessing.cpu_count()        

    if parallel:
        while True:
            results = Parallel(
                n_jobs=sp.minimum(num_cores, len(edges_temp))(delayed(one_swap)(
                    graph, edge_idx, edges_temp, assign, G_cur, dm) for edge_idx in sp.random.sample(
                        range(len(edges_temp)), sp.minimum(num_cores, len(edges_temp)))))
            Gs = [r[0] for r in results]
            found = False
            # go through all the solution that provide a decrease in G, starting from the most advantageous
            for index in [i for i in sp.argsort(Gs) if Gs[i] < G_cur]:
                assign_new = sp.array(results[index][1])
                edge_idx = results[index][2]
                # check if the new partitioning would be connected
                part_0, part_1 = assign_new[edges_temp[edge_idx]]
                if remove_node_is_connected(graph, assign, part_1, edges_temp[edge_idx][1]):
                    found = True
                    G_cur = G_new
                    assign[:] = assign_new
                    nothing_happened = 0
                    edges_temp = list(edges.copy())
                    if verbose:
                        print("Current G: %e" % G_cur)
                        if movie != 'none':
                            append_partitioning(movie, assign)
                else:
                    continue
            if not found:
                # if you haven't found any good move, increase the counter for it by the number of attempts and
                # delete those attempts from the edges_temp list (be careful to delete the right ones!)
                nothing_happened += sp.minimum(num_cores, len(edges_temp))
                for edge_idx in sorted([r[2] for r in results], reverse=True):
                    edges_temp.pop(edge_idx)
                    
            if nothing_happened < threshold and len(edges_temp) > 0:
                continue
            else:
                break
                    
    else: # serial algorithm
        while True:
            # pick an edge of the graph at random and see where the nodes belong
            edge_idx = sp.random.choice(range(len(edges_temp)))
            edge = edges_temp[edge_idx]
            part_0, part_1 = assign[edge]
            found = False
            if part_0 != part_1:
                assign_new = assign.copy()
                assign_new[edge[1]] = part_0
                G_new =  G_whole(dm, assign_new)
                if G_new < G_cur:
                    if remove_node_is_connected(graph, assign, part_1, edge[1]):
                        found = True
                        G_cur = G_new
                        assign[:] = assign_new
                        nothing_happened = 0
                        edges_temp = list(edges.copy())
                        if verbose:
                            print("Current G: %e" % G_cur)
                        if movie != 'none':
                            append_partitioning(movie, assign)
            if not found:
                nothing_happened += 1
                edges_temp.pop(edge_idx)
                if nothing_happened < threshold and len(edges_temp) > 0:
                    continue
                else:
                    print "done"
                    break
    return assign


def digestive_ripening_metropolis(dm, assign, verbose=False, movie='none'):

    graph = distancemap_to_nxgraph(dm)
    # generate list of all edges in the graph
    edges = sp.asarray(sp.where(dm == 1)).T
    # Calculate free energy of starting partitioned graph
    G_cur = G_whole(dm, assign)

    nothing_happened = 0
    threshold = len(edges)
    edges_temp = list(edges.copy())
    T, const = 0., .5
    while True:
        # pick an edge of the graph at random and see where the nodes belong
        edge_idx = sp.random.choice(range(len(edges_temp)))
        edge = edges_temp[edge_idx]
        part_0, part_1 = assign[edge]
        found = False
        if part_0 != part_1:
            assign_new = assign.copy()
            assign_new[edge[1]] = part_0
            G_new =  G_whole(dm, assign_new)
            if G_new < G_cur or sp.exp(-const*(1.01*G_new - G_cur)/T) > sp.random.random():
                if remove_node_is_connected(graph, assign, part_1, edge[1]):
                    found = True
                    if verbose:
                        print("Delta G: %e,\t T: %f" % (G_new - G_cur,T))
                    if movie != 'none':
                        append_partitioning(movie, assign)
                    # WRONG: currently heats up when good moves are found. should be the opposite
                    T = 0.5*(sp.array([G_cur - G_new, 0.]).max()/const + T)

                    G_cur = G_new
                    assign[:] = assign_new
                    nothing_happened = 0
                    edges_temp = list(edges.copy())
                    # temperature is changed at every step: its inverse is proportional to the last change in energy, but contains history of all past

        if not found:
            nothing_happened += 1
            edges_temp.pop(edge_idx)
            if nothing_happened < threshold and len(edges_temp) > 0:
                continue
            else:
                print "done"
                break
    return assign


def digestive_ripening(dm, assign, algorithm='monte_carlo', verbose=False, movie_name='none'):

    """
    Digestive ripening of a graph to equalise size and optimise convexity of partitioning given initial partitioning guess.

    Parameters:

    dm: array_like
        Array with shape (number_of_nodes, number_of_nodes), dtype int
        Shortest path length distance matrix of the whole graph. 

    assign: array_like
        Array with shape (number_of_nodes), dtype int
        Initial partitioning attribute for each node in the graph.
        Must produce connected subgraphs.

    algorithm: string
        Available possibilities:
        'steepest_descent' (very accurate, slower), 'fullrand' (accurate, fastest), 'monte_carlo' (somewhere in between)
        Working but deprecated methods:
        'steepest_descent_QE'
        Methods under development (use only for testing):
        'metropolis', 'fullrand_par'

    verbose: logical
        If True, print out information about intermediate steps of the ripening process.

    movie_name: string
       Name of the file on which to append the assign array step by step in csv format.
       If 'none', no output is created.
    """
    
    assign[:] = (sp.array(assign) - sp.array(assign).min()).astype(int) # convert to zero-based Python
    dm = sp.asarray(dm).astype(int)
    # Write the initial partitioning to the movie file
    if movie_name != 'none':
        with open(movie_name, 'wb') as f:
            sp.savetxt(f, assign[None], fmt='%d', delimiter=',')
    
    if algorithm == 'steepest_descent':
        assign = digestive_ripening_steepest(dm, assign, quick_exit=False, verbose=verbose, movie=movie_name)
    elif algorithm == 'steepest_descent_QE':
        assign = digestive_ripening_steepest(dm, assign, quick_exit=True, verbose=verbose, movie=movie_name)
    elif algorithm == 'monte_carlo':
        assign = digestive_ripening_MC(dm, assign, verbose=verbose, movie=movie_name)
    elif algorithm == 'fullrand':
        assign = digestive_ripening_fullrand(dm, assign, verbose=verbose, movie=movie_name, parallel=False)
    elif algorithm == 'fullrand_par' and is_parallel:
        assign = digestive_ripening_fullrand(dm, assign, verbose=verbose, movie=movie_name, parallel=True)
    elif algorithm == 'metropolis':
        assign = digestive_ripening_metropolis(dm, assign, verbose=verbose, movie=movie_name)
    else:
        print("No ripening")
    return assign
    
