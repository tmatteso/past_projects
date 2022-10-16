
def max_gen_ancestor(ancestor_set):
    max_gen = 0
    max_gen_ancestor = None
    for ancestor in ancestor_set:
        if ancestor.generation_number > max_gen:
            max_gen = ancestor.generation_number
            max_gen_ancestor = ancestor
    return max_gen_ancestor


def upward_traversal(node, seen_set, root_list):
    # just keep going up from node, keeping track of what nodes you see along the way
    if node in root_list:
        #print(node.sample_id, "root")
        return seen_set

    seen_set.add(node)
    if node.mother and node.father:
        return upward_traversal(node.mother, seen_set, root_list), upward_traversal(node.father, seen_set, root_list)
    elif node.mother:
        return upward_traversal(node.mother, seen_set, root_list)
    elif node.father:
        return upward_traversal(node.father, seen_set, root_list)
    else:
        #print(node.sample_id, "non root")
        return seen_set


# find roots for downward traversals of tree
def find_roots(node_list):
    root_list = []
    #print(node_list[3].sample_id, node_list[3].father.sample_id, node_list[3].mother.sample_id)
    for node in node_list:
        if type(node.father) == str or type(node.mother) == str:
            root_list.append(node)
    return root_list


# find leaves for upward traversals of tree -- not already handled
# but the upward traversal itself is
# given a node_list, find  nodes with no children
def find_leaves(node_list):
    leaf_list = []
    for node in node_list:
        if len(node.progeny) == 0:
            leaf_list.append(node)
    return leaf_list


# need a recursive func for parental connection
def build_tree(pedigree_map, genotype_map):
    # build tree from pedigree
    # keep track of connectivity by adding a node to root list if it has no parent
    node_list = []
    for sample in (range(len(pedigree_map["id"]))):
        node_list.append(Node(pedigree_map["id"][sample], pedigree_map["sire"][sample],
             pedigree_map["dam"][sample], pedigree_map["generation"][sample], genotype_map))
    for node in node_list:
        node.connect_nodes(node_list)
    return find_roots(node_list), node_list

def get_all_ancestors(node_list, root_list):
    for node in node_list:
        node.find_ancestors(root_list)
    return node_list

# init recomb map
#recomb_map = dict()

# node will have genotype attribute, some nodes will have an additional .1 at the end, denoting lab transfer,
# use these labels as he sam

# (i, i+1, L -> S) -> mouse_id
# mouse_id is calculated as Intercept over all mice with this recombination (L-> S for i, i+1) on the ancestor of the of all the mice
class Node:
    def __init__(self, sample_id, father, mother, generation_number, genotype_map):
        self.sample_id = sample_id
        # mother and father are Node objects, taken from pedigree import
        self.father = father
        self.mother = mother
        self.generation_number = generation_number
        self.progeny = set()
        # will be set of Node objects
        # each mouse has an ancestor set dict: mouse -> ancestor set
        # is x an ancestor in mouse, then just check set
        self.ancestor_set = set()
        self.genotype = self.determine_genotype(genotype_map)


    def connect_nodes(self, node_list):
        # change parent strs to nodes, update progeny set of parents
        for node in node_list:
            if node.sample_id == self.father:
                self.father = node
                node.progeny.add(self)
            if node.sample_id == self.mother:
                self.mother = node
                node.progeny.add(self)


    def find_ancestors(self, root_list):
        seen_set = set()
        tuple_of_sets = upward_traversal(self, seen_set, root_list)
        #print([i.sample_id for i in tuple_of_sets[0]][:5])
        for set_e in tuple_of_sets:
            pass
        #self.set = set(ancestor_ls)
        #print(type(self.ancestor_set), "ls")
        for i in self.ancestor_set:
            if i.sample_id == self.sample_id:
                self.ancestor_set.remove(i)
                break


    def view_ancestors(self):
        return set(node.sample_id for node in self.ancestor_set)

    def is_ancestor(self, other_mouse):
        if other_mouse in self.view_ancestors(): return True
        return False
    # nodes with parents, genotype, sample_id, generation_number

    def determine_genotype(self, genotype_map):
        # genotype data must be imported from a vcf (which one-> use chr1 calls for now)
        # find sample numbers that correspond to related indivs
        # how is this accomplished -- literally matching based on sample number, but ignore numbers after dec
        sample_id_wo_lab_transfer = self.sample_id.split(".")[0]
        for sample in genotype_map.keys():
            if sample[2:] == sample_id_wo_lab_transfer:
                return genotype_map[sample]

# to find MRCA  between x,y : find max gen of ancestor of x and ancestor of y
# this is the old non greedy MRCA
def find_MRCA(mouse_list):
    # perform a set intercept between each mouse in the list
    # convert back to node objects
    # fix first index as first mice and iteratively intercept from there
    if len(mouse_list) == 1:
        return mouse_list[0]
    node_set = mouse_list[0].ancestor_set
    for other_mouse_index in range(1, len(mouse_list)):
        node_set.intersection(mouse_list[other_mouse_index].ancestor_set)

    # then run max gen ancestor
    return max_gen_ancestor(node_set)


# if blank, ignore them, no answer
# if no single MRCA, where do we map the events to?
# try greedy approach where we iteratively intersect the mice in the mouse list, if null, stop and start new partition, repeat.
# iteratively intersect, backtrack when null (need a list for memory)
# for now we keep track of 1 MRCA
def greedy_MRCA(mouse_list):
    # perform a set intercept between each mouse in the list
    # convert back to node objects
    # fix first index as first mice and iteratively intercept from there
    if len(mouse_list) == 1:
        return mouse_list[0]
    #print(mouse_list)
    node_set = mouse_list[0].ancestor_set
    keep_track = []
    for other_mouse_index in range(1, len(mouse_list)):
        node_set.intersection(mouse_list[other_mouse_index].ancestor_set)
        if len(node_set) != 0:
            keep_track.append(node_set.copy())
        else:
            # if you would break it with an intercept, return it right with the set you have right before
            if len(keep_track) != 0:
                return max_gen_ancestor(keep_track.pop())
            else:
                return max_gen_ancestor(set())
    # then run max gen ancestor
    return max_gen_ancestor(node_set)


def sample_to_node_map(node_list):
    # make a map from each sample_id_wo_lab_transfer to node for fast lookups
    sample_to_node_map = dict()

    for node in node_list:
        sample_id_wo_lab_transfer = node.sample_id.split(".")[0]
        sample_to_node_map[sample_id_wo_lab_transfer] = node

    return sample_to_node_map

