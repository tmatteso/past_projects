from DTMP import DTMP_tree as T
import numpy as np

def first_pass(loci_index_1, loci_index_2, mouse_list, recomb_map, allele_set, homo_dict):
    # first pass
    for mouse in mouse_list[-100:]:
        if mouse.genotype is not None:
            for i in allele_set:
                for j in allele_set:
                    if i != j and dual_homo(mouse.genotype[loci_index_1], mouse.genotype[loci_index_2], i, j):
                        homo_dict[(i, j)].append(mouse)
            # if the mouse_list is not empty, find the MRCA of the mice and store it in the recomb map
            for key, values in homo_dict.items():
                # print(values)
                if len(values) != 0:
                    # sometimes a mouse list does not have a common ancestor! -- ASK ABHI
                    if T.greedy_MRCA(values) is not None:
                        #print(T.greedy_MRCA(values), "first pass")
                        recomb_map[(loci_index_1, loci_index_2, key)] = T.greedy_MRCA(values)
    return homo_dict, recomb_map


def exactly_one_desc_LS_SL(mouse, recomb_map_entry):
    if ((mouse.father.is_ancestor(recomb_map_entry) and not mouse.mother.is_ancestor(recomb_map_entry))
    or (mouse.mother.is_ancestor(recomb_map_entry) and not mouse.father.is_ancestor(recomb_map_entry))):
        return True
    return False


def neither_is_desc(mouse, recomb_map_entry):
    if (not mouse.father.is_ancestor(recomb_map_entry)) and (not mouse.mother.is_ancestor(recomb_map_entry)):
        return True
    return False


def hetero_handling(mouse, i,j, loci_index_1, loci_index_2, recomb_map):
    # first check if this entry exists in Recomb map
    # what do we do if it isn't in the recomb map already?
    if (loci_index_1, loci_index_2, (i, j)) in recomb_map.keys():
        # check if mouse has exactly one parent that is a descendant of L,S or S,L
        if exactly_one_desc_LS_SL(mouse, recomb_map[(loci_index_1, loci_index_2, (i, j))]):
            # we have to reassign the mouse genotype
            mouse.genotype[loci_index_1], mouse.genotype[loci_index_2] = i, j
        # if mouse has two parents that are descendants, we cannot do anything, don't add anything

        # if neither is a descendant
        if neither_is_desc(mouse, recomb_map[(loci_index_1, loci_index_2, (i, j))]):
            reassess_recomb = [recomb_map[(loci_index_1, loci_index_2, (i, j))], mouse]
            # what if no mrca?
            if T.greedy_MRCA(reassess_recomb) is not None:
                recomb_map[(loci_index_1, loci_index_2, (i, j))] = T.greedy_MRCA(reassess_recomb)
            # now rerun the the first two checks
            # check if mouse has exactly one parent that is a descendant of L,S or S,L
            # could shorten to just the first key
            if exactly_one_desc_LS_SL(mouse, recomb_map[(loci_index_1, loci_index_2, (i, j))]):
                # we have to reassign the mouse genotype
                mouse.genotype[loci_index_1], mouse.genotype[loci_index_2] = i, j
        # if mouse has two parents that are descendants, we cannot do anything

    return mouse


def second_pass(loci_index_1, loci_index_2, mouse_list, recomb_map, allele_set, homo_dict):
    # first pass
    for mouse in mouse_list[-100:]:
        if mouse.genotype is not None:
            for i in allele_set:
                for j in allele_set:
                    # make this a subroutine
                    if i != j and single_hetero(mouse.genotype[loci_index_1], mouse.genotype[loci_index_2], i, j):
                        #print(mouse.genotype[loci_index_1], mouse.genotype[loci_index_2])
                        mouse = hetero_handling(mouse, i,j, loci_index_1, loci_index_2, recomb_map)
                        homo_dict[(i, j)].append(mouse)

            # if the mouse_list is not empty, find the MRCA of the mice and store it in the recomb map
            for key, values in homo_dict.items():
                # print(values)
                if len(values) != 0:
                    # sometimes a mouse list does not have a common ancestor! -- ASK ABHI
                    #recomb_map[(loci_index_1, loci_index_2, key)] = T.find_MRCA(values)
                    if T.greedy_MRCA(values) is not None:
                        #print(T.greedy_MRCA(values), "second pass")
                        recomb_map[(loci_index_1, loci_index_2, key)] = T.greedy_MRCA(values)
    return homo_dict, recomb_map

# 4a. For each genotyped mouse in G56, impute in genotypes for its nuclear family when possible.
# Move up the pedigree generation by generation until all nuclear families of genotyped mice have
# been considered (forward pass).

# how do we use the adjacent genotypes for imputation?


# how do we do the forward pass?
# we have a leaf set and can then do an upward pass, problem solved

def third_pass(loci_index_1, loci_index_2, leaf_list, recomb_map, allele_set, homo_dict):
    # move up from bottom nodes
    for mouse in leaf_list:
        # still necessary, otherwise there is no phasing to be done
        if mouse.genotype is not None:
            for i in allele_set:
                for j in allele_set:
                    # need to recursively call the modded upward pass here

                    pass

# def rules of inheritance

# find siblings (defined as children of same parent, calc individually for each parent)
# said that he wants this as a forward pass (upward), so start at leaves, go up one level from each leaf, keeping track
def find_siblings_of_parent(parent):
    grandparents = [parent.father, parent.mother]
    s_set = set()
    for grandparent in grandparents:
        s_set.union(grandparent.progeny)
    return s_set


def possible_donated_alleles(s_set, marker1, marker2):
    ls = []
    for mouse in s_set:
        ls.append([mouse.genotype[marker1], mouse.genotype[marker2]])
    return ls


# need a pair of markers here
def resolve_haplotype(mouse, i, j):
    parents = [mouse.father, mouse.mother]
    possible_parent_alleles = []
    for parent in parents:
        if parent.genotype is not None:
            possible_alleles = possible_donated_alleles(set(parent), i, j)
        else:
            sibling_set = find_siblings_of_parent(parent)
            possible_alleles = possible_donated_alleles(sibling_set, i, j)
        possible_parent_alleles.append(possible_alleles)
    possible_parent_alleles = np.array(possible_parent_alleles)
    print(possible_parent_alleles)
    #if (mouse.genotype[i] in possible_parent_alleles[0]) and (mouse.genotype[j] in possible_parent_alleles[0]):
        #pass


    # if not find siblings of each parent
    #


# modify this function to find parents
# what is the most effective data structure for this purpose?
def upward_traversal(node, seen_set, root_list):
    # just keep going up from node, keeping track of what nodes you see along the way
    # {child: [parent1, parent2]}
    # perform inheritance task
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




# then wrap in a function with the two for loops
def all_locii_pairs(genotype_map, mouse_list):
    recomb_map = dict()
    homo_dict = {("L", "S"): [], ("S", "L"): [], ("B", "S"): [], ("S", "B"): [],
                ("B", "L"): [], ("L", "B"): []}
    allele_set = ["L", "S", "B"]
    # has_LLLS = []
    # subset the locii number
    for loci_index_1 in range(9):
        homo_dict, recomb_map = first_pass(loci_index_1, loci_index_1+1, mouse_list, recomb_map, allele_set, homo_dict)
        homo_dict, recomb_map = second_pass(loci_index_1, loci_index_1+1, mouse_list, recomb_map, allele_set, homo_dict)
        #print("homo", homo_dict)
        #print("recomb", recomb_map)

    # print(recomb_map)

def determine_mouse_phase(genotype_map):
    # init_recomb_map()
    pass

    # upward_traversal(node, seen_set, root_list)

    # downward_traversal(node, seen_set, root_list)

    # add_to_recomb_map()
# might be best to have an allele class
# then have a genotype class that inherits from allele
# we'll come back to this

# ask about this!
def determine_parent_identity(marker, recomb_map):
    # split the marker into parental alleles
    allele_1, allele_2 = marker[0], marker[2]
    # if recomb_map shows event before parent, then will assume typical Mendelian allele assortment
    parent_allele_1 = "a"
    # if recomb_map shows recomb at parent
    parent_allele_2 = "b"
    return parent_allele_1 + " " + parent_allele_2



def dual_homo(str1, str2, desired_1, desired_2):
    if str1[0] == str1[2] == desired_1 and str2[0] == str2[2] == desired_2:
        return True
    return False

def single_hetero(str1, str2, desired_1, desired_2):
    if str1[0] == desired_1 and str1[2] == desired_2 and str1[0] != str1[2] and str2[0] == str2[2] == desired_1:
        return True
    return False
"""
def one_marker_is_unphased(marker_1, marker_2):
    if is_unphased_genotype(marker_1) or is_unphased_genotype(marker_2):
        if is_phased_genotype(marker_1) or is_phased_genotype(marker_2):
            return True
    return False

def both_markers_are_unphased(marker_1, marker_2):
    if is_unphased_genotype(marker_1) and is_unphased_genotype(marker_2):
        return True
    return False

def both_markers_are_phased(marker_1, marker_2):
    if is_phased_genotype(marker_1) and is_phased_genotype(marker_2):
        return True
    return False
"""


