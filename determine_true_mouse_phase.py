from DTMP import DTMP_import as I
from DTMP import DTMP_tree as T
from DTMP import DTMP_phasing as P


def main():
    col_names, pedigree_map = I.import_pedigree("pedigree.lgsm.G56.txt")
    col_list, genotype_map = I.import_genotypes("ail.genos.lsob.chr8.vcf")
    roots, all_nodes = T.build_tree(pedigree_map, genotype_map)
    P.all_locii_pairs(genotype_map, all_nodes)
    leaf_list = T.find_leaves(all_nodes)
    count = 0
    for mouse in leaf_list:
        if mouse.genotype is not None:
            for i in range(len(mouse.genotype)):
                print(mouse.genotype[i], mouse.genotype[i+1], "hey")
                print(mouse.father.genoytype[i], mouse.father.genoytype[i+1], )
                P.resolve_haplotype(mouse, i, i+1)
                #print(mouse.father.genotype[i], mouse.father.genotype[i+1], mouse.mother.genotype[i], mouse.mother.genotype[i+1])
                break
            break



    #print((genotype_map["52044.1"])[0])
    # this a col, each entry in the ls is a row for the mouse
    # print(type(T.sample_to_node_map(all_nodes)["54868"].genotype))
    #print(len(T.sample_to_node_map(all_nodes)["52044"].genotype))
    #P.all_locii_pairs(genotype_map, all_nodes)
    #print([i.sample_id for i in roots])
    #print(T.get_all_ancestors(all_nodes, roots)[0].sample_id)
    #print()
    #print(len(all_nodes), len(roots))
    #print([root.sample_id for root in roots])

    """




    # take in a sys arg here eventually
    marker_ls = []

    # data type for pedigree?
    # dict with key as mouse sample, value is tuple of parents
    # a linked_list could be nice too
    # we should really be thinking about the I/O required here and what to optimize
    # mouse_1 -> mouse_2, mouse_3 -> (mouse2a, mouse2b), (mouse3a, mouse3b) -> ...
    # obviously this kind of traversal is terrible
    pedigree = dict()

    # consider that the input may be vcf-like
    # must know identity of a mouse given sample_name
    # key is sample_name, value is ls with genotype for each position
    # we are indexing form the marker_ls after all, so we will know which positions and they will be constant
    mouse_identity = dict()
    # recursive statement
    for marker_1, marker_2 in marker_ls:
        # ident mice with m_1 = L|L, m_2 = S|S, add MRCA of these mice to recomb_map as origin
        if both_markers_are_phased(marker_1, marker_2):
            # checks LL and SS, as well as SS and LL
            if both_are_L(marker_1) or both_are_L(marker_2):
                if both_are_S(marker_1) or both_are_S(marker_2):
                    recomb_map[(marker_1, marker_2)] = find_MRCA()

        elif both_markers_are_unphased(marker_1, marker_2):
            pass

        elif one_marker_is_unphased(marker_1, marker_2):
            # ident all m_1 = L|L, m_2 = L\S where \ means the genotype here is not phased.
            # same but flip m_1 and m_2
            if both_are_L(marker_1) or both_are_L(marker_2):
                if each_is_different(marker_1) or each_is_different(marker_2):
                    # L|L and (L|S, S|L) or vice versa
                    # consult recomb_map to determine identity of each parent
                    # do we have to keep track of larger chains of descendance via the pedigree and sample look up
                    # or is it suffcicient to determine the identity of the parent and consult the recomb map to ensure no funny business?
                    if each_is_different(determine_parent_identity(marker_1, recomb_map)):
                        if each_is_different(determine_parent_identity(marker_2, recomb_map)):
                            # then haplotype is ambiguous
                            # what do I do here?
                            # does this mean I set the phased genotype to unphased, given that one of the two was already unphased?
                            pass
                        elif both_are_L(determine_parent_identity(marker_2, recomb_map)):
                            # describe what the hell this is!
                            # phase the alleles for the markers to each parent

                    elif both_are_L(determine_parent_identity(marker_1, recomb_map)):
                        if each_is_different(determine_parent_identity(marker_2, recomb_map)):


                    recomb_map[(marker_1, marker_2)] = find_MRCA()"""

if __name__ == "__main__":
    main()