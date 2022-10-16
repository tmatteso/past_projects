# make tree first from pedigree
    # then fil in nodes from genotype data
    # then begin algorithm
    # the pedigree wil be fully imputed
    # and unphased vcf files


def build_lookup_table(col_list, ls_of_ls):
    lookup_table = dict()
    # put into lookup_table
    for index in range(len(col_list)):
        lookup_table[col_list[index]] = [ls_of_ls[0][index]]
        #print("oh hi", ls_of_ls[index])
        for row_index in range(1, len(ls_of_ls)):
            lookup_table[col_list[index]].append(ls_of_ls[row_index][index])
    return lookup_table


def import_genotypes(filename):
    with open(filename) as f:
        lookup_table = dict()
        ls_of_ls = []
        counter = 0
        for line in f:
            if line.startswith("#CHROM"):
                # read in the first non-skipped line as the column names
                col_list = line.strip("\n").split("\t")
                counter += 1
                continue
            if counter >= 1 and line:
                line_in = (line.strip("\n")).split("\t")
                ls_of_ls.append(line_in)
                counter += 1
            if counter == 1000:
                break

    return col_list, build_lookup_table(col_list, ls_of_ls)


def import_pedigree(filename):
    with open(filename) as f:
        ls_of_ls = []
        for line in f:
            if line.startswith("id"):
                # read in the first non-skipped line as the column names
                col_list = line.strip("\n").split("\t")
            else:
                line_in = (line.strip("\n")).split("\t")
                ls_of_ls.append(line_in)
    # print(ls_of_ls[0])
    return col_list, build_lookup_table(col_list, ls_of_ls)





