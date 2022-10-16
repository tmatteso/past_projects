from functions import MatchIdName
import numpy as np
import csv
import pulp

# grab the column names from the given file 
def get_cols(file_name):
    # 'renal_zero_one_matrix.csv' w/ threshold 0 was the inital data I used
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        col_names = next(reader)
    return col_names
    
# this converts a row of binary numbers {0,1} for each col into a set of column numbers
def bin_vect_to_set(bin_vect):
    out_set = set()
    for i in range(len(bin_vect)):
        if bin_vect[i].any() == 1:
            out_set.add(i)
    return tuple(out_set)    
            
# this applies bin_vect_to_set across a list of binary vectors (rows, so really row vectors)
def all_bvs_to_sets(ls_of_bvs):
    output_ls = []
    for bin_vect in ls_of_bvs:
        output_ls.append(bin_vect_to_set(bin_vect))
    return output_ls

# each row is a drug
# each col is a cell line from that tumor primary site

# now i want you to go though and comment this correctly and relabel w appropriate var names
def define_model(col_names, file_name):
    # subset the columns by omitting the drug_names (the first col)
    import_cols = tuple([i for i in range(1, len(col_names))])
    # there will be a loss of info from this, need to keep track of indices later
    drug_to_cell_line_effect_set = list(set(all_bvs_to_sets(np.loadtxt(open(file_name, "rb"), 
                                             delimiter=",", skiprows=1, usecols =import_cols))))
    # maximal set of cell_lines, i.e. all cell_lines in a set (technically a tuple here)
    all_cell_lines_effected = tuple([i for i in range(len(col_names)-1)])
    
    # if one drug covers all cell lines, tell the user that answer is trivial
    if all_cell_lines_effected in drug_to_cell_line_effect_set:
        print("answer is trivial")
        
    # Define the ILP model for Minimal Hitting Set from our drugs to our cell_lines
    ILP_model = pulp.LpProblem("MHS Model", pulp.LpMinimize)

    # Define the decision variables, with lower bound 0, upper bound 1, as ints
    # the decision variables are in terms of the rows (how a drug effects all the cell lines)
    x = pulp.LpVariable.dicts(
        "table", drug_to_cell_line_effect_set, lowBound=0, upBound=1, cat=pulp.LpInteger)

    # now we add the obj funct, which is a minimization of the sum of the rows 
    # i.e. the minimal number of rows we need to satisfy the minimization
    # the RHS of the minimization has not yet been satisfied
    ILP_model += (pulp.lpSum([x[drug_cover] for drug_cover in drug_to_cell_line_effect_set]))

    # now we add the constraints governing the use of each row (drug)
    # we specify that we must cover all cell_lines at least once, though additional coverage
    # is acceptable if it still achieves the minimization of the obj funct
    for cell_line in all_cell_lines_effected:
        ILP_model += (
            pulp.lpSum(
                [x[drug_cover] for drug_cover in drug_to_cell_line_effect_set if cell_line in drug_cover]
            ) >= 1, "Drug_must_cover_%s" % cell_line,)
        
    # now we add an additional set of constraints
    # we specify the maximum number of rows (drugs) we can use to satisfy the obj function
    # i.e. the max number of drugs we can use to cover all the cell lines
    # the maxima here is from the size of the set we need to cover, 
    # so if there are 5 cell lines, the max number of drugs used is 5
    ILP_model += (
        pulp.lpSum(
            [x[drug_cover] for drug_cover in drug_to_cell_line_effect_set]) 
        <= len(all_cell_lines_effected),"Maximum_number_of_drugs",)
    
    # return the unsolved model with obj funct and all constraints
    # as well as x, i.e. the decision variables, and the drug_to_cell_line_effect_set 
    return ILP_model, x, drug_to_cell_line_effect_set

            
# translate a class of drugs' coverage to the set of drugs that have this same coverage of the cell lines
def get_set_to_row_map(col_names,file_name):
    # this is a more pure import of the data, retaining the drug name col 
    # while still keeping the col_names separate so np's typing can be consistent 
    lookup_table = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)
    
    # first we fill the map with all the possible drug_covers we found, 
    # initializing the value for each key in the map as an empty list
    row_map = dict()
    for i in lookup_table:
        key = bin_vect_to_set(i[1:])
        row_map[key] = []
        
    # now we fill the values of the map, where we match all the drug numbers from the data
    # with their drug_covers (the set of cell lines a class of drugs covers)
    for j in lookup_table:
        key = bin_vect_to_set(j[1:])
        row_map[key].append(int(j[0]))
        
    # return the map we've built
    return row_map
    
# find all rows that match the set profile given from the solved model
def possible_drug_combos(x, drug_to_cell_line_effect_set, set_to_row_map):
    # for all the drug_covers 
    used_drugs = dict()
    for drug_cover in drug_to_cell_line_effect_set:
        # if a drug_cover is used in the solved model, return the cell lines it covers
        # as well as all the drug numbers that possess that cover
        if x[drug_cover].value() == 1.0:
            used_drugs[drug_cover] = set_to_row_map[drug_cover]

    return used_drugs

# build a final higher wrapper

def MHS_ILP(file_name):
    col_names = get_cols(file_name)
    ILP_model, x, drug_to_cell_line_effect_set = define_model(col_names, file_name)
    set_to_row_map = (get_set_to_row_map(col_names,file_name))
    ILP_model.solve()
    used_drugs = possible_drug_combos(x, drug_to_cell_line_effect_set, set_to_row_map)
    return used_drugs

def Run_ILP(df, dataDir):
    #file_name = "renal_zero_one_matrix.csv"
    # run the ILP
    (MHS_ILP(file_name))
    # use the dict to return the true drug names from the drug cover -> drug number map
    # Input: a list of NSC IDs, Output: a list of Drug Names
    MatchIdName(IDs, dataDir)
    # plot the size of the MHS generated by the ILP
    # Input: a list of the hitting sets sizes, the directory to be saved in, algorithm used (just a str)
    # Output: save the bar plot in the directory "plots"
    PlotBar(HS_sizes, plotDir, algo)
    


if __name__ == "__main__":
    main()