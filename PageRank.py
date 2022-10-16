import scipy.sparse as sparse
import numpy as np
import glob as glob
import warnings
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from scipy.sparse import csr_matrix
import timeit
#from scipy.sparse.linalg import norm

# I want this to at least be ran before you go to bed tonight so you can 
# start debugging tomorrow and have something to talk about
def rank_scores():
    return 

class page_rank:
    def __init__(self, transition_matrix, alpha):
        # p_not is uniform dist over the space, report as m x 1 np arr
        self.r = np.ones((transition_matrix.shape[0], 1)) / transition_matrix.shape[0]
        # m x m np arr
        self.M = transition_matrix # needs to be normalized 
        # p_not is uniform dist over the space, report as m x 1 np arr
        self.p_not = np.ones((transition_matrix.shape[0], 1)) / transition_matrix.shape[0]
        self.alpha = alpha
        
    # just make a vector that has 1 for every row that has all zeros in the transition matrix and multiply it by r
    # what is a 
    def one_step(self):
        # convergence criteria here -- no :)
        # any time M is multiplied it needs to be sparse
        # M_s = boolean true where connection exists
        # M_c boolean false where connection exists
        # print(self.r.sum())
        # this need to be modified to take into accont zero rows of the matrix
        # make some sort of lin alg op here
        zero_rows =(~self.M.sum(axis=1).astype(bool)).astype(int)
        #print(zero_rows.shape)
        #print(self.r.shape)
        #print(zero_rows)
        #print(self.r)
        first_term = (1 - self.alpha) * self.M.T.dot(self.r)
        second_term = (1 - self.alpha)*np.multiply(zero_rows, self.r)
        third_term = self.alpha * self.p_not
        self.r = first_term + second_term + third_term
        #print(self.r.sum())
        #raise exception
    
    # # not highest in magnitude, most positive on top
    def get_WS(self, relevance_scores, weights): # do need indri lists -- not a dot product! some weighted sum of indri list vect and r vect,
        # output is also of size r_vect, elementwise product
        start = timeit.timeit()
        
        new_scores = np.zeros(len(relevance_scores))
        for i in range(len(relevance_scores)):
            if relevance_scores[i] != 0:
                new_scores[i] = relevance_scores[i]*weights[0] + self.r[i]*weights[1]
            else:
                new_scores[i] = 0 
        # rank here? -- need to time the ranking as well
        df = pd.DataFrame(new_scores, columns = ['Score'])
        df = df.sort_values(by = 'Score', ascending=False)
        # now I must specify some weighting -- do I need the retrieval time to take into account
        # the reranking?
        end = timeit.timeit()
        return df, (end - start)
    
    def get_CM(self, relevance_scores): # some other weighting schema would be more appropriate 
        # perhaps take an elementwise maximum? -- use np.max with the correct axis
        # you must ignore those that are nonzero in the relevance scores, that is what the assignment says you can ignore
        # otherwise your input to trec eval will not be correct. Perhaps a simple ls would be better? -- no you need to keep the docids somehow
        start = timeit.timeit()
        new_scores = np.zeros(len(relevance_scores))
        # just take the max of rs and return as a one hot-ish vector and take the dot product
        # column stack
        for i in range(len(relevance_scores)):
            if relevance_scores[i] != 0:
                new_scores[i] = max(-relevance_scores[i], 10000 *self.r[i])
            else:
                new_scores[i] = 0 
                
        df = pd.DataFrame(new_scores, columns = ['Score'])
        df = df.sort_values(by = 'Score', ascending=True)
        end = timeit.timeit()
        return df, (end - start)
    
    def get_NS(self, relevance_scores): # rank only based on the pr scores, sort based on r alone?
        start = timeit.timeit()
        new_scores = np.zeros(len(self.r))
        # just take the max of rs and return as a one hot-ish vector and take the dot product
        # column stack
        for i in range(len(self.r)):
            if relevance_scores[i] != 0:
                new_scores[i] = self.r[i]
            else:
                new_scores[i] = 0 
        df = pd.DataFrame(new_scores, columns = ['Score'])
        df = df.sort_values(by = 'Score', ascending=True)
        end = timeit.timeit()
        return df, (end - start) #prob_dist.dot(self.r)
    
    def retrieve_scores(self, relevance_scores, weights):
        WS_scores, WS_timings = self.get_WS(relevance_scores, weights)
        CM_scores, CM_timings = self.get_CM(relevance_scores)
        NS_scores, NS_timings = self.get_NS(relevance_scores)
        all_scores = [WS_scores, CM_scores, NS_scores]
        # rank the docs and discard those with zero entries -- this is handled by the prep for export function
        all_timings = [WS_timings, CM_timings, NS_timings]
        return all_scores, all_timings
    
# this must be done for each topic!
class personalized_PR(page_rank):
    # overwrite w super the constructor to retrieve the additional info
    # we grab the init distribution from query-topic-distro.txt for each document, for all topics in it
    def __init__(self, transition_matrix, alpha, beta, gamma, user_prob):
        self.r = np.ones((transition_matrix.shape[0], 1)) / transition_matrix.shape[0]
        # m x m np arr
        self.M = transition_matrix
        self.p_t = user_prob
        self.p_not = np.ones((transition_matrix.shape[0], 1)) / transition_matrix.shape[0]
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
    def one_step(self):
        #print(self.r.sum())
        # remember that r = B @ r
        # these will need to be modified for sparse operations
        zero_rows =(~self.M.sum(axis=1).astype(bool)).astype(int)
        # first term calc is costly!
        sparse_dot_product = self.alpha * self.M.T.dot(self.r) 
        # two of these ebing intrpreted as matrices, not vectors, this slows the process enormously
        first_term = sparse_dot_product + self.alpha*np.multiply(zero_rows, self.r) # pass it in as a matrix?
        #print(first_term.shape, "one")
        second_term = (self.beta * self.p_t) # is this slow??
        #print(second_term.shape, "two")
        third_term = (self.gamma * self.p_not) 
        #print(third_term.shape, "three")
        # summing the vectors is costly?
        self.r = np.add(np.add(first_term, second_term), third_term)
        #print(self.r.sum())
        # r is of len document id!
        assert self.r.shape[0] == 81433  
# there is a problem with memory --still an issue!     
            
# for the (u,q) pairs missing in the two files, we treat the p as uniform across all topics     
# now finish this, then check that the grid search function works -- this is done!
def TSPR(tp_dict, transition_matrix, alpha, beta, gamma, conv_criteria, relevance_scores, weight_pair, pr_type, s_types): # need topics, prob dict
    # really just seems like a for loop over personalized_PR
    convergence_times = []
    all_dfs, all_times = [[], [], []], [] # [] look at make time df
    for i in tp_dict.keys():
        #print(i)
        # keeping this many objects in memory seems to be the problem
        # so I need to do everything I need to do and dump the prs before I move onto the next
        topic_SPR = personalized_PR(transition_matrix, alpha, beta, gamma, tp_dict[i])
        # keeping track of the whole PPR might be a problem
        convergence_time = run_till_conv(topic_SPR, conv_criteria) # it's a problem here
        #print(convergence_time, beta, gamma, weight_pair) # the convergence is fast!
        scorings, timings = topic_SPR.retrieve_scores(relevance_scores[i], weight_pair)
        convergence_times.append(convergence_time)
        all_times.append(timings)
        # they want the averaged run time across queries for PPRs 
        for j in range(len(s_types)):
            #print(s_types[j])
            # uq_pair, score, s_type, pr_type
            df = (prepare_for_export(i, scorings[j], s_types[j], pr_type))
            all_dfs[j].append(df)
            
        if i == (2, 1):
            make_autograder_ouputs(pr_type, topic_SPR)
    extra_name = str(beta) + "_" + str(gamma)
    print(np.mean(convergence_times), pr_type, beta, gamma, weight_pair)
    # must be averaged across all queries
    make_time_df(pr_type, weight_pair, all_times, s_types)
    # is this all that is reuired or does i
    make_trec_eval_outputs(all_dfs, pr_type, s_types, weight_pair, extra_name)
    

# trec_eval will report my MAP and precison at N, I only need to record what it tells me
# GPR and TSPR will be be of the same len input to trec eval it seems, as we will have different rel scores for differe u,q pairs,
# even though the GPR result will be the same.

def l_one_norm_m(M): # not a sparse matrix, just for a dense vector
    return np.linalg.norm(M, 1)
        
def run_till_conv(pr_instance, conv_criteria):
    start = timeit.timeit()
    # record init r
    prev_r = pr_instance.r
    # immediately take one step
    pr_instance.one_step()
    current_r = pr_instance.r
    while l_one_norm_m(current_r - prev_r) >= conv_criteria:
        # update prev_r, current_r
        prev_r = pr_instance.r
        pr_instance.one_step()
        current_r = pr_instance.r
        #print(current_r.sum())
    end = timeit.timeit()
    return end - start

# MUST IMPORT AS SPARSE MATRIX, use scipy sparse
# normalize by row before importing into spare matrix format
def import_trans_mat(trans_path):
    # load in as np float arr
    pre_op_trans = np.loadtxt(trans_path, delimiter=" ", dtype='int')
    N = (np.max(pre_op_trans)) # 81433 is n -- pretty sure it has to be square
    # row, col, value     csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
    # where data, row_ind and col_ind satisfy the relationship a[row_ind[k], col_ind[k]] = data[k].
    # the normalization is sus as fuck
    post_op_trans = csr_matrix((pre_op_trans[:, -1], (pre_op_trans[:, 0]-1, pre_op_trans[:, 1]-1)), shape=(N, N))
    # magick
    post_op_trans.data = post_op_trans.data / np.repeat(np.add.reduceat(post_op_trans.data, post_op_trans.indptr[:-1]), np.diff(post_op_trans.indptr))
    return post_op_trans

def import_topics_for_docs(topic_docs_path):
    # The document classification information for TSPR algorithms is stored in doc-topics.txt. Each row is a docid-topicid pair
    return np.loadtxt(topic_docs_path, delimiter=" ", dtype='int')

def import_user_topics(user_topic_path):
    # User’s topical interest distribution Pr(t|ua,b) for all topics for PTSPR method is stored in user-topic-distro.txt,
    # a b 1 : p1 2 : p2 . . . . . . 12 : p12
    # where a is the user id, b represents the b-th query by the user, and the items of the form t:pt denote topic probabilities Pr(t|ua,b)
    tuple_ls = dict()
    with open(user_topic_path) as f:
        lines = f.readlines()
    for line in lines:
        split_line = (line[:-1].split(" "))
        # make it a dictionary: int tuple (user id,  b-th query by the user) : [p1 ... p12] float arr
        tuple_ls[(int(split_line[0]), int(split_line[1]))] = np.array([float(split_line[i+2].split(":")[1]) for i in range(len(split_line) - 2)])
        
    return tuple_ls


# For each query, you only need to include the documents that are present in the provided searchrelevance list, 
# documents that do not appear in the search-relevance list can be ignored.
def import_query_topics(query_topic_path):
    # Query’s topical distribution Pr(t|qa,b) for all topics for QTSPR method is stored in query-topic-distro.txt,
    # a b 1 : p12 : p2 . . . . . . 12 : p12
    # where a is the user id, b represents the b-th query by the user, and the items of the form t:pt denote topic probabilities Pr(t|qa,b)
    tuple_ls = dict()
    with open(user_topic_path) as f:
        lines = f.readlines()
    for line in lines:
        split_line = (line[:-1].split(" "))
        # make it a dictionary: int tuple (user id,  b-th query by the user) : [p1 ... p12] float arr
        tuple_ls[(int(split_line[0]), int(split_line[1]))] = np.array([float(split_line[i+2].split(":")[1]) for i in range(len(split_line) - 2)])
    return tuple_ls

def import_rel_scores(indri_list_path, N):
    # inside the indri-list dir, QTSPR-U2Q1.txt
    # this file is the final values of the pagerank algorithm 
    # trained on the user with user_id=2 on query=1 from the query-topic-distro.txt file? We can ignore all the other lines from that file
    rel_scores = glob.glob(indri_list_path)
    tuple_ls = dict()
    all_lines = 0
    for file in rel_scores:
        # ../data/indri-lists/14-1.results.txt
        user = int(file.split("/")[-1].split("-")[0])
        query = int(file.split("/")[-1].split("-")[1].split(".")[0])
        df = pd.read_csv(file, sep=" ", header=None)
        row_index = df[2].to_numpy(int) - 1
        rank = df[3].to_numpy(float)
        score = df[4].to_numpy(float)
        rel_vect = np.zeros((N, 1)) # needs to be nx1 for the math to work out
        for i in range(len(row_index)):
            #rel_vect[row_index[i], 0] = rank[i]
            rel_vect[row_index[i]] = score[i]
        all_lines += len(df.index) # exactly as it should be  -- pretty sure I'm misunderstanding the relevance scores
        tuple_ls[(user, query)] = rel_vect
    #print(all_lines, "shloop")
    return tuple_ls

#     QueryID Q0 DocID Rank Score RunID -- output format -- why do I need trec eval format? oh! only for the queries I'm doing!
#     The QueryID should correspond to the query ID of the query you are evaluating. Q0 is a required constant.
#     The DocID should be the external document ID. The Score should be in descending order, to indicate that
#     your results are ranked. The RunID is an experiment identifier which can be set to anything.
def prepare_for_export(uq_pair, df, s_type, pr_type):
    
    q_id = str(uq_pair[0]) + "-" + str(uq_pair[1])
    df["QueryID"] = q_id
    df["Q0"] = "Q0"
    df["DocID"] = df.index + 1
    df["RunID"] = pr_type + "_" + s_type
    #df.to_csv(pr_type + "_" + s_type, sep=' ') -- then it needs to be subset based for which  rows there are not zeros
    df = df[df['Score'] != 0]
    # reindex the df
    df["Rank"] = [i+1 for i in range(len(df.index))]
    df = df[["QueryID", "Q0", "DocID", "Rank", "Score", "RunID"]]
    #print(df, "moop")
    # For each query, you only need to include the documents that are present in the provided searchrelevance list 
    # -- those that are not should have 0 scores in the vector right? -- stop and understand this more
    # documents that do not appear in the search-relevance list can be ignored.
    return df

def make_time_df(pr_type, weight_pair, all_times, s_types):
    # do not forget the init GPR convergence runtime
    # ex: GPR-WS: x secs for PageRank, y secs for retrieval -- now the other output 
    #     GPR.txt : Final converged GPR values -- just the r vector ouput I assume
    df = pd.DataFrame(all_times, columns=s_types)
    df = df.mean(axis=0) 
    # I assume this is a column wise mean
    df.to_csv(pr_type+"_"+str(weight_pair[0])+"_" +str(weight_pair[1])+"_timings.txt", sep=' ')
    
def make_autograder_ouputs(pr_type, pr_instance):
    # spit out pr_instance.r -- documentID PageRankValue
    df = pd.DataFrame(pr_instance.r, columns = ['PageRankValue'])
    df["documentID"] = df.index +1 
    df = df[["documentID", 'PageRankValue']]
    if pr_type == "GPR":
        df.to_csv("GPR.txt", sep=' ', index=False)
    else:
        # -U2Q1
        name = pr_type +"-U2Q1" + ".txt"
        df.to_csv(name, sep=' ', index=False)

def make_trec_eval_outputs(all_dfs, pr_type, s_types, weight_pair, extra_name):
    for i in range(len(s_types)):
        whole_df = pd.concat(all_dfs[i], axis=0, ignore_index=True)
        #print(whole_df.index)
        whole_df.to_csv(pr_type+"_"+str(weight_pair[0])+"_" +str(weight_pair[1])+"_"+s_types[i]+"_"+extra_name+".txt", sep=' ',index=False)


    
# ignore this for now, you must have the PR and full PPR results go through trec eval before you go to office hours
# AFTER you know these things are working appropriately, then you can finish grid search 
# AND then after that you should write the report
# unit test that your indri-list stuff is working, and how to apply it to the GPR r vct
def grid_search_for_pr(pr_type, args, relevance_scores, conv_criteria):
    s_types = ["WS", "CM", "NS"]
    if pr_type == "GPR":
        transition_matrix = args[0]
        weights = args[1]
        alpha = 0.2
        pr_instance = page_rank(transition_matrix, alpha)
        convergence_time = run_till_conv(pr_instance, conv_criteria)
        print(convergence_time, "GPR")
        # now do WS, CM, NS to get rankings -- need relevance_scores
        for weight_pair in weights:
            all_dfs, all_times = [[], [], []], []
            for key in relevance_scores:
                scorings, timings = pr_instance.retrieve_scores(relevance_scores[key], weight_pair)
                all_times.append(timings)
                for i in range(len(s_types)):
                    # uq_pair, score, s_type, pr_type
                    df = (prepare_for_export(key, scorings[i], s_types[i], "GPR"))
                    all_dfs[i].append(df)
            #  modularize this for use in the topic specific one
            extra_name = ""
            make_time_df(pr_type, weight_pair, all_times, s_types)
            make_autograder_ouputs(pr_type, pr_instance)
            make_trec_eval_outputs(all_dfs, pr_type, s_types, weight_pair, extra_name)
        
    # I would come back to this, but don't write anythin else that does all these runs, just change the beta and gamma input
    else: # ow it will be TPSR so we need to wrap TSPR!
        # transition_matrix, tp_dict, beta, gamma, conv_criteria
        transition_matrix, weights, tp_dict, betas, gammas, conv_criteria = args
        alpha = 0.8 
        for i in range(len(betas)):
            for weight_pair in weights:
                # gets it for all topics at once
                TSPR(tp_dict, transition_matrix, alpha, betas[i], gammas[i], conv_criteria, relevance_scores, weight_pair, pr_type, s_types) 

def map_dt_pairs_to_dist(docid_topicid_pairs, query_topic_distro):
    tuple_dict = dict()
    # need to make tuple dict with topic-specific teleportation vector as vals
    for key in query_topic_distro.keys():
        topic_dist = query_topic_distro[key]
        #print(topic_dist)
        #print(sum(topic_dist))
        topic_indices = [i+1 for i in range(len(topic_dist))]
        #print(topic_indices)
        topic_lists = [[] for i in topic_dist]
        for i in range(docid_topicid_pairs.shape[0]):
            # match on topic id
            #print((docid_topicid_pairs[i, 1]))
            topic_lists[topic_indices.index(docid_topicid_pairs[i, 1])].append(docid_topicid_pairs[i, 0] - 1)
        N = (max(docid_topicid_pairs[:, 0]))
        # some documents have multiple topics! -- it will be slightly more complicated, you told me you would gym today -- 10 more min
        full_vect = np.zeros((N,1)) # this should be a a vector right? 
        for j in range(len(topic_dist)):
            nornalized_dist = topic_dist[j] / len(topic_lists[j])
            for k in topic_lists[j]:
                full_vect[k] += nornalized_dist
                
        tuple_dict[key] = full_vect
    return tuple_dict      
                
def main():
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    # need command line args, need to know struct of input -- do you?
    transition_matrix = import_trans_mat("../data/transition.txt")
    #print(transition_matrix.sum(axis=1))
     # 0.8 for the topic specific prs
    #betas = [0.2, 0.19, 0.1, 0.01, 0]
    #gammas = [0, 0.001, 0.1, 0.19, 0.2]
    #weights = [[0, 1], [0.1, 0.9], [0.5, 0.5], [0.9, 0.1], [1,0]]
    betas = [0.1]
    gammas = [0.1]
    weights = [[0.5, 0.5]]
    conv_criteria = 10**(-8) # [10**(-6), 10**(-7), 10**(-8), 10**(-9), 10**(-10)]
    user_topic_distro = import_user_topics("../data/user-topic-distro.txt")
    #print(user_topic_distro)
    query_topic_distro = import_user_topics("../data/query-topic-distro.txt")
    docid_topicid_pairs = import_topics_for_docs("../data/doc_topics.txt")
    #print(query_topic_distro)
    N = transition_matrix.shape[0]
    rel_scores = (import_rel_scores("../data/indri-lists/*", N))
    pr_type = "GPR"
    args = [transition_matrix, weights]
    grid_search_for_pr(pr_type, args, rel_scores, conv_criteria)
    pr_type = "QTSPR"
    tp_dict = map_dt_pairs_to_dist(docid_topicid_pairs, query_topic_distro)
    args = [transition_matrix, weights, tp_dict, betas, gammas, conv_criteria]
    grid_search_for_pr(pr_type, args, rel_scores, conv_criteria)
    pr_type = "PTSPR" # must recalc all of these now!
    tp_dict = map_dt_pairs_to_dist(docid_topicid_pairs, user_topic_distro)
    args = [transition_matrix, weights, tp_dict, betas, gammas, conv_criteria]
    grid_search_for_pr(pr_type, args, rel_scores, conv_criteria)


if __name__ == '__main__':
    main()