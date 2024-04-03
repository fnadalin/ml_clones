# NEW:
# 1. take the hvgs as input
# 2. split the clones present at day 2 in train and test
# 3. compute PCA on the merged sub-matrices, containing train and test on the hvgs
# 4. run 1-3 on a single split, multiple C, to select the best C (hvg or pca)
# 5. run 1-3 on 100 splits on the C giving the best prediction (hvg or pca)

from optparse import OptionParser
import os
import sys


###################
# DEFAULT VALUES
###################

CONT_OR_BOOL = "cont" # continuous
LOG_EXP_UB = 1
LOG_EXP_LB = 5
NUM_TESTS = 100
HVG = ""
SOLVER = "lbfgs"
PENALTY="l2"
L1_RATIO=0.5
MODE = "hvg"
NUM_HVG = 2000
HVG_MODE = "seurat"
NUM_PC = 50
DIR = "out"
SUFFIX = ""
TEST_FRAC = 0.25


###################
# PARSE OPTIONS
###################

usage = "python %prog [options] <matrix_dir> <clone_info>\n"
arg1 = "\nmatrix_dir: directory containing matrix.mtx, genes.tsv, barcodes.tsv"
arg2 = "\nclone_info: tsv file containing clone expansion information: \"cell\", \"clone\", \"clone_count\""
print("")
parser = OptionParser(usage=usage + arg1 + arg2)

parser.add_option("--log-exp-ub",	action="store", type="float", dest="log_exp_ub", 
					help="expansion rate upper bound for non-expanding clones [default: " + str(LOG_EXP_UB) + "]",
					default = LOG_EXP_UB)
parser.add_option("--log-exp-lb",	action="store", type="float", dest="log_exp_lb", 
					help="expansion rate lower bound for expanding clones [default: " + str(LOG_EXP_LB) + "]",
					default = LOG_EXP_LB)
parser.add_option("--bool",		action="store_true", dest = "bool",
					help="interpret values associated to clones as boolean [surv/non-surv; default = FALSE]")
parser.add_option("--genes",		action="store", type="string", dest="genes", 
					help="list of genes to use [default: compute hvgs on each train set]")
parser.add_option("--pca",		action="store_true", dest="pca", 
					help="apply PCA before training [default: " + str(False) + "]")
parser.add_option("--num-hvg",		action="store", type="int", dest="num_hvg", 
					help="number of hvgs to compute [only if --genes is NOT provided; default: " + str(NUM_HVG) + "]",
					default = NUM_HVG)
parser.add_option("-m", "--hvg-mode",   action="store", type="string", dest="hvg_mode", 
					help="hvg algorithm [only if --genes is NOT provided; default: " + str(HVG_MODE) + "]",
					default = HVG_MODE)
parser.add_option("-p", "--num-pc",	action="store", type="int", dest="num_pc", 
					help="number of PCs [only if --pca is set; default: " + str(NUM_PC) + "]",
					default = NUM_PC)
parser.add_option("-t", "--num-tests",  action="store", type="int", dest="num_tests", 
					help="number of train-test splits [default: " + str(NUM_TESTS) + "]",
					default = NUM_TESTS)
parser.add_option("-f", "--test-frac",  action="store", type="float", dest="test_frac", 
					help="fraction of dataset to be included in the test split [default: " + str(TEST_FRAC) + "]",
					default = TEST_FRAC)
parser.add_option("--penalty",		action="store", type="string", dest="penalty",	
					help="penalty for the logistic regression [l1, l2, elasticnet; default: " + PENALTY + "]",
					default = L1_RATIO)
parser.add_option("--l1-ratio",		action="store", type="float", dest="l1_ratio", 
					help="only used for elastic net penalty [default: " + str(L1_RATIO) + "]",
					default = L1_RATIO)
parser.add_option("-o", "--out",	action="store", type="string", dest="out", 
					help="output directory [default: " + DIR + "]",
					default = DIR)
parser.add_option("-s", "--suffix",	action="store", type="string", dest="suffix",
					help="suffix for output files [default: None]")

(options, args) = parser.parse_args()

if len(args) < 2:
	sys.stderr.write("ERROR: <matrix_dir> and <clone_info> are required\nTry --help for help\n")
	exit()
MATRIX_DIR=args[0]
CLONE_INFO=args[1]


if not (options.genes is None) :
	sys.stderr.write("WARNING: --genes overrides --hvg-mode and --num-hvg\n")
if (not (options.num_pc is None)) & (not options.pca):
	sys.stderr.write("ERROR: --num_pc requires --pca\nTry --help for help\n")
	exit()

if options.log_exp_ub:
	LOG_EXP_UB=options.log_exp_ub
if options.log_exp_lb:
	LOG_EXP_LB=options.log_exp_lb
if options.bool:
	CONT_OR_BOOL="bool"
if options.genes:
	HVG=options.genes
if options.pca:
	MODE="pca"
if options.num_hvg:
	NUM_HVG=options.num_hvg
if options.hvg_mode:
	HVG_MODE=options.hvg_mode
if options.num_pc:
	NUM_PC=options.num_pc
if options.num_tests:
	NUM_TESTS=options.num_tests
if options.penalty:
	PENALTY=options.penalty
if options.l1_ratio:
	L1_RATIO=options.l1_ratio
if options.out:
	DIR=options.out
if options.suffix:
	SUFFIX=options.suffix

if PENALTY!="l2":
	SOLVER="saga"


###################
# PREPROCESSING
###################

import scanpy as sc
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import sparse
import pickle

# CUSTOM FUNCTIONS
import ml_func

if not os.path.exists(DIR):
	os.mkdir(DIR)

filename = DIR + "/params" + SUFFIX + ".txt"
out = open(filename, "w")
out.write(str(args) + "\n")
out.write(str(options) + "\n")
out.close()

C_ARRAY = [1,0.5,0.1,0.05,0.01,0.005,0.001] # inverse of the regularization weight
MAX_ITER = 50000


###################
# PROCESSING
###################

# load the matrix and the annotations
# N.B.: use 10x reader here because the matrix must be transposed
M = sc.read(MATRIX_DIR + "/matrix.mtx") # or scanpy.read_mtx
genes = pd.read_csv(MATRIX_DIR + "/genes.tsv", header = None)
barcodes = pd.read_csv(MATRIX_DIR + "/barcodes.tsv", header = None)
M.var = genes
M.var.rename(columns = {0 : "feature"}, inplace = True)
anno = pd.read_csv(CLONE_INFO, sep = "\t")

if HVG != "":
	hvg_list = pd.read_csv(HVG, header = None)
else:
	hvg_list = genes

# match cells with barcodes in the matrix
a_bool = np.isin(anno["cell"], barcodes)
a_idx = np.where(a_bool)[0] # annotated cells that are also in the matrix
clone_idx = [] # index of common cells in the matrix
for i in range(anno.shape[0]):
	if anno["cell"][i] in barcodes.values:
		c_idx = barcodes[barcodes[0] == anno["cell"][i]].index[0]
		clone_idx.append(c_idx)
ncells = barcodes.shape[0]
clone_idx = np.asarray(clone_idx)
if CONT_OR_BOOL == "bool":
	data_dict = {"cell" : barcodes[0].values, "clone" : ['nan'] * ncells, "clone_name" : [""] * ncells, "is_surv_union" : [0] * ncells} 
else:
	data_dict = {"cell" : barcodes[0].values, "clone" : ['nan'] * ncells, "clone_name" : [""] * ncells, "clone_count" : [0] * ncells}
M.obs = pd.DataFrame.from_dict(data_dict)
a2 = anno.iloc[a_idx,:]
idx = range(anno.iloc[a_idx,:].shape[0])
dict_idx = {a2.index.values[x] : x for x in idx}
a2.rename(dict_idx,inplace=True)
M.obs.iloc[clone_idx,:] = a2
# print("M.obs: ", M.obs)
# print("M.X: ", M.X)
# print("M.X.shape: ", M.X.shape)

if CONT_OR_BOOL == "cont":
	clone_ab = np.array(M.obs["clone_count"]) # clone-wise 
	idx = (clone_ab <= LOG_EXP_UB) | (clone_ab > LOG_EXP_LB) 
	clone_ab = clone_ab[idx]
	clone_ab_class = (clone_ab > LOG_EXP_LB).astype(int) # bool: expanded (1) or not (0) 
	out = open(DIR + "/clone_class.txt", "w")
	np.savetxt(out, clone_ab_class)
	out.close()

clone_unique = list(set(M.obs["clone"])) # unique occurrence of clone names
nclones = len(clone_unique)

# subset on cells and genes
hvg_bool = np.isin(M.var.values, hvg_list)
hvg_idx = np.where(hvg_bool)[0]
if CONT_OR_BOOL == "bool":
	M_sub = M[:,hvg_idx]
if CONT_OR_BOOL == "cont":
	M_sub = M[idx,hvg_idx] # this is a view on anndata object: need to create the new object explicitly
M = sc.AnnData(
        X = M_sub.X,
        obs = M_sub.obs,
        var = M_sub.var
        )

# select the best regularization inverse weight 
best_acc = 0
best_C = 1.0
for C in C_ARRAY:
	# logistic regression
	if (PENALTY == "elasticnet") :
		pipe = make_pipeline(StandardScaler(with_mean = False), 
		        linear_model.LogisticRegression(max_iter = MAX_ITER, penalty = PENALTY, l1_ratio = L1_RATIO, C = C, solver = SOLVER))
	else :
		pipe = make_pipeline(StandardScaler(with_mean = False),
		        linear_model.LogisticRegression(max_iter = MAX_ITER, penalty = PENALTY, C = C, solver = SOLVER))
	if CONT_OR_BOOL == "bool":
		acc, tp, fn, tn, fp = runTrainTestBool(M, clone_unique, 0, pipe, hvg_list, HVG_MODE, NUM_HVG, MODE)
	else:
		acc, tp, fn, tn, fp = runTrainTest(M, clone_unique, 0, pipe, LOG_EXP_LB, hvg_list, HVG_MODE, NUM_HVG, MODE)
	if acc > best_acc:
		best_acc = acc
		best_C = best_C

filename = DIR + "/best_param" + SUFFIX + ".txt"
out = open(filename, "w")
out.write("C=" + str(best_C) + "\n")
out.close()

# run 100 random train/test splits
acc_array = []
tp_array = []
fn_array = []
tn_array = []
fp_array = []
if (PENALTY == "elasticnet") :
	pipe = make_pipeline(StandardScaler(with_mean = False),
		linear_model.LogisticRegression(max_iter = MAX_ITER, penalty = PENALTY, l1_ratio = L1_RATIO, C = best_C, solver = SOLVER))
else :
	pipe = make_pipeline(StandardScaler(with_mean = False),
		linear_model.LogisticRegression(max_iter = MAX_ITER, penalty = PENALTY, C = best_C, solver = SOLVER))
for r in range(NUM_TESTS):
	if CONT_OR_BOOL == "bool":
		acc, tp, fn, tn, fp = runTrainTestBool(M, clone_unique, r+1, pipe, hvg_list, HVG_MODE, NUM_HVG, MODE)
	else:
		acc, tp, fn, tn, fp = runTrainTest(M, clone_unique, r+1, pipe, LOG_EXP_LB, hvg_list, HVG_MODE, NUM_HVG, MODE)
	acc_array.append(acc)
	tp_array.append(tp)
	fn_array.append(fn)
	tn_array.append(tn)
	fp_array.append(fp)


###################
# PRINT OUT
###################

acc_array = np.array(acc_array)
filename = DIR + "/balanced_accuracy" + SUFFIX + ".txt"
out = open(filename, "w")
np.savetxt(out, acc_array)
out.close()

tp_array = np.array(tp_array)
filename = DIR + "/TP" + SUFFIX + ".txt"
out = open(filename, "w")
np.savetxt(out, tp_array)
out.close()

fn_array = np.array(fn_array)
filename = DIR + "/FN" + SUFFIX + ".txt"
out = open(filename, "w")
np.savetxt(out, fn_array)
out.close()

tn_array = np.array(tn_array)
filename = DIR + "/TN" + SUFFIX + ".txt"
out = open(filename, "w")
np.savetxt(out, tn_array)
out.close()

fp_array = np.array(fp_array)
filename = DIR + "/FP" + SUFFIX + ".txt"
out = open(filename, "w")
np.savetxt(out, fp_array)
out.close()



exit()


