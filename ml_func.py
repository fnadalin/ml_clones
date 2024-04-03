
###################
# BEGIN FUN
###################

def runTrainTestLogit(M, clone_unique, r, C, mode):

	ncells = M.X.get_shape()[0]
	ngenes = M.X.get_shape()[1]

	# learning pre-processing
	cl_train, cl_test = train_test_split(clone_unique, random_state = r)

	cl_train_cell = []
	for x in range(ncells):
		if M.obs["clone"].array[x] in cl_train:
			cl_train_cell.append(x)

	cl_test_cell = []
	for x in range(ncells):
		if M.obs["clone"].array[x] in cl_test:
			cl_test_cell.append(x)

	# compute highly variable genes on day 2 and subset the matrix
	# N.B.: use seurat v2 because seurat v3 requires count data (non-normalized)
	X = M[cl_train_cell,:]
	# set subset=True to subset X columns on the hvgs
	sc.pp.highly_variable_genes(X, flavor=HVG_METHOD, n_top_genes = N_TOP_HVG, subset=True, inplace=True)
	hvg = X.var[0].array

	# subset the test matrix
	all_genes = M.var[0].array
	hvg_idx = []
	for x in range(ngenes):
		if all_genes[x] in hvg:
			hvg_idx.append(x)

	Z = M[cl_test_cell,hvg_idx]

	ncells_train = X.X.get_shape()[0]
	ncells_test = Z.X.get_shape()[0]

	# link clone abundance at day 4 & 6 to day 2, cell-wise
	b_train = []
	for x in range(ncells_train):
		b_train.append(clone_ab[X.obs["clone"].array[x]])

	b_test = []
	for x in range(ncells_test):
		b_test.append(clone_ab[Z.obs["clone"].array[x]])

	y = np.array(b_train)
	y_bool = (y > LOG_EXP).astype(int) # bool: expanded (1) or not (0)

	w = np.array(b_test)
	w_bool = (w > LOG_EXP).astype(int) # bool: expanded (1) or not (0)

	# logistic regression
	pipe_lr = make_pipeline(StandardScaler(with_mean = False),
				linear_model.LogisticRegression(max_iter = 10000, C = C, class_weight = "balanced"))

	if mode == "hvg":
		# regress on hvg
		pipe_lr.fit(X.X, y_bool)
		coef = pipe_lr[1].coef_
		pred = pipe_lr.predict(Z.X)
		acc = accuracy_score(pred, w_bool)
		tp = sum((w_bool == 1) & (pred == 1))
		fn = sum((w_bool == 1) & (pred == 0))
		tn = sum((w_bool == 0) & (pred == 0))
		fp = sum((w_bool == 0) & (pred == 1))
	
	if mode == "pca":
		# compute pca
		Z.uns = X.uns # set the same hvg info, otherwise merging does not work
		merged = X.concatenate(Z)
		sc.pp.pca(merged, n_comps=NUM_PC)

		sparse_merged = sparse.csr_matrix(merged.obsm["X_pca"])
		Xpca = sparse_merged[range(ncells_train),]
		Zpca = sparse_merged[range(ncells_train,ncells),]

		# regress on pca
		pipe_lr.fit(Xpca, y_bool)
		coef = pipe_lr[1].coef_
		pred = pipe_lr.predict(Zpca)
		acc = accuracy_score(pred, w_bool)
		tp = sum((w_bool == 1) & (pred == 1))
		fn = sum((w_bool == 1) & (pred == 0))
		tn = sum((w_bool == 0) & (pred == 0))
		fp = sum((w_bool == 0) & (pred == 1))

	return acc, tp, fn, tn, fp, coef, hvg



###################
# BEGIN FUN
###################

def runTrainTest(M, clone_unique, r, pipe, LOG_EXP_LB, HVG, HVG_METHOD, NUM_HVG, mode):

	ncells = M.X.get_shape()[0]
	ngenes = M.X.get_shape()[1]
#	clone_ab = np.array(M.obs["clone_count"])

	# learning pre-processing
	cl_train, cl_test = train_test_split(clone_unique, random_state = r)

	cl_train_cell = []
	for x in range(ncells):
		if M.obs["clone"].array[x] in cl_train:
			cl_train_cell.append(x)

	cl_test_cell = []
	for x in range(ncells):
		if M.obs["clone"].array[x] in cl_test:
			cl_test_cell.append(x)

	X = M[cl_train_cell,:]
	if len(HVG) > 0:
		Z = M[cl_test_cell,:]
	else:
		# compute highly variable genes on day 2 and subset the matrix
		# N.B.: use seurat v2 because seurat v3 requires count data (non-normalized)
		# set subset=True to subset X columns on the hvgs
		sc.pp.highly_variable_genes(X, flavor=HVG_METHOD, n_top_genes = NUM_HVG, subset=True, inplace=True)
		hvg = X.var[0].array
		# subset the test matrix
		all_genes = M.var[0].array
		hvg_idx = []
		for x in range(ngenes):
			if all_genes[x] in hvg:
				hvg_idx.append(x)
		Z = M[cl_test_cell,hvg_idx]

	ncells_train = X.X.get_shape()[0]
	ncells_test = Z.X.get_shape()[0]

	# link clone abundance at day 4 & 6 to day 2, cell-wise
#	b_train = []
#	for x in range(ncells_train):
#		b_train.append(clone_ab[X.obs["clone"].array[x]])
	b_train = X.obs["clone_count"].array
#	b_test = []
#	for x in range(ncells_test):
#		b_test.append(clone_ab[Z.obs["clone"].array[x]])
	b_test = Z.obs["clone_count"].array
	
	y = np.array(b_train)
	y_bool = (y > LOG_EXP_LB).astype(int) # bool: expanded (1) or not (0)

	w = np.array(b_test)
	w_bool = (w > LOG_EXP_LB).astype(int) # bool: expanded (1) or not (0)

	if mode == "hvg":
		# regress on hvg
		pipe.fit(X.X, y_bool)
		pred = pipe.predict(Z.X)
		acc = balanced_accuracy_score(pred, w_bool)
		tp = sum((w_bool == 1) & (pred == 1))
		fn = sum((w_bool == 1) & (pred == 0))
		tn = sum((w_bool == 0) & (pred == 0))
		fp = sum((w_bool == 0) & (pred == 1))
	
	if mode == "pca":
		# compute pca
		Z.uns = X.uns # set the same hvg info, otherwise merging does not work
		merged = X.concatenate(Z)
		sc.pp.pca(merged, n_comps=NUM_PC)

		sparse_merged = sparse.csr_matrix(merged.obsm["X_pca"])
		Xpca = sparse_merged[range(ncells_train),]
		Zpca = sparse_merged[range(ncells_train,ncells),]

		# regress on pca
		pipe.fit(Xpca, y_bool)
		pred = pipe.predict(Zpca)
		acc = balanced_accuracy_score(pred, w_bool)
		tp = sum((w_bool == 1) & (pred == 1))
		fn = sum((w_bool == 1) & (pred == 0))
		tn = sum((w_bool == 0) & (pred == 0))
		fp = sum((w_bool == 0) & (pred == 1))

	return acc, tp, fn, tn, fp

# END FUN


def runTrainTestBool(M, clone_unique, r, pipe, HVG, HVG_METHOD, NUM_HVG, mode):

	ncells = M.X.get_shape()[0]
	ngenes = M.X.get_shape()[1]
#	clone_ab = np.array(M.obs["is_surv_union"])
	# learning pre-processing
	cl_train, cl_test = train_test_split(clone_unique, random_state = r)

	cl_train_cell = []
	for x in range(ncells):
		if M.obs["clone"].array[x] in cl_train:
			cl_train_cell.append(x)

	cl_test_cell = []
	for x in range(ncells):
		if M.obs["clone"].array[x] in cl_test:
			cl_test_cell.append(x)

	X = M[cl_train_cell,:]
	if len(HVG) > 0:
		Z = M[cl_test_cell,:]
	else:
		# compute highly variable genes on day 2 and subset the matrix
		# N.B.: use seurat v2 because seurat v3 requires count data (non-normalized)
		# set subset=True to subset X columns on the hvgs
		sc.pp.highly_variable_genes(X, flavor=HVG_METHOD, n_top_genes = NUM_HVG, subset=True, inplace=True)
		hvg = X.var[0].array
		# subset the test matrix
		all_genes = M.var[0].array
		hvg_idx = []
		for x in range(ngenes):
			if all_genes[x] in hvg:
				hvg_idx.append(x)
		Z = M[cl_test_cell,hvg_idx]

	ncells_train = X.X.get_shape()[0]
	ncells_test = Z.X.get_shape()[0]
	
	# link clone survival at day >= 13 to day 0, cell-wise
#	b_train = []
#	for x in range(ncells_train):
#		b_train.append(clone_ab[X.obs["clone"].array[x]])
	b_train = X.obs["is_surv_union"].array
#	b_test = []
#	for x in range(ncells_test):
#		b_test.append(clone_ab[Z.obs["clone"].array[x]])
	b_test = Z.obs["is_surv_union"].array
	
	y_bool = np.array(b_train).astype(int) # bool: surviving (1) or not (0)
	w_bool = np.array(b_test).astype(int) # bool: surviving (1) or not (0)

	if mode == "hvg":
		# regress on hvg
		pipe.fit(X.X, y_bool)
		pred = pipe.predict(Z.X)
		acc = balanced_accuracy_score(pred, w_bool)
		tp = sum((w_bool == 1) & (pred == 1))
		fn = sum((w_bool == 1) & (pred == 0))
		tn = sum((w_bool == 0) & (pred == 0))
		fp = sum((w_bool == 0) & (pred == 1))

	if mode == "pca":
		# compute pca
		Z.uns = X.uns # set the same hvg info, otherwise merging does not work
		merged = X.concatenate(Z)
		sc.pp.pca(merged, n_comps=NUM_PC)

		sparse_merged = sparse.csr_matrix(merged.obsm["X_pca"])
		Xpca = sparse_merged[range(ncells_train),]
		Zpca = sparse_merged[range(ncells_train,ncells),]

		# regress on pca
		pipe.fit(Xpca, y_bool)
		pred = pipe.predict(Zpca)
#		print("pred\n")
#		np.savetxt(sys.stdout, pred)
		acc = balanced_accuracy_score(pred, w_bool)
		tp = sum((w_bool == 1) & (pred == 1))
		fn = sum((w_bool == 1) & (pred == 0))
		tn = sum((w_bool == 0) & (pred == 0))
		fp = sum((w_bool == 0) & (pred == 1))

	return acc, tp, fn, tn, fp

# END FUN

