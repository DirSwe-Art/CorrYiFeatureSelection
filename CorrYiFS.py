"""A Correlation Based Feature Selection Algorithm"""

# Authors: DirarSweidan
# License: DSB 3-Claus

import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from sklearn.model_selection 	import	StratifiedKFold

def CorrYiFS(X_df, corr_method='pearson', theta_2=0.68, column_names=None):
	T_name   		= X_df.columns[-1]
	R1	  			= X_df.corr(method=corr_method)

	R1_TF_corr_abs  = np.abs(R1[T_name][:-1])
	theta_1     	= np.mean(R1_TF_corr_abs)
	
	# Exclude columns having high correlations with target
	F1   			= [X_df.columns[i] for i, v in enumerate(R1_TF_corr_abs) if abs(v) <= theta_1]
	R1 				= R1.drop(index=F1, columns=F1)
	
	
	# Exclude the column having less correlation with the target from each highly correlated pair
	R2				= R1.sort_values(by=T_name, key=abs, ascending=False).sort_values(axis=1, by=T_name, key=abs, ascending=False)
	R2_TF_corr 		= R2.iloc[0, 1:]
	R2_FF_corr 		= R2.iloc[1:, 1:]
	
	R2 = R1.copy()
	F2				= []
	for i, j in combinations(range(len(R2_FF_corr)),2):
		if R2_TF_corr.index[i] not in F2 and R2_TF_corr.index[j] not in F2 and  abs(R2_FF_corr.values[i][j]) > theta_2:
			if abs(R2_TF_corr[i]) <= abs(R2_TF_corr[j]): 
				F2.append(R2_TF_corr.index[i])
			else: 
				F2.append(R2_TF_corr.index[j])

	
	S = [R2_TF_corr.index[t_id] for t_id in range(len(R2_TF_corr)) if R2_TF_corr.index[t_id] not in set(F2)]
	#print('optimal features', len(S))
	#for f_name in S: print(f_name)
	return S

def CorrYiFSCV(X, cv=None, corr_method='pearson', theta_2=0.68, column_names=None):
	# Convert input data into a DataFrame format
	if isinstance(X, pd.DataFrame):
		df = X.copy()
		if column_names is not None:
			df.columns = column_names
	elif not isinstance(X, np.ndarray):
		if column_names is None:
			column_names = [f'Column_{i+1}' for i in range(X.shape[1])]
		df = pd.DataFrame(np.array(X), columns=column_names)

	# Drop columns having a single value
	X_df    		= df.loc[:, df.nunique() > 1].copy()
	column_names	= X_df.columns
	dropped_cols  	= set(df.columns) - set(X_df.columns)
	if dropped_cols:
		print(f'\nWarning!\nThe following columns have a \
		single value and will be temporarily dropped from \
		the data frame:\n{dropped_cols}\n')

	# Set Cross-Validation train and test sets
	target_name				= X_df.columns[-1]
	
	# Without Cross-Validation
	if cv == 0:		
		S = CorrYiFS(X_df, corr_method=corr_method, theta_2=theta_2, column_names=column_names)
		print("Optimal features %d" % len(S))
		for f in S:
			print(f)
		return S
	
	# With an integer: Set the number of splits
	elif isinstance(cv, int):
		if np.issubdtype(np.array(set(X_df[target_name])).dtype, np.integer):	# Discrete target: startified k-fold
			cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
			splits = cv.split(X_df, X_df[target_name])
		else:
			cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
			splits = cv.split(X_df, X_df[target_name])
	
	# With None: Default Cross-Validation k=5	
	elif not cv:
		if np.issubdtype(np.array(set(X_df[target_name])).dtype, np.integer):	# Discrete target: startified k-fold
			cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
			splits = cv.split(X_df, X_df[target_name])
		else:
			cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
			splits = cv.split(X_df, X_df[target_name])
	
	# With an iterable tuple: Splits are predefined
	elif isinstance(cv, tuple):
		splits = cv
	
	# With a CV_Splitter: Use the given CV_Splitter	
	else:
		splits = cv.split(X_df, X_df[target_name])
		
	
	S_val					= []
	for train_index, _ in splits:
		X_df_fold = X_df.iloc[train_index]
		S_fold    = CorrYiFS(X_df_fold, corr_method=corr_method, theta_2=theta_2, column_names=column_names)
		S_val.append(S_fold)	

	S_val = np.array([f for S_fold in S_val for f in S_fold])
	S = [cmn[0] for cmn in Counter(S_val).most_common() if cmn[1] == cv.get_n_splits()]

	if not S:
		S = CorrYiFS(X_df, corr_method=corr_method, theta_2=theta_2, column_names=column_names)
	
	print("Optimal features %d" % len(S))
	for f in S:
		print(f)

	return S