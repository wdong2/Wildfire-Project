# Wildfire-Project
3_region_result(folder)(code: 3_regions.py): data analysis for wild-fire data seperated to 3 parts based on latitude.
ASP_result(code: RegretBased_ASP.py): analysis for V-V*

Var_compare.py: variance comparison analysis. change the estimator in the main function compare_var([estimator1.pkl],[estimatir2.pkl])
	estimator variance files: var_ltp_h1_['MMR','optReg':RIS,'optVar':MM], var_IS. l,t,p are indexes.

data_processing.py: some utility functions used everywhere and some functions are dealing with the original data.
	You can call any function in the main. (change process_data to True to run the data analysis for the fire data, otherwise you can access any function definatrions in the file)

CI_script.py: the script for running CI estimation. result .pkl file: [# itration]r_CI_result_[estimator].pkl
	run_CI(runs,100,None,"h1.pkl"):
		Change runs to change # of itrations,
		Change 100 for number of observations,
		None for using all the method: CLT,T,chi and f (do not change this)
		"h1.pkl" for different critical level distribution, other critical level distribution can be obtained by "read_p" in "simulation study plot_Wang" folder.
	line30: ins = CI.get_delta(n_ins, p_astar, pi_l, i): the 'i' is for randomization seed, you can change this to others like: i+2000, i*20, for different itrations
CI_util.py: contains all the utility function used in CI_script.py.
CI_analyze.py: you can do 3 things; before them, first run "get_info_coverage([pkl file for the result by script],[either "max" or "min" for high coverage or low],[estimators name for resulting .pkl file which contains a list of index either for high coverage case or low])"
	       After run get_info_coverage, you will get a "mean_[# ieration]r_[estimator name you enter].pkl" file: that is for estimates mean over [# ieration] of itration	
	analyze_mean([mean_file],[index_file],[mode]): enter the file "mean_[# ieration]r_[estimator name you enter].pkl" from the previous function, the index file from previous function to get the distribution plot either for high variance case or low.
		[mode] is either an index for the scenario(0-1209)(the second file does not metter in this case), "inf" for all the scenarios in the second input file.
	scatter_plot([coverage_file: ending '_MMR' for the MMR estimaor, no ending for MM], [mode], [variance_file](optional) )
		[mode]: False for general plot of coverage-variance (variance is by formula)
			or "mean" for variance-coverage (variance is by [#] of iterations) scatter plot (in this case, you need to enter [variance_file] from CI_script.py),
			or "VvsV" for variance vs variance plot (you need [variance_file])
			or could be a real number 'a' for seperating variance <=a and >a, for example 0.5 for variance-coverage plot (variance by formula).
(for CI analysis, only CI_script.py need compute canada)
