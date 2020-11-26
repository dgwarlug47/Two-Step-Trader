import pickle
rets_file_name = 'bagged_lstm'
infile = open(rets_file_name + '.pkl', 'rb')
rets_results = pickle.load(infile)
infile.close()

# list of dimension number of models, with rows as labels, columns are days:
for hyper_parameters in iterator_helper(hyper_parameters_list):
    all_aggregated_results = rets_results[tuple(hyper_parameters + [0])] - rets_results[tuple(hyper_parameters + [0])]
    break

def results_aggregator(rets_results, hyper_parameters_list):
  counter = 0
  for hyper_parameters in iterator_helper(hyper_parameters_list):
      num_trials = performance_results[tuple(hyper_parameters)]
      for trial in range(num_trials):
        counter = counter + 1
        all_aggregated_results = all_aggregated_results + rets_results[tuple(hyper_parameters + [trial])]
  all_aggregated_results = (all_aggregated_results) / counter