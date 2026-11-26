/*
  L-SHADE implemented by C++ for Special Session & Competition on Real-Parameter Single Objective Optimization at CEC-2014
  See the details of L-SHADE in the following paper:

  * Ryoji Tanabe and Alex Fukunaga: Improving the Search Performance of SHADE Using Linear Population Size Reduction,  Proc. IEEE Congress on Evolutionary Computation (CEC-2014), Beijing, July, 2014.

  Version: 1.1  Date: 9/Jun/2014
  Written by Ryoji Tanabe (rt.ryoji.tanabe [at] gmail.com)
*/
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include "de.h"
#include "conversao.hpp"

#include <stdexcept>
#include <chrono>
#include <random>
#include <sstream>
#include <unistd.h>

double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;

int g_function_number;
int g_problem_size;
unsigned int g_max_num_evaluations;
bool g_use_time_limit = false;
double g_max_time_ms = 0;
// time factor f (ms) of the flowshop stopping criterion n_jobs * n_machines * f
const double TIME_FACTOR_MS = 90.0;

const int MAX_JOBS = 500;
unsigned int n_jobs;
unsigned int n_machines;
unsigned int jobs[MAX_JOBS];// Array to store the maximum time for each job


int g_pop_size;
double g_arc_rate;
int g_memory_size;
double g_p_best_rate;
FlowShop::FlowShopEngine *g_flowshop_engine = nullptr;
// flowshop objective currently being optimized: Makespan, Tardiness, LMAX or Completion
string g_objective;

void cleanup_binary_func();

// objectives run by "--i <instance>", in this order (also the column order of the results line)
static const vector<string> ALL_OBJECTIVES = {"Makespan", "Tardiness", "LMAX", "Completion"};
static const string RESULTS_DIR = "resultados";

static bool is_objective(const string &name) {
  for (const string &objective : ALL_OBJECTIVES)
    if (objective == name) return true;
  return false;
}

// "Ta001_2.txt" or "sub/Ta001_2.txt" -> "Ta001_2"
static string base_name(const string &file_name) {
  size_t slash = file_name.find_last_of('/');
  string name = (slash == string::npos) ? file_name : file_name.substr(slash + 1);
  size_t dot = name.find_last_of('.');
  return (dot == string::npos || dot == 0) ? name : name.substr(0, dot);
}

// plain notation with a decimal comma: 14062.6 -> "14062,6", 14043 -> "14043"
static string format_value(double value) {
  std::ostringstream ss;
  ss << std::setprecision(10) << value;
  string text = ss.str();
  for (char &c : text)
    if (c == '.') c = ',';
  return text;
}

int main(int argc, char **argv) {
  string instance_path;
  string instance_name;
  // objectives run in this execution: one (--i <objective> <instance>) or all four (--i <instance>)
  vector<string> objectives;
  //number of runs
  int num_runs = 30;
    //dimension size. please select from 10, 30, 50, 100
  g_problem_size = 10;
  //available number of fitness evaluations
  g_max_num_evaluations = g_problem_size * 10000;

  // seed from the OS entropy source: parallel jobs started in the same second get different seeds
  srand(std::random_device{}());

  //L-SHADE parameters
  g_pop_size = (int)round(g_problem_size * 18);
  g_memory_size = 6;
  g_arc_rate = 2.6;
  g_p_best_rate = 0.11;

  // DM-L-SHADE parameters
  double elite_rate = 0.1;
  double clusters_rate = 0.1468;
  int mining_generation_step = 168;

  // Parse command-line arguments for function number
  int function_start = 1;
  int function_end = 31;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--f") == 0 && i + 1 < argc) {
      int func_num = atoi(argv[i + 1]);
      if (func_num >= 1 && func_num <= 31) {
        function_start = func_num;
        function_end = func_num;
      } else {
        cerr << "Invalid function number. Please use a value between 1 and 31." << endl;
        return 1;
      }
      break;
    }
    else if (strcmp(argv[i], "--i") == 0) {
      // --i <instance>              -> runs the four objectives, writes resultados/<instance>.csv
      // --i <objective> <instance>  -> runs one objective, writes resultados/<objective>/<instance>.csv
      if (i + 1 >= argc) {
        cerr << "Usage: " << argv[0] << " --i [Makespan|Tardiness|LMAX|Completion] <instance_file>" << endl;
        return 1;
      }
      if (is_objective(argv[i + 1])) {
        if (i + 2 >= argc) {
          cerr << "Usage: " << argv[0] << " --i [Makespan|Tardiness|LMAX|Completion] <instance_file>" << endl;
          return 1;
        }
        objectives = {argv[i + 1]};
        instance_name = argv[i + 2];
        i += 2;
      } else {
        objectives = ALL_OBJECTIVES;
        instance_name = argv[i + 1];
        i += 1;
      }
      instance_path = "src/instancias/" + instance_name;
      function_start = 1;
      function_end = 1;
    }
  }
  if (instance_path.empty() && function_start != function_end) {
    cerr << "Error: No input file specified. Please use the --i option to provide an input file." << endl;
    return 1;
  }

  string results_path;
  FlowShop::InstanceData instance;
  if (!instance_path.empty()) {
    try {
      instance = FlowShop::FlowShopEngine::carregar_instancia_txt(instance_path);
      g_problem_size = instance.n_jobs;
      g_max_num_evaluations = (g_problem_size * 10000);// Numero de execuções
      g_use_time_limit = true;// criterio de parada por tempo: n * m * f
      g_max_time_ms = instance.n_jobs * instance.n_machines * TIME_FACTOR_MS;
      g_pop_size = (int)round(g_problem_size * 18);
      g_flowshop_engine = new FlowShop::FlowShopEngine(instance.n_jobs, instance.n_machines, instance.proc_times, instance.due_dates);
    } catch (const std::exception &error) {
      cerr << "Error loading instance: " << error.what() << endl;
      return 1;
    }

    string results_dir = objectives.size() == 1 ? RESULTS_DIR + "/" + objectives[0] : RESULTS_DIR;
    results_path = results_dir + "/" + base_name(instance_name) + ".csv";
    // checked before the runs, so a job does not spend its whole time limit and then fail to save
    if (access(results_dir.c_str(), W_OK) != 0) {
      cerr << "Error: folder '" << results_dir << "/' does not exist or is not writable." << endl;
      delete g_flowshop_engine;
      return 1;
    }
  } else {
    // CEC mode (--f): one pass, nothing is saved
    objectives = {""};
  }

  if (function_start == function_end) {
    g_function_number = function_start;
    string label = instance_path.empty() ? "F" + std::to_string(g_function_number) : base_name(instance_name);
    // results line: instance;mean;best[;mean;best...] in the order of `objectives`
    string results_line = base_name(instance_name);

    for (const string &objective : objectives) {
      g_objective = objective;
      vector<Fitness> bsf_fitness_array(num_runs);
      auto objective_start = std::chrono::steady_clock::now();

      for (int j = 0; j < num_runs; j++) {
        //searchAlgorithm *alg = new LSHADE();
        int max_elite_size = std::round(elite_rate * g_pop_size);
        int number_of_patterns = std::round(clusters_rate * max_elite_size);
        searchAlgorithm *alg = new DMLSHADE(max_elite_size, number_of_patterns, mining_generation_step);
        bsf_fitness_array[j] = alg->run();
      }

      Fitness mean_bsf_fitness = 0;
      for (int j = 0; j < num_runs; j++) mean_bsf_fitness += bsf_fitness_array[j];
      mean_bsf_fitness /= num_runs;
      Fitness best_bsf_fitness = bsf_fitness_array[0];
      for (int j = 1; j < num_runs; j++) best_bsf_fitness = min(best_bsf_fitness, bsf_fitness_array[j]);
      double objective_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - objective_start).count();

      results_line += ";" + format_value(mean_bsf_fitness) + ";" + format_value(best_bsf_fitness);
      cout << label << (objective.empty() ? "" : " " + objective) << ": medio " << format_value(mean_bsf_fitness)
           << " melhor " << format_value(best_bsf_fitness) << " (" << format_value(std::round(objective_time * 10) / 10)
           << " s)" << endl;
    }

    // only flowshop mode (--i) saves; the file is overwritten, so a rerun replaces the old line
    if (!results_path.empty()) {
      std::ofstream results_file(results_path);
      results_file << results_line << "\n";
      results_file.close();
      if (!results_file) {
        cerr << "Error: could not write '" << results_path << "'." << endl;
        delete g_flowshop_engine;
        cleanup_binary_func();
        return 1;
      }
      cout << results_path << " <- " << results_line << endl;
    }
  }
  delete g_flowshop_engine;
  cleanup_binary_func();
  return 0;
}
