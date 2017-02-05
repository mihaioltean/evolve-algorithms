//---------------------------------------------------------------------------
//   Evolving Evolutionary Algorithms for MultiObjective problems
//   Version 2017.02.05.0

//   Compiled with Microsoft Visual C++ 2013

//   MIT License

//   New versions of this program will be available at: https://github.com/mihaioltean/evolve-algorithms

//   Please reports any sugestions and/or bugs to mihai.oltean@gmail.com


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h> 

#include "lista_voidp.h"

#define NUM_MICRO_EA_OPERATORS 2

#define MICRO_EA_CROSSOVER 0
#define MICRO_EA_MUTATION 1
#define MICRO_EA_RANDOM_INIT 2
#define MICRO_EA_DOMINATOR_SELECTION 3
#define MICRO_EA_DOMINATED_SELECTION 4
#define REPLACE_IF_DOMINATED 5

#define MAX_ARITY 3

//---------------------------------------------------------------------------
struct t_code3{// three address code
	int op;		// operators are the MICRO EA OPERATORS
	int addr[3];    // pointers to arguments
	int dest;
};
//---------------------------------------------------------------------------
struct t_meta_gp_chromosome{
	t_code3 *prg;        // the program - a string of genes

	double fitness;        // the fitness (or the error)
};
//---------------------------------------------------------------------------
struct t_meta_gp_parameters{
	int code_length;             // number of instructions in a chromosome
	int num_generations;
	int pop_size;                // population size
	double mutation_probability, crossover_probability;
	double mutation_occurence_prob, crossover_occurence_prob;
};
//---------------------------------------------------------------------------
struct t_micro_ea_parameters{
	double mutation_probability;
	int num_runs;
	int num_bits_per_dimension;
	int pop_size;
	int num_generations;
};
//---------------------------------------------------------------------------
void allocate_meta_chromosome(t_meta_gp_chromosome &c, t_meta_gp_parameters &params)
{
	c.prg = new t_code3[params.code_length];
}
//---------------------------------------------------------------------------
void delete_meta_chromosome(t_meta_gp_chromosome &c)
{
	if (c.prg) {
		delete[] c.prg;
		c.prg = NULL;
	}
}
//---------------------------------------------------------------------------
void copy_individual(t_meta_gp_chromosome& dest, const t_meta_gp_chromosome& source, t_meta_gp_parameters &params)
{
	for (int i = 0; i < params.code_length; i++)
		dest.prg[i] = source.prg[i];
	dest.fitness = source.fitness;
}
//---------------------------------------------------------------------------
void generate_random_meta_chromosome(t_meta_gp_chromosome &a_chromosome, t_meta_gp_parameters &meta_gp_params, int micro_ea_pop_size) // randomly initializes the individuals
{
	// for all other genes we put either an operator, variable or constant
	for (int i = 0; i < meta_gp_params.code_length; i++) {
		double p = rand() / (double)RAND_MAX;

		a_chromosome.prg[i].op = rand() % NUM_MICRO_EA_OPERATORS;

		for (int a = 0; a < MAX_ARITY; a++)
			a_chromosome.prg[i].addr[a] = rand() % micro_ea_pop_size;
		a_chromosome.prg[i].dest = rand() % micro_ea_pop_size;

	}
}
//---------------------------------------------------------------------------
void mutate_meta_chromosome(t_meta_gp_chromosome &a_chromosome, t_meta_gp_parameters& params, int micro_ea_pop_size) // mutate the individual
{
	// mutate each symbol with the given probability
	// no mutation for the first micro_ea_pop_size genes because they are only used for initialization of the micro ea population
	// other genes

	for (int i = 0; i < params.code_length; i++) {
		double p = rand() / (double)RAND_MAX;      // mutate the operator
		if (p < params.mutation_probability) {
			// we mutate it, but we have to decide what we put here
			a_chromosome.prg[i].op = rand() % NUM_MICRO_EA_OPERATORS;
		}
		for (int a = 0; a < MAX_ARITY; a++) {
			p = rand() / (double)RAND_MAX;      // mutate the first address  (adr1)
			if (p < params.mutation_probability)
				a_chromosome.prg[i].addr[a] = rand() % micro_ea_pop_size;
		}

		p = rand() / (double)RAND_MAX;      // mutate the destination
		if (p < params.mutation_probability)
			a_chromosome.prg[i].dest = rand() % micro_ea_pop_size;
	}
}
//---------------------------------------------------------------------------
void one_cut_point_crossover(const t_meta_gp_chromosome &parent1, const t_meta_gp_chromosome &parent2, t_meta_gp_parameters &params, t_meta_gp_chromosome &offspring1, t_meta_gp_chromosome &offspring2)
{
	int cutting_pct = rand() % params.code_length;
	for (int i = 0; i < cutting_pct; i++) {
		offspring1.prg[i] = parent1.prg[i];
		offspring2.prg[i] = parent2.prg[i];
	}
	for (int i = cutting_pct; i < params.code_length; i++) {
		offspring1.prg[i] = parent2.prg[i];
		offspring2.prg[i] = parent1.prg[i];
	}
}
//---------------------------------------------------------------------------
void uniform_crossover(const t_meta_gp_chromosome &parent1, const t_meta_gp_chromosome &parent2, t_meta_gp_parameters &params, t_meta_gp_chromosome &offspring1, t_meta_gp_chromosome &offspring2)
{
	for (int i = 0; i < params.code_length; i++)
		if (rand() % 2) {
			offspring1.prg[i] = parent1.prg[i];
			offspring2.prg[i] = parent2.prg[i];
		}
		else {
			offspring1.prg[i] = parent2.prg[i];
			offspring2.prg[i] = parent1.prg[i];
		}
}
//---------------------------------------------------------------------------
void print_meta_chromosome(t_meta_gp_chromosome& an_individual, int code_length)
{
	printf("The chromosome is:\n");

	for (int i = 0; i < code_length; i++)
		switch (an_individual.prg[i].op) {
		case MICRO_EA_RANDOM_INIT:
			printf("%d: MICRO_EA_RANDOM_INIT\n", i);
			break;
		case MICRO_EA_DOMINATOR_SELECTION:
			printf("%d: MICRO_EA_DOMINATOR_SELECTION(%d, %d, %d)\n", i, an_individual.prg[i].addr[0], an_individual.prg[i].addr[1], an_individual.prg[i].addr[2]);
			break;
		case MICRO_EA_DOMINATED_SELECTION:
			printf("%d: MICRO_EA_DOMINATED_SELECTION(%d, %d, %d)\n", i, an_individual.prg[i].addr[0], an_individual.prg[i].addr[1], an_individual.prg[i].addr[2]);
			break;
		case MICRO_EA_CROSSOVER:
			printf("%d: MICRO_EA_CROSSOVER(%d, %d)\n", i, an_individual.prg[i].addr[0], an_individual.prg[i].addr[0]);
			break;
		case MICRO_EA_MUTATION:
			printf("%d: MICRO_EA_MUTATION(%d)\n", i, an_individual.prg[i].addr[0]);
			break;
	}

	printf("Fitness = %lf\n", an_individual.fitness);
}
//---------------------------------------------------------------------------

int sort_function(const void *a, const void *b)
{
	if (((t_meta_gp_chromosome *)a)->fitness < ((t_meta_gp_chromosome *)b)->fitness)
		return 1;
	else
		if (((t_meta_gp_chromosome *)a)->fitness > ((t_meta_gp_chromosome *)b)->fitness)
			return -1;
		else
			return 0;
}
//--------------------------------------------------------------------
int tournament_selection(t_meta_gp_chromosome *pop, int pop_size, int tournament_size)     // Size is the size of the tournament
{
	int r, p;
	p = rand() % pop_size;
	for (int i = 1; i < tournament_size; i++) {
		r = rand() % pop_size;
		p = pop[r].fitness < pop[p].fitness ? r : p;
	}
	return p;
}
//---------------------------------------------------------------------------
// function to be optimized
//---------------------------------------------------------------------------
//ZDT1
//---------------------------------------------------------------------------
double f1(double *p, int num_dimensions)
{
	return p[0];
}
//---------------------------------------------------------------------------
double f2(double* p, int num_dimensions)
{
	// test function T1

	double sum = 0;
	for (int i = 1; i < num_dimensions; i++)
		sum += p[i];
	double g = 1 + (9 * sum) / (double)(num_dimensions - 1);

	double h = 1 - sqrt(f1(p, num_dimensions) / g);

	return g * h;
}
//---------------------------------------------------------------------------
double binary_to_real(char* b_string, int num_bits_per_dimension, double min_x, double max_x)
{
	// transform a binary string of num_bits_per_dimension size into a real number in [min_x ... max_x] interval
	unsigned int x_int = 0;
	for (int j = 0; j < num_bits_per_dimension; j++) {
		//	x_real = x_real * 2 + (int)b_string[j];    // now I have them in interval [0 ... 2 ^ num_bits_per_dimension - 1]
		x_int <<= 1;
		x_int |= b_string[j];
	}
	double x_real = x_int;
	x_real /= ((1 << num_bits_per_dimension) - 1); // now I have them in [0 ... 1] interval
	x_real *= (max_x - min_x);  // now I have them in [0 ... max_x - min_x] interval
	x_real += min_x;            // now I have them in [min_x ... max_x] interval

	return x_real;
}
//--------------------------------------------------------------------
bool dominates(double* p1, double* p2) // returns true if p1 dominates p2
{
	for (int i = 0; i < 2; i++)
		if (p2[i] < p1[i])
			return false;
	for (int i = 0; i < 2; i++)
		if (p1[i] < p2[i])
			return true;
	return false;
}
//---------------------------------------------------------------------------
void sort_list(TLista &nondominated)
{
	bool sorted = false;
	while (!sorted) {
		sorted = true;
		for (node_double_linked * node_p = nondominated.head; node_p->next; node_p = node_p->next) {
			double *p = (double*)nondominated.GetCurrentInfo(node_p);
			double *p_next = (double*)nondominated.GetCurrentInfo(node_p->next);
			if (p[0] > p_next[0]) {
				void *tmp_inf = node_p->inf;
				node_p->inf = node_p->next->inf;
				node_p->next->inf = tmp_inf;
				sorted = false;
			}
		}
	}
}
//---------------------------------------------------------------------------
double compute_hypervolume(TLista &nondominated, double *reference)
{
	sort_list(nondominated);

	double hyper_volume = 0;
	for (node_double_linked * node_p = nondominated.head; node_p->next; node_p = node_p->next) {
		double *p = (double*)nondominated.GetCurrentInfo(node_p);
		double *p_next = (double*)nondominated.GetCurrentInfo(node_p->next);
		hyper_volume += (p_next[0] - p[0]) * (reference[1] - p[1]);
	}
	double *p = (double*)nondominated.GetTailInfo();
	hyper_volume += (reference[0] - p[0]) * (reference[1] - p[1]);

	return hyper_volume;
}
//---------------------------------------------------------------------------
int get_dominator_index(int individuals[3], double **micro_fitness)
{
	if (dominates(micro_fitness[individuals[0]], micro_fitness[individuals[1]]))
		if (dominates(micro_fitness[individuals[0]], micro_fitness[individuals[2]]))
			return 0;
		else
			return 2;
	else
		if (dominates(micro_fitness[individuals[1]], micro_fitness[individuals[2]]))
			return 1;
		else
			return 2;
}
//---------------------------------------------------------------------------
int get_dominated_index(int individuals[3], double **micro_fitness)
{
	if (dominates(micro_fitness[individuals[0]], micro_fitness[individuals[1]]))
		if (dominates(micro_fitness[individuals[2]], micro_fitness[individuals[1]]))
			return 1;
		else
			return 2;
	else
		if (dominates(micro_fitness[individuals[2]], micro_fitness[individuals[0]]))
			return 0;
		else
			return 2;
}
//---------------------------------------------------------------------------
double make_one_run(t_meta_gp_chromosome &an_individual, int code_length, t_micro_ea_parameters &micro_params, int num_dimensions, double min_x, double max_x, double **micro_fitness, char **micro_values, double *x, FILE *f)
{
	for (int i = 0; i < micro_params.pop_size; i++) {
		for (int j = 0; j < num_dimensions; j++)
			for (int k = 0; k < micro_params.num_bits_per_dimension; k++)
				micro_values[i][j * micro_params.num_bits_per_dimension + k] = rand() % 2; // random values

		// compute fitness of that micro chromosome
		// transform to base 10
		for (int j = 0; j < num_dimensions; j++)
			x[j] = binary_to_real(micro_values[i] + j * micro_params.num_bits_per_dimension, micro_params.num_bits_per_dimension, min_x, max_x);

		micro_fitness[i][0] = f1(x, num_dimensions);// apply f - compute fitness of micro
		micro_fitness[i][1] = f2(x, num_dimensions);// apply f - compute fitness of micro
	}

	// create a list with nondominated
	TLista nondominated;
	nondominated.Add(micro_fitness[0]);

	for (int i = 1; i < micro_params.pop_size; i++) {
		node_double_linked * node_p = nondominated.head;
		bool dominated = false;
		while (node_p) {
			double *p = (double*)nondominated.GetCurrentInfo(node_p);
			if (dominates(p, micro_fitness[i])) {
				dominated = true;
				break;
			}
			else
				if (dominates(micro_fitness[i], p))
					node_p = nondominated.DeleteCurrent(node_p);
				else// move to the next one
					node_p = node_p->next;
		}
		if (!dominated)
			nondominated.Add(micro_fitness[i]);
	}

	// compute the distance to front
	double reference[2] = { 11, 11 };

	double hyper_volume1 = compute_hypervolume(nondominated, reference);


	int destination_index;
	int dominator_index;
	int dominated_index;

	double tmp_micro_fitness[2];
	char *tmp_micro_values = new char[num_dimensions * micro_params.num_bits_per_dimension];

	for (int g = 1; g < micro_params.num_generations; g++) {

		for (int i = 0; i < code_length; i++) {

			switch (an_individual.prg[i].op) {

			case MICRO_EA_DOMINATOR_SELECTION:  // Selection keeps the dominator
				destination_index = an_individual.prg[i].dest;
				dominator_index = get_dominator_index(an_individual.prg[i].addr, micro_fitness);

				memcpy(micro_values[destination_index], micro_values[dominator_index], micro_params.num_bits_per_dimension * num_dimensions);
				micro_fitness[destination_index][0] = micro_fitness[dominator_index][0];
				micro_fitness[destination_index][1] = micro_fitness[dominator_index][1];
				break;

			case MICRO_EA_DOMINATED_SELECTION:  // Selection keeps the dominated
				destination_index = an_individual.prg[i].dest;
				dominated_index = get_dominated_index(an_individual.prg[i].addr, micro_fitness);

				memcpy(micro_values[destination_index], micro_values[dominated_index], micro_params.num_bits_per_dimension * num_dimensions);
				micro_fitness[destination_index][0] = micro_fitness[dominated_index][0];
				micro_fitness[destination_index][1] = micro_fitness[dominated_index][1];

				break;

			case MICRO_EA_CROSSOVER:  // crossover
				destination_index = an_individual.prg[i].dest;
				for (int j = 0; j < num_dimensions * micro_params.num_bits_per_dimension; j++){
						int p = rand() % 2;
						if (p)
							tmp_micro_values[j] = micro_values[an_individual.prg[i].addr[0]][j];
						else
							tmp_micro_values[j] = micro_values[an_individual.prg[i].addr[1]][j];
					}

				// compute fitness of that micro chromosome
				// transform to base 10
				for (int j = 0; j < num_dimensions; j++)
					x[j] = binary_to_real(tmp_micro_values + j * micro_params.num_bits_per_dimension, micro_params.num_bits_per_dimension, min_x, max_x);

				tmp_micro_fitness[0] = f1(x, num_dimensions);// apply f - compute fitness of micro
				tmp_micro_fitness[1] = f2(x, num_dimensions);// apply f - compute fitness of micro

				if (dominates(tmp_micro_fitness, micro_fitness[an_individual.prg[i].dest])) {
					memcpy(micro_values[destination_index], tmp_micro_values, micro_params.num_bits_per_dimension * num_dimensions);
					micro_fitness[destination_index][0] = tmp_micro_fitness[0];
					micro_fitness[destination_index][1] = tmp_micro_fitness[1];
				}

				break;

			case MICRO_EA_MUTATION:  // Mutation with a fixed mutation probability
				destination_index = an_individual.prg[i].dest;
				for (int j = 0; j < num_dimensions * micro_params.num_bits_per_dimension; j++) {
						double p = rand() / (double)RAND_MAX;
						if (p < micro_params.mutation_probability)
							tmp_micro_values[j] = 1 - micro_values[an_individual.prg[i].addr[0]][j];
						else
							tmp_micro_values[j] = micro_values[an_individual.prg[i].addr[0]][j];
					}

				// compute fitness of that micro chromosome
				// transform to base 10
				for (int j = 0; j < num_dimensions; j++)
					x[j] = binary_to_real(tmp_micro_values + j * micro_params.num_bits_per_dimension, micro_params.num_bits_per_dimension, min_x, max_x);

				tmp_micro_fitness[0] = f1(x, num_dimensions); // apply f - compute fitness of micro
				tmp_micro_fitness[1] = f2(x, num_dimensions); // apply f - compute fitness of micro

				if (dominates(tmp_micro_fitness, micro_fitness[an_individual.prg[i].dest])) {
					memcpy(micro_values[destination_index], tmp_micro_values, micro_params.num_bits_per_dimension * num_dimensions);
					micro_fitness[destination_index][0] = tmp_micro_fitness[0];
					micro_fitness[destination_index][1] = tmp_micro_fitness[1];
				}
				break;

			case REPLACE_IF_DOMINATED:  //
				destination_index = an_individual.prg[i].dest;
				if (dominates(micro_fitness[an_individual.prg[i].addr[0]], micro_fitness[an_individual.prg[i].dest])) {
					memcpy(micro_values[destination_index], micro_values[an_individual.prg[i].addr[0]], micro_params.num_bits_per_dimension * num_dimensions);
					micro_fitness[destination_index][0] = micro_fitness[an_individual.prg[i].addr[0]][0];
					micro_fitness[destination_index][1] = micro_fitness[an_individual.prg[i].addr[0]][1];
				}
				break;
			}
		}

		// create a list with nondominated
		nondominated.Clear();
		nondominated.Add(micro_fitness[0]);

		for (int i = 1; i < micro_params.pop_size; i++) {
			node_double_linked * node_p = nondominated.head;
			bool dominated = false;
			while (node_p) {
				double *p = (double*)nondominated.GetCurrentInfo(node_p);
				if (dominates(p, micro_fitness[i])) {
					dominated = true;
					break;
				}
				else
					if (dominates(micro_fitness[i], p))
						node_p = nondominated.DeleteCurrent(node_p);
					else// move to the next one
						node_p = node_p->next;
			}
			if (!dominated)
				nondominated.Add(micro_fitness[i]);
		}

		// compute the distance to front

		double hyper_volume2 = compute_hypervolume(nondominated, reference);
		if (hyper_volume1 > hyper_volume2) {
			break;
		}
		hyper_volume1 = hyper_volume2;
	}

	if (f) {
		fprintf(f, "%lf\n", hyper_volume1);
		fprintf(f, "%d\n", nondominated.count);
		for (node_double_linked* node_p = nondominated.head; node_p; node_p = node_p->next) {
			double *p = (double*)nondominated.GetCurrentInfo(node_p);
			fprintf(f, "%lf %lf ", p[0], p[1]);
		}
		fprintf(f, "\n");
	}

	delete[] tmp_micro_values;
	nondominated.Clear();
	return hyper_volume1;
}
//---------------------------------------------------------------------------
void compute_fitness(t_meta_gp_chromosome &an_individual, int code_length, t_micro_ea_parameters &micro_params, int num_dimensions, double min_x, double max_x, double **micro_fitness, char **micro_values, double *x, FILE *f)
{
	double average_hypervolume = 0;  // average fitness over all runs
	// evaluate code
	for (int r = 0; r < micro_params.num_runs; r++) {// micro ea is run on multi runs
		//srand(r);
		double hyper_volume = make_one_run(an_individual, code_length, micro_params, num_dimensions, min_x, max_x, micro_fitness, micro_values, x, f);
		// add to average
		average_hypervolume += hyper_volume;
	}


	average_hypervolume /= (double)micro_params.num_runs;

	an_individual.fitness = average_hypervolume;
}
//---------------------------------------------------------------------------
void start_steady_state_mep(t_meta_gp_parameters &meta_gp_params, t_micro_ea_parameters &micro_params, int num_dimensions, double min_x, double max_x, FILE *f_out)       // Steady-State 
{
	// a steady state approach:
	// we work with 1 population
	// newly created individuals will replace the worst existing ones (only if they are better).

	// allocate memory
	t_meta_gp_chromosome *population;
	population = new t_meta_gp_chromosome[meta_gp_params.pop_size];
	for (int i = 0; i < meta_gp_params.pop_size; i++)
		allocate_meta_chromosome(population[i], meta_gp_params);

	t_meta_gp_chromosome offspring1, offspring2;
	allocate_meta_chromosome(offspring1, meta_gp_params);
	allocate_meta_chromosome(offspring2, meta_gp_params);


	double *x = new double[num_dimensions]; // buffer for storing real values

	double **micro_fitness;

	micro_fitness = new double*[micro_params.pop_size]; // fitness for each micro EA chromosome
	for (int i = 0; i < micro_params.pop_size; i++)
		micro_fitness[i] = new double[2];

	// allocate some memory
	char **micro_values;
	micro_values = new char*[micro_params.pop_size];
	for (int i = 0; i < micro_params.pop_size; i++)
		micro_values[i] = new char[num_dimensions * micro_params.num_bits_per_dimension];

	clock_t start_time = clock();

	double mean_fitness = 0;
	// initialize
	for (int i = 0; i < meta_gp_params.pop_size; i++) {
		generate_random_meta_chromosome(population[i], meta_gp_params, micro_params.pop_size);
		compute_fitness(population[i], meta_gp_params.code_length, micro_params, num_dimensions, min_x, max_x, micro_fitness, micro_values, x, NULL);
		mean_fitness += population[i].fitness;
	}
	// sort ascendingly by fitness
	qsort((void *)population, meta_gp_params.pop_size, sizeof(population[0]), sort_function);

	mean_fitness /= (double)meta_gp_params.pop_size;
	printf("generation %d, best fitness = %lf mean = %lf\n", 0, population[0].fitness, mean_fitness);
	// print the front of the best
	fprintf(f_out, "%d\n", 0);
	srand(0);
	make_one_run(population[0], meta_gp_params.code_length, micro_params, num_dimensions, min_x, max_x, micro_fitness, micro_values, x, f_out);

	for (int g = 1; g < meta_gp_params.num_generations; g++) {// for each generation
		for (int k = 0; k < meta_gp_params.pop_size; k += 2) {
			// choose the parents using binary tournament
			int r1 = rand() % meta_gp_params.pop_size;
			int r2 = rand() % meta_gp_params.pop_size;
			// crossover
			double p = rand() / double(RAND_MAX);
			if (p < meta_gp_params.crossover_probability)
				one_cut_point_crossover(population[r1], population[r2], meta_gp_params, offspring1, offspring2);
			else {// no crossover so the offspring are a copy of the parents
				copy_individual(offspring1, population[r1], meta_gp_params);
				copy_individual(offspring2, population[r2], meta_gp_params);
			}
			// mutate the result and compute fitness
			mutate_meta_chromosome(offspring1, meta_gp_params, micro_params.pop_size);
			compute_fitness(offspring1, meta_gp_params.code_length, micro_params, num_dimensions, min_x, max_x, micro_fitness, micro_values, x, NULL);
			// mutate the other offspring and compute fitness
			mutate_meta_chromosome(offspring2, meta_gp_params, micro_params.pop_size);
			compute_fitness(offspring2, meta_gp_params.code_length, micro_params, num_dimensions, min_x, max_x, micro_fitness, micro_values, x, NULL);

			// replace the worst in the population
			if (offspring1.fitness > population[meta_gp_params.pop_size - 1].fitness) {
				copy_individual(population[meta_gp_params.pop_size - 1], offspring1, meta_gp_params);
				qsort((void *)population, meta_gp_params.pop_size, sizeof(population[0]), sort_function);
			}
			if (offspring2.fitness > population[meta_gp_params.pop_size - 1].fitness) {
				copy_individual(population[meta_gp_params.pop_size - 1], offspring2, meta_gp_params);
				qsort((void *)population, meta_gp_params.pop_size, sizeof(population[0]), sort_function);
			}
		}
		mean_fitness = 0;
		for (int k = 0; k < meta_gp_params.pop_size; k++)
			mean_fitness += population[k].fitness;
		mean_fitness /= (double)meta_gp_params.pop_size;

		// print the front of the best
		fprintf(f_out, "%d\n", g);
		srand(0);
		//make_one_run(population[0], meta_gp_params.code_length, micro_params, num_dimensions, min_x, max_x, micro_fitness, micro_values, x, f_out);
		printf("generation %d, best fitness = %lf mean = %lf\n", g, population[0].fitness, mean_fitness);

	}

	clock_t end_time = clock();

	double running_time_2d = (double)(end_time - start_time) / CLOCKS_PER_SEC;

	// print best chromosome
	//	print_meta_chromosome(population[0], meta_gp_params.code_length);

	printf("running time = %lf\n", running_time_2d);

	// free memory
	delete_meta_chromosome(offspring1);
	delete_meta_chromosome(offspring2);

	for (int i = 0; i < meta_gp_params.pop_size; i++)
		delete_meta_chromosome(population[i]);
	delete[] population;


	for (int i = 0; i < meta_gp_params.code_length; i++)
		delete[] micro_values[i];
	delete[] micro_values;

	for (int i = 0; i < meta_gp_params.code_length; i++)
		delete[] micro_fitness[i];
	delete[] micro_fitness;

	delete[] x;

}
//--------------------------------------------------------------------
bool read_reference_points(char *file_name, TLista &reference_points)
{
	FILE * f = fopen(file_name, "r");

	if (!f)
		return false;

	char c;
	for (int i = 0; i < 100; i++) {
		double *p = new double[2];
		fscanf(f, "%lf%c%lf", &p[0], &c, &p[1]);
		reference_points.Add(p);
	}

	fclose(f);
	return true;
}
//--------------------------------------------------------------------
int main(void)
{
	int t = sizeof(char*);

	t_meta_gp_parameters meta_gp_params;

	meta_gp_params.pop_size = 50;						    // the number of individuals in population  (must be an even number!)
	meta_gp_params.code_length = 100;
	meta_gp_params.num_generations = 100;					// the number of generations
	meta_gp_params.mutation_probability = 0.02;             // mutation probability
	meta_gp_params.crossover_probability = 0.9;             // crossover probability

	meta_gp_params.mutation_occurence_prob = 0.3;
	meta_gp_params.crossover_occurence_prob = 0.3;

	t_micro_ea_parameters micro_ea_params;
	micro_ea_params.mutation_probability = 0.001;
	micro_ea_params.num_bits_per_dimension = 30;
	micro_ea_params.num_runs = 30;
	micro_ea_params.pop_size = 100;
	micro_ea_params.num_generations = 100;

	FILE *f_out = fopen("c:\\temp\\zdt1_front.txt", "w");

	printf("evolution started...\n");
	srand(0);
	start_steady_state_mep(meta_gp_params, micro_ea_params, 30, 0, 1, f_out);

	fclose(f_out);


	printf("Press enter ...");
	getchar();

	return 0;
}
//--------------------------------------------------------------------