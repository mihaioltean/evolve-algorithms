//---------------------------------------------------------------------------
//   Multi Expression Programming for evolving Evolutionary Algorithms for MultiObjective problems
//   Copyright (C) 2002-2016, Mihai Oltean  (mihai.oltean@gmail.com)
//   Version 2016.11.22.0

//   Compiled with Microsoft Visual C++ 2013
//   Just create a console application and set this file as the main file of the project

//   MIT License

//   New versions of this program will be available at: 

//   Please reports any sugestions and/or bugs to       



#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#include "lista_voidp.h"

#define NUM_MICRO_EA_OPERATORS 4

#define MICRO_EA_RANDOM_INIT 0
#define MICRO_EA_DOMINATOR_SELECTION 1
#define MICRO_EA_CROSSOVER 2
#define MICRO_EA_MUTATION 3

//---------------------------------------------------------------------------
struct t_code3{// three address code
	int op;		// operators are the MICRO EA OPERATORS
	int adr1, adr2;    // pointers to arguments
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
};
//---------------------------------------------------------------------------
struct t_micro_ea_parameters{
	double mutation_probability;
	int num_runs;
	int num_bits_per_dimension;
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
void generate_random_meta_chromosome(t_meta_gp_chromosome &a, t_meta_gp_parameters &params) // randomly initializes the individuals
{
	a.prg[0].op = MICRO_EA_RANDOM_INIT; // init only

	// for all other genes we put either an operator, variable or constant
	for (int i = 1; i < params.code_length; i++) {
		a.prg[i].op = rand() % NUM_MICRO_EA_OPERATORS;

		a.prg[i].adr1 = rand() % i;
		a.prg[i].adr2 = rand() % i;
	}
}
//---------------------------------------------------------------------------
void mutate_meta_chromosome(t_meta_gp_chromosome &a_chromosome, t_meta_gp_parameters& params) // mutate the individual
{
	// mutate each symbol with the given probability
	// no mutation for the first gene
	// other genes

	for (int i = 1; i < params.code_length; i++) {
		double p = rand() / (double)RAND_MAX;      // mutate the operator
		if (p < params.mutation_probability) {
			// we mutate it, but we have to decide what we put here
			p = rand() / (double)RAND_MAX;

			a_chromosome.prg[i].op = rand() % NUM_MICRO_EA_OPERATORS;
		}

		p = rand() / (double)RAND_MAX;      // mutate the first address  (adr1)
		if (p < params.mutation_probability)
			a_chromosome.prg[i].adr1 = rand() % i;

		p = rand() / (double)RAND_MAX;      // mutate the second address   (adr2)
		if (p < params.mutation_probability)
			a_chromosome.prg[i].adr2 = rand() % i;
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
int sort_function(const void *a, const void *b)
{// comparator for quick sort
	if (((t_meta_gp_chromosome *)a)->fitness > ((t_meta_gp_chromosome *)b)->fitness)
		return 1;
	else
		if (((t_meta_gp_chromosome *)a)->fitness < ((t_meta_gp_chromosome *)b)->fitness)
			return -1;
		else
			return 0;
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
			printf("%d: MICRO_EA_DOMINATOR_SELECTION(%d, %d)\n", i, an_individual.prg[i].adr1, an_individual.prg[i].adr2);
			break;
		case MICRO_EA_CROSSOVER:
			printf("%d: MICRO_EA_CROSSOVER(%d)\n", i, an_individual.prg[i].adr1);
			break;
		case MICRO_EA_MUTATION:
			printf("%d: MICRO_EA_MUTATION(%d, %d)\n", i, an_individual.prg[i].adr1, an_individual.prg[i].adr2);
			break;
	}
		
	printf("Fitness = %lf\n", an_individual.fitness);
}
//---------------------------------------------------------------------------
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
	double x_real = 0;
	for (int j = 0; j < num_bits_per_dimension; j++)
		x_real = x_real * 2 + (int)b_string[j];    // now I have them in interval [0 ... 2 ^ num_bits_per_dimension - 1]
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
inline double euclidian_distance(double* p1, double* p2)
{
	return sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]));
}
//---------------------------------------------------------------------------
double make_one_run(t_meta_gp_chromosome &an_individual, int code_length, t_micro_ea_parameters &micro_params, int num_dimensions, double min_x, double max_x, double** reference_points, int num_reference_points, double **micro_fitness, char **micro_values, double *x, FILE *f)
{
	for (int i = 0; i < code_length; i++) {
		switch (an_individual.prg[i].op) {
		case  MICRO_EA_RANDOM_INIT:       // Initialization
			for (int j = 0; j < num_dimensions; j++)
				for (int k = 0; k < micro_params.num_bits_per_dimension; k++)
					micro_values[i][j * micro_params.num_bits_per_dimension + k] = rand() % 2; // random values

			// compute fitness of that micro chromosome
			// transform to base 10
			for (int j = 0; j < num_dimensions; j++)
				x[j] = binary_to_real(micro_values[i], micro_params.num_bits_per_dimension, min_x, max_x);

			micro_fitness[i][0] = f1(x, num_dimensions);// apply f - compute fitness of micro
			micro_fitness[i][1] = f2(x, num_dimensions);// apply f - compute fitness of micro
			break;

		case MICRO_EA_DOMINATOR_SELECTION:  // Selection (binary tournament)
			if (dominates(micro_fitness[an_individual.prg[i].adr1], micro_fitness[an_individual.prg[i].adr2])) {
				memcpy(micro_values[i], micro_values[an_individual.prg[i].adr1], micro_params.num_bits_per_dimension * num_dimensions);
				micro_fitness[i][0] = micro_fitness[an_individual.prg[i].adr1][0];
				micro_fitness[i][1] = micro_fitness[an_individual.prg[i].adr1][1];
			}
			else { // p2 dominates or are nondominated
				memcpy(micro_values[i], micro_values[an_individual.prg[i].adr2], micro_params.num_bits_per_dimension * num_dimensions);
				micro_fitness[i][0] = micro_fitness[an_individual.prg[i].adr2][0];
				micro_fitness[i][1] = micro_fitness[an_individual.prg[i].adr2][1];
			}
			break;

		case MICRO_EA_CROSSOVER:  // Mutation with a fixed mutation probability
			for (int j = 0; j < num_dimensions; j++)
				for (int k = 0; k < micro_params.num_bits_per_dimension; k++) {
					int p = rand() % 2;
					if (p)
						micro_values[i][j * micro_params.num_bits_per_dimension + k] = micro_values[an_individual.prg[i].adr1][j * micro_params.num_bits_per_dimension + k];
					else
						micro_values[i][j * micro_params.num_bits_per_dimension + k] = micro_values[an_individual.prg[i].adr2][j * micro_params.num_bits_per_dimension + k];
				}

			// compute fitness of that micro chromosome
			// transform to base 10
			for (int j = 0; j < num_dimensions; j++)
				x[j] = binary_to_real(micro_values[i], micro_params.num_bits_per_dimension, min_x, max_x);

			micro_fitness[i][0] = f1(x, num_dimensions);// apply f - compute fitness of micro
			micro_fitness[i][1] = f2(x, num_dimensions);// apply f - compute fitness of micro

			break;

		case MICRO_EA_MUTATION:  // Mutation with a fixed mutation probability
			for (int j = 0; j < num_dimensions; j++)
				for (int k = 0; k < micro_params.num_bits_per_dimension; k++) {
					double p = rand() / (double)RAND_MAX;
					if (p < micro_params.mutation_probability)
						micro_values[i][j * micro_params.num_bits_per_dimension + k] = 1 - micro_values[an_individual.prg[i].adr1][j * micro_params.num_bits_per_dimension + k];
					else
						micro_values[i][j * micro_params.num_bits_per_dimension + k] = micro_values[an_individual.prg[i].adr1][j * micro_params.num_bits_per_dimension + k];
				}

			// compute fitness of that micro chromosome
			// transform to base 10
			for (int j = 0; j < num_dimensions; j++)
				x[j] = binary_to_real(micro_values[i], micro_params.num_bits_per_dimension, min_x, max_x);

			micro_fitness[i][0] = f1(x, num_dimensions);// apply f - compute fitness of micro
			micro_fitness[i][1] = f2(x, num_dimensions);// apply f - compute fitness of micro

			break;

		}
	}

	// create a list with nondominated
	TLista nondominated;
	nondominated.Add(micro_fitness[0]);

	for (int i = 1; i < code_length; i++) {
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
	double hyper_volume = 0;
	for (node_double_linked * node_p = nondominated.head; node_p; node_p = node_p->next) {
		double *p = (double*)nondominated.GetCurrentInfo(node_p);
		double min_dist = euclidian_distance(p, reference_points[0]);
		for (int j = 1; j < num_reference_points; j++) {
			double dist = euclidian_distance(p, reference_points[j]);
			if (dist < min_dist)
				min_dist = dist;
		}
		hyper_volume += min_dist;
	}

	hyper_volume /= (double)nondominated.count;

	if (f) {
		fprintf(f, "%lf\n", hyper_volume);
		fprintf(f, "%d\n", nondominated.count);
		for (node_double_linked * node_p = nondominated.head; node_p; node_p = node_p->next) {
			double *p = (double*)nondominated.GetCurrentInfo(node_p);
			fprintf(f, "%lf %lf ", p[0], p[1]);
		}
		fprintf(f, "\n");
	}

	return hyper_volume;
}
//---------------------------------------------------------------------------
void compute_fitness(t_meta_gp_chromosome &an_individual, int code_length, t_micro_ea_parameters &micro_params, int num_dimensions, double min_x, double max_x, double** reference_points, int num_reference_points, double **micro_fitness, char **micro_values, double *x, FILE *f)
{
	double average_hypervolume = 0;  // average fitness over all runs
	// evaluate code
	for (int r = 0; r < micro_params.num_runs; r++) {// micro ea is run on multi runs
		srand(r);
		double hyper_volume = make_one_run(an_individual, code_length, micro_params, num_dimensions, min_x, max_x, reference_points, num_reference_points, micro_fitness, micro_values, x, f);
		// add to average
		average_hypervolume += hyper_volume;
	}


	average_hypervolume /= (double)micro_params.num_runs;

	an_individual.fitness = average_hypervolume;
}
//---------------------------------------------------------------------------
void start_steady_state_mep(t_meta_gp_parameters &meta_gp_params, t_micro_ea_parameters &micro_params, int num_dimensions, double min_x, double max_x, double** reference_points, int num_reference_points, FILE *f_out)       // Steady-State 
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
	double **micro_fitness = new double*[meta_gp_params.code_length]; // fitness for each micro EA chromosome
	for (int i = 0; i < meta_gp_params.code_length; i++)
		micro_fitness[i] = new double[2];

	// allocate some memory
	char **micro_values;
	micro_values = new char*[meta_gp_params.code_length];
	for (int i = 0; i < meta_gp_params.code_length; i++)
		micro_values[i] = new char[num_dimensions * micro_params.num_bits_per_dimension];


	
	// initialize
	for (int i = 0; i < meta_gp_params.pop_size; i++) {
		generate_random_meta_chromosome(population[i], meta_gp_params);
		compute_fitness(population[i], meta_gp_params.code_length, micro_params, num_dimensions, min_x, max_x, reference_points, num_reference_points, micro_fitness, micro_values, x, NULL);
	}
	// sort ascendingly by fitness
	qsort((void *)population, meta_gp_params.pop_size, sizeof(population[0]), sort_function);

	printf("generation %d, best fitness = %lf\n", 0, population[0].fitness);
	// print the front of the best
	fprintf(f_out, "%d\n", 0);
	srand(0);
	make_one_run(population[0], meta_gp_params.code_length, micro_params, num_dimensions, min_x, max_x, reference_points, num_reference_points, micro_fitness, micro_values, x, f_out);

	for (int g = 1; g < meta_gp_params.num_generations; g++) {// for each generation
		for (int k = 0; k < meta_gp_params.pop_size; k += 2) {
			// choose the parents using binary tournament
			int r1 = tournament_selection(population, meta_gp_params.pop_size, 2);
			int r2 = tournament_selection(population, meta_gp_params.pop_size, 2);
			// crossover
			double p = rand() / double(RAND_MAX);
			if (p < meta_gp_params.crossover_probability)
				one_cut_point_crossover(population[r1], population[r2], meta_gp_params, offspring1, offspring2);
			else {// no crossover so the offspring are a copy of the parents
				copy_individual(offspring1, population[r1], meta_gp_params);
				copy_individual(offspring2, population[r2], meta_gp_params);
			}
			// mutate the result and compute fitness
			mutate_meta_chromosome(offspring1, meta_gp_params);
			compute_fitness(offspring1, meta_gp_params.code_length, micro_params, num_dimensions, min_x, max_x, reference_points, num_reference_points, micro_fitness, micro_values, x, NULL);
			// mutate the other offspring and compute fitness
			mutate_meta_chromosome(offspring2, meta_gp_params);
			compute_fitness(offspring2, meta_gp_params.code_length, micro_params, num_dimensions, min_x, max_x, reference_points, num_reference_points, micro_fitness, micro_values, x, NULL);

			// replace the worst in the population
			if (offspring1.fitness < population[meta_gp_params.pop_size - 1].fitness) {
				copy_individual(population[meta_gp_params.pop_size - 1], offspring1, meta_gp_params);
				qsort((void *)population, meta_gp_params.pop_size, sizeof(population[0]), sort_function);
			}
			if (offspring2.fitness < population[meta_gp_params.pop_size - 1].fitness) {
				copy_individual(population[meta_gp_params.pop_size - 1], offspring2, meta_gp_params);
				qsort((void *)population, meta_gp_params.pop_size, sizeof(population[0]), sort_function);
			}
		}
		printf("generation %d, best fitness = %lf\n", g, population[0].fitness);
		// print the front of the best
		fprintf(f_out, "%d\n", g);
		srand(0);
		make_one_run(population[0], meta_gp_params.code_length, micro_params, num_dimensions, min_x, max_x, reference_points, num_reference_points, micro_fitness, micro_values, x, f_out);

	}
	// print best chromosome
	print_meta_chromosome(population[0], meta_gp_params.code_length);

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
bool read_reference_points(char *file_name, double **reference_points)
{
	FILE * f = fopen(file_name, "r");

	if (!f)
		return false;

	char c;
	for (int i = 0; i < 100; i++)
		fscanf(f, "%lf%c%lf", &reference_points[i][0], &c, &reference_points[i][1]);

	fclose(f);
	return true;
}
//--------------------------------------------------------------------
int main(void)
{
	t_meta_gp_parameters meta_gp_params;

	meta_gp_params.pop_size = 100;						    // the number of individuals in population  (must be an even number!)
	meta_gp_params.code_length = 1000;
	meta_gp_params.num_generations = 100;					// the number of generations
	meta_gp_params.mutation_probability = 0.01;              // mutation probability
	meta_gp_params.crossover_probability = 0.9;             // crossover probability

	t_micro_ea_parameters micro_ea_params;
	micro_ea_params.mutation_probability = 0.01;
	micro_ea_params.num_bits_per_dimension = 30;
	micro_ea_params.num_runs = 30;

	int num_reference_points = 100;

	double **reference_points = new double*[100];
	for (int i = 0; i < num_reference_points; i++)
		reference_points[i] = new double[2];

	if (!read_reference_points("c:\\Mihai\\Dropbox\\evolve-algorithms\\data\\zdt1_100.txt", reference_points)) {
		printf("Cannot read input file!");
		return 1;
	}

	FILE *f_out = fopen("c:\\temp\\zdt1_front.txt", "w");

	srand(0);
	start_steady_state_mep(meta_gp_params, micro_ea_params, 30, 0, 1, reference_points, num_reference_points, f_out);

	fclose(f_out);

	for (int i = 0; i < num_reference_points; i++)
		delete[] reference_points[i];
	delete[] reference_points;

	printf("Press enter ...");
	getchar();

	return 0;
}
//--------------------------------------------------------------------