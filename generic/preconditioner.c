/* preconditioner.c
	An implementation of the bounding-box constrained K-means clustering algorithmn of Perini et al (2013)
*/

/*
Included from preconditioner

NUM_CLUSTER_VARS -> the number of variables to cluster
CLUSTER_VAR_INDEXES -> the indexes in the mechanism of the cluster variables (the first should be 0, Temperature)
B_NUM -> 2^NUM_CLUSTER_VARS
*/
#include "preconditioner.h"
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

// OpenMP
#ifdef _OPENMP
	#include <omp.h>
#else
	#define omp_get_max_threads() 1
 	#define omp_get_num_threads() 1
	#define omp_get_thread_num() 0
	#define omp_set_num_threads(dummy) 0
#endif


#define E_T (20)
#define s_y (4) //include zero and one (i.e. 1/e_y + 1)
#define e_y (0.25)
#define iter_max (50)
#define y_min (1e-9)
#define cutoff (0.05)

static inline void print(const double* y_vals, const bool* is_on)
{
	for (int i = 0; i < NUM_CLUSTER_VARS; i++)
	{
		if (is_on[i])
		{
			if (i)
				printf(", ");
			printf("%le", y_vals[i]);
		}
	}
	printf("\n");
}

static inline double get_distance(const double* center, const double* point, const bool* is_on)
{

	double sum = 0;
	for (int i = 0; i < NUM_CLUSTER_VARS; ++i)
	{
		if (is_on[i])
			sum += fabs(center[i] - point[i]);
	}
	return sum;
}

static inline int populate_indexes(const int index, const double* y_norm, const double* mins, const double* maxs, const bool* is_on, const double e_t, int* index_list)
{
	//do the rest
	for(int var = 0; var < NUM_CLUSTER_VARS - 1; ++var)
	{
		if (is_on[var])
		{
			index_list[var * 2] = floor(y_norm[index + var] * s_y);
			if (index_list[var * 2] == s_y)
				index_list[var * 2] -= 1;
			index_list[var * 2 + 1] = index_list[var * 2] + 1;
		}
	}
	//do Temperature
	index_list[2 * (NUM_CLUSTER_VARS - 1)] = floor(y_norm[index] / e_t);
	index_list[(2 * NUM_CLUSTER_VARS) - 1] = index_list[2 * (NUM_CLUSTER_VARS - 1)] + 1;
}

//from the binary representation of 'num', this function quickly determines the appropriate cluster center as a permutation of the index_list
static inline int get_cluster_index(const int num, const int* index_list, const bool* is_on)
{
	int cluster_center = 0;
	int count = 0;
	int center_count = 1;
	for (int i = 1; i < B_NUM; i <<= 1)
	{
		if (is_on[count / 2])
		{
			cluster_center += center_count * ((num & i) ? index_list[count + 1] : index_list[count]);
			//the last one will be incorrect, but it isn't used again so who cares
			center_count *= (s_y + 1);
		}
		count+=2;
	}
	return cluster_center;
}

static inline int get_cluster(const int index, const double* y_norm, const double* cluster_centers, const double* mins, const double* maxs, const bool* is_on, const double e_t, int* index_list)
{
	int thread = omp_get_thread_num();
	//first get all indexes
	populate_indexes(index, y_norm, mins, maxs, is_on, e_t, index_list);

	//printf("\t\t\t");
	//print(&y_norm[index], is_on);
	
	double min_dist = -1;
	int min_c = -1;
	for (int i = 0; i < B_NUM; i++)
	{
		int cindex = get_cluster_index(i, index_list, is_on);
		double dist = get_distance(&cluster_centers[cindex * NUM_CLUSTER_VARS], &y_norm[index], is_on);
		//printf("%le\t\t", dist);
		//print(&cluster_centers[cindex * NUM_CLUSTER_VARS], is_on);
		if (dist < min_dist || min_c == -1)
		{
			min_dist = dist;
			min_c = cindex;
		}
	}
	return min_c;
}

static void initialize(const double* mins, const double* maxs, const bool* is_on, int** cluster_count, double** cluster_centers, int* num_centers_out, double* e_t_out)
{
	int num_temperature_centers = (int)fmin((int)round(1 + (maxs[NUM_CLUSTER_VARS - 1] - mins[NUM_CLUSTER_VARS - 1]) / E_T), 500);
	double e_t = 1.0 / (num_temperature_centers - 1);
	int num_centers = 1;
	int lengths[NUM_CLUSTER_VARS] = {1};
	for (int i = 0; i < NUM_CLUSTER_VARS; ++i)
	{
		if (!is_on[i])
			continue;
		lengths[i] = num_centers;
		num_centers *= ((i != NUM_CLUSTER_VARS - 1) ? (s_y + 1) : num_temperature_centers);
	}

	//malloc memory
	(*cluster_count) = (int*)malloc(num_centers * sizeof(int));
	(*cluster_centers) = (double*)malloc(num_centers * NUM_CLUSTER_VARS * sizeof(double));
	double* ccenter = *cluster_centers;
	int* ccount = *cluster_count;

	double sample_cluster[NUM_CLUSTER_VARS] = {0};
	#pragma omp parallel for shared(ccount, ccenter) private(sample_cluster)
	for (int tid = 0; tid < num_centers; ++tid)
	{
		ccount[tid] = 1;
		//figure out the sample cluster
		for (int i = 0; i < NUM_CLUSTER_VARS - 1; ++i)
		{
			sample_cluster[i] = e_y * ((tid / lengths[i]) % (s_y + 1));
		}
		sample_cluster[NUM_CLUSTER_VARS - 1] = e_t * (tid / lengths[NUM_CLUSTER_VARS - 1]);
		memcpy(&ccenter[tid * NUM_CLUSTER_VARS], sample_cluster, NUM_CLUSTER_VARS * sizeof(double));
	}

	/*
	double sample_cluster[NUM_CLUSTER_VARS] = {0};
	/*memcpy(ccenter, sample_cluster, NUM_CLUSTER_VARS * sizeof(double));
	ccount[0] = 1;
	for (int tid = 1; tid < num_centers; ++tid)
	{	
		//set the count
		ccount[tid] = 1;
		//update the sample cluster
		for (int i = 0; i < NUM_CLUSTER_VARS; i++)
		{
			//reset if gone through all
			if (i != NUM_CLUSTER_VARS - 1 && tid % lengths[i + 1] == 0)
			{
				sample_cluster[i] = 0;
			}
			//add if reached end of row
			else if (tid % lengths[i] == 0)
			{
				sample_cluster[i] += (i != NUM_CLUSTER_VARS - 1 ? e_y : e_t);
			}
		}
		memcpy(&ccenter[tid * NUM_CLUSTER_VARS], sample_cluster, NUM_CLUSTER_VARS * sizeof(double));
	}*/

	*num_centers_out = num_centers;
	*e_t_out = e_t;
}

static void get_range(const int num_threads, const int NUM, const int padded, const double* y_host, double* mins, double* maxs, bool* is_on)
{
	double* t_mins = (double*)malloc(NUM_CLUSTER_VARS * num_threads * sizeof(double));
	double* t_maxs = (double*)malloc(NUM_CLUSTER_VARS * num_threads * sizeof(double));
	for (int i = 0; i < NUM_CLUSTER_VARS; ++i)
	{
		for (int j = 0; j < num_threads; j++)
		{
			t_mins[j * NUM_CLUSTER_VARS + i] = 1e6;
			t_maxs[j * NUM_CLUSTER_VARS + i] = -1;
		}
	}

	int index = 0, var = 0;
	//find min/max of the various clustering variables
	#pragma omp parallel for shared(y_host, t_mins, t_maxs) private(index, var)
	for (int tid = 0; tid < NUM; ++tid) {
		index = omp_get_thread_num();
		for (var = 0; var < NUM_CLUSTER_VARS; var++)
		{
			assert(index == omp_get_thread_num());
			t_mins[index * NUM_CLUSTER_VARS + var] = fmin(t_mins[index * NUM_CLUSTER_VARS + var], y_host[CLUSTER_VAR_INDEXES[var] * padded + tid]);
			t_maxs[index * NUM_CLUSTER_VARS + var] = fmax(t_maxs[index * NUM_CLUSTER_VARS + var], y_host[CLUSTER_VAR_INDEXES[var] * padded + tid]);
		}
	}

	//reduce and count how many clusters are on
	for (int i = 0; i < num_threads; ++i)
	{
		for (int j = 0; j < NUM_CLUSTER_VARS; j++)
		{
			mins[j] = fmin(mins[j], t_mins[i * NUM_CLUSTER_VARS + j]);
			maxs[j] = fmax(maxs[j], t_maxs[i * NUM_CLUSTER_VARS + j]);
			if (!is_on[j] && mins[j] != maxs[j] && (maxs[j] > y_min))
			{
				is_on[j] = true;
			}
		}
	}
	free(t_mins);
	free(t_maxs);
}

static void normalize(const int NUM, const int padded, const double* y_host, const double* mins, const double* maxs, const bool* is_on, double* y_norm)
{
	#pragma omp parallel for shared(y_host, y_norm, mins, maxs, is_on)
	for (int tid = 0; tid < NUM; ++tid) 
	{
		for (int var = 0; var < NUM_CLUSTER_VARS; var++)
		{
			if (!is_on[var])
				continue;
			y_norm[var + tid * NUM_CLUSTER_VARS]  = (y_host[CLUSTER_VAR_INDEXES[var] * padded + tid] - mins[var]) / (maxs[var] - mins[var]);
		}
	}
}

void precondition(const int NUM, const int padded, double* y_host, double* pr_global, int* mask)
{
	int num_threads = omp_get_max_threads();
	omp_set_num_threads(num_threads);

	//get range of values
	bool is_on[NUM_CLUSTER_VARS] = {[0 ... NUM_CLUSTER_VARS - 2] = false, [NUM_CLUSTER_VARS - 1] = true};
	double mins[NUM_CLUSTER_VARS] = {[0 ... NUM_CLUSTER_VARS - 1] = 1e6};
	double maxs[NUM_CLUSTER_VARS] = {[0 ... NUM_CLUSTER_VARS - 1] = -1};
	get_range(num_threads, NUM, padded, y_host, mins, maxs, is_on);

	//create normalized copy
	//note y_norm stored in a simpler manner for CPU work (i.e. not strided)
	double* y_norm = (double*)malloc(NUM_CLUSTER_VARS * NUM * sizeof(double));
	normalize(NUM, padded, y_host, mins, maxs, is_on, y_norm);

	//initialize cluster centers
	double e_t = 0;
	int num_centers = 0;
	double* cluster_centers = 0;
	double* cluster_centers_new = 0;
	int* cluster_count;
	initialize(mins, maxs, is_on, &cluster_count, &cluster_centers, &num_centers, &e_t);
	cluster_centers_new = (double*)malloc(num_centers * NUM_CLUSTER_VARS * sizeof(double));
	memcpy(cluster_centers_new, cluster_centers, num_centers * NUM_CLUSTER_VARS * sizeof(double));

	//workspace for storing indexes
	int workspace[2 * num_threads * NUM_CLUSTER_VARS];
	int t_swaps[num_threads];

	int* cluster_assignment = (int*)malloc(NUM * sizeof(int));
	for (int i = 0; i < NUM; ++i)
	{
		cluster_assignment[i] = -1;
	}
	int iter = 0;
	do
	{
		//reset swap count
		memset(t_swaps, 0, num_threads * sizeof(int));
		#pragma omp parallel for shared(y_norm, cluster_centers, cluster_centers_new, cluster_count, mins, maxs, is_on, e_t)
		for (int tid = 0; tid < NUM; ++tid)
		{
			double* y = &y_norm[tid * NUM_CLUSTER_VARS];
			//get new and old centers
			int center = get_cluster(tid * NUM_CLUSTER_VARS, y_norm, cluster_centers, mins, maxs, is_on, e_t, &workspace[omp_get_thread_num() * 2 * NUM_CLUSTER_VARS]);
			int old_center = cluster_assignment[tid];
			if (center != old_center)
			{
				//if old center was set, need to update
				if (old_center != -1)
				{
					
					double* old = &cluster_centers_new[old_center * NUM_CLUSTER_VARS];
					for (int i = 0; i < NUM_CLUSTER_VARS; i++)
					{
						if (is_on[i])
						{
							#pragma omp atomic
							old[i] -= y[i];
						}
					}
					#pragma omp atomic
					cluster_count[old_center]--;
				}

				//update new center
				double* newc = &cluster_centers_new[center * NUM_CLUSTER_VARS];
				for (int i = 0; i < NUM_CLUSTER_VARS; i++)
				{
					if (is_on[i])
					{
						#pragma omp atomic
						newc[i] += y[i];
					}
				}
				#pragma omp atomic
				cluster_count[center]++;

				cluster_assignment[tid] = center;
				t_swaps[omp_get_thread_num()]++;
			}
		}

		//check swap count
		int swaps = 0;
		for (int i = 0; i < num_threads; i++)
		{
			swaps += t_swaps[i];
		}

		//if an iteration has completed, and no swaps
		if (iter && swaps < cutoff * NUM)
			break;

		if (iter + 1 < iter_max)
		{
			//if we're continuing, we need to copy back
			#pragma omp parallel for shared(cluster_centers, cluster_centers_new, cluster_count)
			for (int tid = 0; tid < num_centers; ++tid)
			{
				if (cluster_count[tid] > 1)
				{
					double ratio = 1.0 / ((double)cluster_count[tid]);
					int offset = tid * NUM_CLUSTER_VARS;
					for (int i = 0; i < NUM_CLUSTER_VARS; ++i)
					{
						cluster_centers[offset + i] = cluster_centers_new[offset + i] * ratio;
					}
				}
				else
				{
					assert(cluster_count[tid] > 0);
					memcpy(&cluster_centers[tid * NUM_CLUSTER_VARS], &cluster_centers_new[tid * NUM_CLUSTER_VARS], NUM_CLUSTER_VARS * sizeof(double));
				}
			}
		}
	} while (iter++ < iter_max);

	//now group
	for (int tid = 0; tid < num_centers; tid++)
	{
		if(cluster_count[tid] > 1)
		{
			printf("%d\t%d\t", tid, cluster_count[tid]);
			print(&cluster_centers[tid * NUM_CLUSTER_VARS], is_on);
		}
	}
	free(cluster_centers);
	free(cluster_centers_new);
	free(cluster_assignment);
	free(cluster_count);
}