/*
Copyright (c) 2016-2020 Jeremy Iverson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

how to run the program with gcc & MPI
1. mpicc -std=c99 -o MPI-collective-comunications MPI-collective-comunications.c
2. mpiexec -n 4 ./MPI-collective-comunications ML-ratings-medium.txt  //Parameters are the number of processors,the program name, and textfile ...

how to run the program with gcc, MPI, and OpenMP
1. mpicc -fopenmp -std=c99 -o MPI-collective-comunications.edited MPI-collective-comunications-edited.c
2. ./MPI-collective-comunications.edited ML-ratings-medium.

mpicc -fopenmp -O2 -std=c99 -o MPI-collective-comunications.edited MPI-collective-comunications-edited.c
 mpiexec -n 8 ./MPI-collective-comunications.edited ML-ratings-xlarge.txt


/* assert */
#include <assert.h>

/* fabs */
#include <math.h>

/* MPI API */
#include <mpi.h>

/* OMP API */
#include <omp.h>

/* printf, fopen, fclose, fscanf, scanf */
#include <stdio.h>

/* EXIT_SUCCESS, malloc, calloc, free, qsort */
#include <stdlib.h>

#define MPI_SIZE_T MPI_UNSIGNED_LONG

struct distance_metric {
  size_t viewer_id;
  double distance;
};

static int
cmp(void const *ap, void const *bp)
{
  struct distance_metric const a = *(struct distance_metric*)ap;
  struct distance_metric const b = *(struct distance_metric*)bp;

  return a.distance < b.distance ? -1 : 1;
}

int
main(int argc, char * argv[])
{
  int ret, p, rank;
  size_t n, m, k;
  double * rating;

  /* Initialize MPI environment. */
  ret = MPI_Init(&argc, &argv);
  assert(MPI_SUCCESS == ret);

  /* Get size of world communicator. */
  ret = MPI_Comm_size(MPI_COMM_WORLD, &p);
  assert(ret == MPI_SUCCESS);

  /* Get my rank. */
  ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(ret == MPI_SUCCESS);

  /* Validate command line arguments. */
  assert(2 == argc);

  /* Read input --- only if your rank 0. */
  if (0 == rank) {
    /* ... */
    char const * const fn = argv[1];

    /* Validate input. */
    assert(fn);

    /* Open file. */
    FILE * const fp = fopen(fn, "r");
    assert(fp);

    /* Read file. */
    fscanf(fp, "%zu %zu", &n, &m);

    /* Allocate memory. */
    rating = malloc(n * m * sizeof(*rating));

    /* Check for success. */
    assert(rating);

    for (size_t i = 0; i < n; i++) {
      for (size_t j = 0; j < m; j++) {
        fscanf(fp, "%lf", &rating[i * m + j]);
      }
    }

    /* Close file. */
    ret = fclose(fp);
    assert(!ret);
  }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /* Using MPI_Bcast funtion to send n & m to all other threads. */
  ret = MPI_Bcast(&n, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
  assert(MPI_SUCCESS == ret);
  ret = MPI_Bcast(&m, 1, MPI_SIZE_T, 0, MPI_COMM_WORLD);
  assert(MPI_SUCCESS == ret);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* Compute base number of viewers. */
size_t const base = 1 + ((n - 1) / p); // ceil(n / p)
/* Compute local number of viewers. */
size_t const ln = (rank + 1) * base > n ? n - rank * base : base;

/* creating size of rating for all other threads */
if (rank != 0){
  /* Allocate memory for all other threads besides thread 0*/
  rating = malloc(ln * m * sizeof(*rating));
  assert(rating);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  int *sendcount;
  int *displs;
  /* Send viewer data to rest of processes. */
  if (0 == rank) {
    sendcount = malloc(p* sizeof(*sendcount));
      assert(sendcount);
    displs = malloc(p* sizeof(*displs));
      assert(displs);

    for (int r = 0; r < p; r++) {
      size_t const rn = (r + 1) * base > n ? n - r * base : base;
      sendcount[r] = rn * m;
      displs[r] = r * base * m;
    }
  }


  /* Trying to scatter the rating data to all other threads */
  /* This is for the simple version -- I think meaning */
  /* Ultimatly we want to use scatterv because we want to sent different abouts of data to threads */
  // MPI_Scatter(rating, base * m, MPI_DOUBLE, rating, base * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  //send count would be rn * m...
  ret = MPI_Scatterv(rating, sendcount, displs, MPI_DOUBLE, rating, ln * m, MPI_DOUBLE, 0, MPI_COMM_WORLD);;
  ret = (MPI_SUCCESS == ret);

  if (rank == 0){
    free(sendcount);
    free(displs);
  }




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double * const urating = malloc((m - 1) * sizeof(*urating));
assert(urating);
/* Get user input and send it to rest of processes. */
if (0 == rank)
{
  fflush(stdout);
  for (size_t j = 0; j < m - 1; j++) {
    printf("Enter your rating for movie %zu: ", j + 1);
    fflush(stdout);
    scanf("%lf", &urating[j]);
  }
  ret = MPI_Bcast(urating, m - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  assert(MPI_SUCCESS == ret);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /* Allocate more memory. */
  double *  distance = calloc(n, sizeof(*distance));

  /* Check for success. */
  assert(distance);
  /* every thread has a different tiume */
  double ts = omp_get_wtime();
  /* Compute distances. */
  for (size_t i = 0; i < ln; i++) {
    for (size_t j = 0; j < m - 1; j++) {
      distance[i] += fabs(urating[j] - rating[i * m + j]);
    }
  }
  double te = omp_get_wtime();
  double local_elapsed = te - ts;
  double global_elapsed;
  // how to get the max time (te -ts)
  // finding the max time value for all the threads...
  ret = MPI_Reduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  assert(MPI_SUCCESS == ret);

  int * sendcount2;
  int * displs2;
  double * newDistance;
  /* Send viewer data to rest of processes. */
  if (0 == rank) {
    sendcount2 = malloc(p* sizeof(*sendcount2));
      assert(sendcount2);
    displs2 = malloc(p* sizeof(*displs2));
      assert(displs2);
    newDistance = malloc(n* sizeof(*newDistance));

    for (int r = 0; r < p; r++)
    {
      size_t const rn = (r + 1) * base > n ? n - r * base : base;
      sendcount2[r] = rn;
      displs2[r] = r * base;
    }
  }

  MPI_Gatherv(distance, ln, MPI_DOUBLE, newDistance, sendcount2, displs2, MPI_DOUBLE, 0, MPI_COMM_WORLD);



  if(0 == rank){
    free(sendcount2);
    free(displs2);
    printf("Elapsed time: %f\n", global_elapsed);

    // creating distance metric for final data
    struct distance_metric * final = malloc(n * sizeof(*final));
    assert(final);

    //giving the distance metric its values..
    for (size_t k = 0; k < n; k++){
      final[k].viewer_id = k;
      final[k].distance = newDistance[k];
    }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /* Sort distances. */
  qsort(distance, n, sizeof(*distance), cmp);
  fflush(stdout);
  /* Get user input. */
  printf("Enter the number of similar viewers to report: ");
  fflush(stdout);
  scanf("%zu", &k);

  /* Output k viewers who are least different from the user. */
  fflush(stdout);
  printf("Viewer ID   Movie five   Distance\n");
  fflush(stdout);
  printf("---------------------------------\n");

  for (size_t i = 0; i < k; i++) {
    printf("%9zu   %10.1lf   %8.1lf\n", final[i].viewer_id + 1,
      rating[final[i].viewer_id * m + 4], final[i].distance);
  }

  fflush(stdout);
  printf("---------------------------------\n");
  fflush(stdout);

  /* Compute the average to make the prediction. */
  double sum = 0.0;
  for (size_t i = 0; i < k; i++) {
    sum += rating[final[i].viewer_id * m + 4];
  }

  /* Output prediction. */
  fflush(stdout);
  printf("The predicted rating for movie five is %.1lf.\n", sum / k);
  fflush(stdout);

  /* Deallocate memory. */
  free(rating);
  free(urating);
  free(final);

}
  ret = MPI_Finalize();
  assert(MPI_SUCCESS == ret);

  return EXIT_SUCCESS;

}
