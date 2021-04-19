#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#define N 1000
#define pi 3.141592
#define eps 0.00001

int* perThreads;
int* startingPoints;
int rank, threads;

double norm(const double *vec) {
    double threadRes = 0.0;

    for (int i = 0; i < perThreads[rank]; ++i) {
        threadRes += (vec[i] * vec[i]);
    }

    double res;
    MPI_Allreduce(&threadRes, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    res = sqrt(res);

    return res;
}

void mulByScalar(const double scalar, const double *vec, double *res) {
    for (int i = 0; i < perThreads[rank]; ++i) {
        res[i] = vec[i] * scalar;
    }
}

double scalarMul(const double* vec1, const double *vec2) {
    double threadRes = 0.0;

    for (int i = 0; i < perThreads[rank]; ++i) {
        threadRes += (vec1[i] * vec2[i]);
    }

    double res;
    MPI_Allreduce(&threadRes, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return res;
}

void initSplittingData() {
    perThreads = (int *) malloc(threads * sizeof(int));
    startingPoints = (int *) malloc(threads * sizeof(int));

    int k = N % threads;
    for (int i = 0; i < threads; ++i) {
        perThreads[i] = (i < k ? (N / threads) + 1 : N / threads);
        startingPoints[i] = (i == 0 ? 0 : startingPoints[i - 1] + perThreads[i - 1]);
    }
}

void initStartData(double *A, double *X, double *B, double *R, double *Z) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < perThreads[rank]; ++j) {
            A[i * perThreads[rank] + j] = (i == startingPoints[rank] + j ? 2.0 : 1.0);
        }
    }

    for (int i = 0; i < perThreads[rank]; ++i) {
        X[i] = 0.0;
        B[i] = 0.0;
    }

    for (int k = 0; k < threads; ++k) {
        for (int i = 0; i < perThreads[rank]; ++i) {
            double threadRes = 0.0;

            for (int j = 0; j < perThreads[rank]; ++j) {
                threadRes += (A[(i + startingPoints[k]) * perThreads[rank] + j]
                        * sin(2 * pi * (j + startingPoints[rank]) / N));
            }

            MPI_Reduce(&threadRes, &B[i], 1, MPI_DOUBLE, MPI_SUM, k, MPI_COMM_WORLD);
        }
    }

    double *Ax = (double *) malloc(N * sizeof(double ));

    for (int k = 0; k < threads; ++k) {
        for (int i = 0; i < perThreads[rank]; ++i) {
            double threadRes = 0.0;

            for (int j = 0; j < perThreads[rank]; ++j) {
                threadRes += (A[(i + startingPoints[k]) * perThreads[rank] + j] * X[j]);
            }

            MPI_Reduce(&threadRes, &Ax[i], 1, MPI_DOUBLE, MPI_SUM, k, MPI_COMM_WORLD);
        }
    }

    for (int i = 0; i < perThreads[rank]; ++i) {
        R[i] = B[i] - Ax[i];
        Z[i] = R[i];
    }
    free(Ax);
}

void iterativePart(const double *A, double *X, double *B, double *R, double *Z) {
    double *Az = (double *) malloc(perThreads[rank] * sizeof(double ));
    double *alphaZ = (double *) malloc(perThreads[rank] * sizeof(double ));
    double *alphaAz = (double *) malloc(perThreads[rank] * sizeof(double ));
    double *newR = (double *) malloc(perThreads[rank] * sizeof(double ));
    double *betaZ = (double *) malloc(perThreads[rank] * sizeof(double ));

    double normOfB = norm(B);

    while (norm(R) / normOfB >= eps) {
        //1.
        for (int k = 0; k < threads; ++k) {
            for (int i = 0; i < perThreads[rank]; ++i) {
                double threadRes = 0.0;

                for (int j = 0; j < perThreads[rank]; ++j) {
                    threadRes += (A[(i + startingPoints[k]) * perThreads[rank] + j] * Z[j]);
                }

                MPI_Reduce(&threadRes, &Az[i], 1, MPI_DOUBLE, MPI_SUM, k, MPI_COMM_WORLD);
            }
        }

        double alpha = scalarMul(R, R) / scalarMul(Az, Z);

        //2.

        mulByScalar(alpha, Z, alphaZ);

        for (int i = 0; i < perThreads[rank]; ++i) {
            X[i] += alphaZ[i];
        }

        //3.

        mulByScalar(alpha, Az, alphaAz);

        for (int i = 0; i < perThreads[rank]; ++i) {
            newR[i] = R[i] - alphaAz[i];
        }

        //4.

        double beta = scalarMul(newR, newR) / scalarMul(R, R);

        for (int i = 0; i < perThreads[rank]; ++i) {
            R[i] = newR[i];
        }

        //5.

        mulByScalar(beta, Z, betaZ);

        for (int i = 0; i < perThreads[rank]; ++i) {
            Z[i] = R[i] + betaZ[i];
        }
    }

    free(Az);
    free(alphaZ);
    free(alphaAz);
    free(newR);
    free(betaZ);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    initSplittingData();

    double *A = (double *) malloc(perThreads[rank] * N * sizeof(double ));
    double *X = (double *) malloc(perThreads[rank] * sizeof(double ));
    double *B = (double *) malloc(perThreads[rank] * sizeof(double ));
    double *R = (double *) malloc(perThreads[rank] * sizeof(double ));
    double *Z = (double *) malloc(perThreads[rank] * sizeof(double ));

    initStartData(A, X, B, R, Z);

    double start = MPI_Wtime();

    iterativePart(A, X, B, R, Z);

    if (rank == 0) {
        printf("%f ", MPI_Wtime() - start);
    }

    free(A);
    free(X);
    free(B);
    free(R);
    free(Z);

    MPI_Finalize();
    return 0;
}
