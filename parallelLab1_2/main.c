#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#define N 25000
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
    double *buffer = (double *) malloc(perThreads[rank] * sizeof(double ));

    for (int i = 0; i < perThreads[rank]; ++i) {
        buffer[i] = vec[i] * scalar;
    }

    MPI_Allgatherv(buffer, perThreads[rank], MPI_DOUBLE, res,
                   perThreads, startingPoints, MPI_DOUBLE, MPI_COMM_WORLD);
}

double scalarMul(const double* vec1, const double *vec2) {
    double threadRes = 0.0;

    for (int i = 0; i < perThreads[rank]; ++i) {
        threadRes += (vec1[i] * vec1[i]);
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
    double *bufferX = (double *) malloc(perThreads[rank] * sizeof(double ));
    for (int i = 0; i < perThreads[rank]; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = ((i + startingPoints[rank]) == j ? 2.0 : 1.0);
        }

        bufferX[i] = 0.0;
    }
    MPI_Allgatherv(bufferX, perThreads[rank], MPI_DOUBLE, X,
                   perThreads, startingPoints, MPI_DOUBLE, MPI_COMM_WORLD);
    free(bufferX);

    double *bufferB = (double *) malloc(perThreads[rank] * sizeof(double ));
    for (int i = 0; i < perThreads[rank]; ++i) {
        bufferB[i] = 0.0;

        for (int j = 0; j < N; ++j) {
            bufferB[i] += (A[i * N + j] * sin(2 * pi * j / N));
        }
    }
    MPI_Allgatherv(bufferB, perThreads[rank], MPI_DOUBLE, B,
                   perThreads, startingPoints, MPI_DOUBLE, MPI_COMM_WORLD);
    free(bufferB);

    double *Ax = (double *) malloc(N * sizeof(double ));
    double *bufferAx = (double *) malloc(perThreads[rank] * sizeof(double ));
    for (int i = 0; i < perThreads[rank]; ++i) {
        bufferAx[i] = 0.0;

        for (int j = 0; j < N; ++j) {
            bufferAx[i] += (A[i * N + j] * X[j]);
        }
    }
    MPI_Allgatherv(bufferAx, perThreads[rank], MPI_DOUBLE, Ax,
                   perThreads, startingPoints, MPI_DOUBLE, MPI_COMM_WORLD);
    free(bufferAx);


    for (int i = 0; i < N; ++i) {
        R[i] = B[i] - Ax[i];
        Z[i] = R[i];
    }
    free(Ax);
}

void iterativePart(const double *A, double *X, double *B, double *R, double *Z) {
    double *Az = (double *) malloc(N * sizeof(double ));
    double *alphaZ = (double *) malloc(N * sizeof(double ));
    double *alphaAz = (double *) malloc(N * sizeof(double ));
    double *newR = (double *) malloc(N * sizeof(double ));
    double *betaZ = (double *) malloc(N * sizeof(double ));

    double normOfB = norm(B);

    while (norm(R) / normOfB >= eps) {
        //1.
        double *bufferAz = (double *) malloc(perThreads[rank] * sizeof(double ));
        for (int i = 0; i < perThreads[rank]; ++i) {
            bufferAz[i] = 0.0;

            for (int j = 0; j < N; ++j) {
                bufferAz[i] += (A[i * N + j] * Z[j]);
            }
        }
        MPI_Allgatherv(bufferAz, perThreads[rank], MPI_DOUBLE, Az,
                       perThreads, startingPoints, MPI_DOUBLE, MPI_COMM_WORLD);
        free(bufferAz);

        double alpha = scalarMul(R, R) / scalarMul(Az, Z);

        //2.

        mulByScalar(alpha, Z, alphaZ);

        for (int i = 0; i < N; ++i) {
            X[i] += alphaZ[i];
        }

        //3.

        mulByScalar(alpha, Az, alphaAz);

        for (int i = 0; i < N; ++i) {
            newR[i] = R[i] - alphaAz[i];
        }

        //4.

        double beta = scalarMul(newR, newR) / scalarMul(R, R);

        for (int i = 0; i < N; ++i) {
            R[i] = newR[i];
        }

        //5.

        mulByScalar(beta, Z, betaZ);

        for (int i = 0; i < N; ++i) {
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
    double *X = (double *) malloc(N * sizeof(double ));
    double *B = (double *) malloc(N * sizeof(double ));
    double *R = (double *) malloc(N * sizeof(double ));
    double *Z = (double *) malloc(N * sizeof(double ));

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
