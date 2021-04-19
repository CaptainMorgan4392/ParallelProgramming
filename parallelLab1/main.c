#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 10000
#define pi 3.141592
#define eps 0.00001

static inline double norm(const double *vec) {
    double res = 0.0;

    for (int i = 0; i < N; ++i) {
        res += (vec[i] * vec[i]);
    }
    res = sqrt(res);

    return res;
}

static inline void mulByScalar(const double scalar, const double *vec, double *res) {
    for (int i = 0; i < N; ++i) {
        res[i] = vec[i] * scalar;
    }
}

static inline double scalarMul(const double *vec1, const double *vec2) {
    double res = 0.0;

    for (int i = 0; i < N; ++i) {
        res += (vec1[i] * vec2[i]);
    }

    return res;
}

int main() {
    //1. Create data

    double *A = (double *)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = (i == j ? 2.0 : 1.0);
        }
    }

    double *X = (double *) malloc(N * sizeof(double ));
    double *B = (double *) malloc(N * sizeof(double ));

    for (int i = 0; i < N; ++i) {
        X[i] = 0.0;

        B[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            B[i] += (A[i * N + j] * sin(2 * pi * j / N));
        }
    }

    //2. Init steps of algorithm

    double *R = (double *) malloc(N * sizeof(double ));
    double *Z = (double *) malloc(N * sizeof(double ));
    double *Ax = (double *) malloc(N * sizeof(double ));

    for (int i = 0; i < N; ++i) {
        Ax[i] = 0.0;

        for (int j = 0; j < N; ++j) {
            Ax[i] += (A[i * N + j] * X[j]);
        }

        R[i] = B[i] - Ax[i];
        Z[i] = R[i];
    }

    //3. Iterating part
    const double normOfB = norm(B);

    double *Az = (double *) malloc(N * sizeof(double ));
    double *alphaZ = (double *) malloc(N * sizeof(double ));
    double *alphaAz = (double *) malloc(N * sizeof(double ));
    double *newR = (double *) malloc(N * sizeof(double ));
    double *betaZ = (double *) malloc(N * sizeof(double ));

    for (int k = 0; k < 10; ++k) {

        //1.
        for (int i = 0; i < N; ++i) {
            Az[i] = 0.0;

            for (int j = 0; j < N; ++j) {
                Az[i] += (A[i * N + j] * Z[j]);
            }
        }

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

    free(alphaZ);
    free(alphaAz);
    free(Az);
    free(newR);
    free(betaZ);

    //Deallocating
    free(A);
    free(X);
    free(B);
    free(Z);
    free(R);

    return 0;
}
