#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//Измерения. Нужно двумерное
#define NUM_DIMS (2)

void createColumn(MPI_Datatype *columns_type, const int columns_per_process, const int N, const int K) {
    MPI_Datatype column_type, column_resized_type;
    MPI_Type_vector(N, 1, K, MPI_DOUBLE, &column_type);
    MPI_Type_create_resized(column_type, 0, sizeof(double), &column_resized_type);
    MPI_Type_contiguous(columns_per_process, column_resized_type, columns_type);
    MPI_Type_commit(columns_type);
}

void createColumnCont(MPI_Datatype *columns_cont_type, const int columns_per_process, const int N) {
    MPI_Type_contiguous(N * columns_per_process, MPI_DOUBLE, columns_cont_type);
    MPI_Type_commit(columns_cont_type);
}

void createCell(MPI_Datatype *cell_resized_type,
                const int lines_per_process,
                const int columns_per_process,
                const int K) {
    MPI_Datatype cellC_type;
    MPI_Type_vector(lines_per_process, columns_per_process, K, MPI_DOUBLE, &cellC_type);
    MPI_Type_create_resized(cellC_type, 0, sizeof(double), cell_resized_type);

    MPI_Type_commit(cell_resized_type);
}

void multiply(const double *A,
              const double *B,
              double *C,
              const int M,
              const int N,
              const int K,
              const int size,
              const int rank) {
    MPI_Comm comm_2D;
    MPI_Comm comm_lines;
    MPI_Comm comm_columns;
    //Создаём декартову решётку.
    int dims[NUM_DIMS] = {0, 0};
    int periods[NUM_DIMS] = {0, 0};
    //Определение оптимального соотношения сторон
    MPI_Dims_create(size, NUM_DIMS, dims);
    //Непосредственное создание коммуникатора
    MPI_Cart_create(MPI_COMM_WORLD, NUM_DIMS, dims, periods, 0, &comm_2D);

    //Создаём коммуникаторы внутри решётки - линии, а так же столбцы
    int remain_lines[NUM_DIMS] = {0, 1};
    int remain_columns[NUM_DIMS] = {1, 0};
    MPI_Cart_sub(comm_2D, remain_lines, &comm_lines);
    MPI_Cart_sub(comm_2D, remain_columns, &comm_columns);

    //Присвоение локальных рангов и размеров в подгруппах
    int column_rank, column_size;
    int line_rank, line_size;

    MPI_Comm_rank(comm_lines, &line_rank);
    MPI_Comm_size(comm_lines, &line_size);

    MPI_Comm_rank(comm_columns, &column_rank);
    MPI_Comm_size(comm_columns, &column_size);

    if (rank == 0) {
        printf("Size: %dx%d\n", line_size, column_size);
    }
    //Технически-очевидные моменты
    int lines_per_process = M / line_size;
    int columns_per_process = K / column_size;
    int line_elements_per_process = lines_per_process * N;
    int column_elements_per_process = columns_per_process * N;

    double *lines_A = (double *) malloc(line_elements_per_process * sizeof(double));
    double *columns_B = (double *) malloc(column_elements_per_process * sizeof(double));
    double *result = (double *) malloc(lines_per_process * columns_per_process * sizeof(double));

    //Типы данных для корректного разрезания матрицы
    MPI_Datatype columns_type, columns_contiguous_type, cell_type;
    //Создаём нужные типы данных для столбцов
    createColumn(&columns_type, columns_per_process, N, K);

    createColumnCont(&columns_contiguous_type, columns_per_process, N);

    //Создаём "клетки" - для упрощения сбора данных в финале вычислений
    createCell(&cell_type, lines_per_process, columns_per_process, K);

    int *recvcounts = NULL;
    int *displs = NULL;

    double start, end;

    //Начало умножениея
    start = MPI_Wtime();
    //Шаг 1.1 - скаттер матрицы по 1-й строке
    if (column_rank == 0) {
        MPI_Scatter(A,
                    line_elements_per_process,
                    MPI_DOUBLE,
                    lines_A,
                    line_elements_per_process,
                    MPI_DOUBLE,
                    0,
                    comm_lines);
    }

    //Шаг 1.2 - скаттер матрицы по  1-му столбцу
    if (line_rank == 0) {
        MPI_Scatter(B, 1, columns_type, columns_B, 1, columns_contiguous_type, 0, comm_columns);
    }

    MPI_Type_free(&columns_type);

    //Шаг 2 - раздача матрицы в линиях и столбцах
    MPI_Bcast(columns_B, 1, columns_contiguous_type, 0, comm_lines);
    MPI_Bcast(lines_A, line_elements_per_process, MPI_DOUBLE, 0, comm_columns);

    MPI_Type_free(&columns_contiguous_type);

    //Раздача завершена.  Умножение матрицы.
    for (int i = 0; i < lines_per_process; ++i) {
        for (int j = 0; j < columns_per_process; ++j) {
            result[i * columns_per_process + j] = 0.0;
            for (int k = 0; k < N; ++k) {
                result[i * columns_per_process + j] += lines_A[i * N + k] * columns_B[j * N + k];
            }
        }
    }


    //Высчитываем данные для сбора данных в 0-м процессе.
    if (rank == 0) {
        //Поскольку MPI_Gatherv собирает в указанном процессе, то нет смысла держать эти данные в иных процессах.
        recvcounts = (int *) malloc(line_size * column_size * sizeof(int));
        displs = (int *) malloc(line_size * column_size * sizeof(int));

        for (int j = 0; j < column_size; ++j) {
            for (int i = 0; i < line_size; ++i) {
                recvcounts[j * line_size + i] = 1;
                displs[j * line_size + i] = j * columns_per_process + i * K * lines_per_process;
            }
        }
    }

    //Сборка данных в главном процессе.
    MPI_Gatherv(result,
                columns_per_process * lines_per_process,
                MPI_DOUBLE,
                C,
                recvcounts,
                displs,
                cell_type,
                0,
                comm_2D);
    MPI_Type_free(&cell_type);
    end = MPI_Wtime();
    //Конец умножения

    if (rank == 0) {
        printf("\n--------- TIME TAKEN --------\n");
        printf("%f sec.\n", end - start);
    }

    free(lines_A);
    free(columns_B);
    free(result);
    free(recvcounts);
    free(displs);
}

int main(int argc, char **argv) {

    int M = 2000;
    int N = 1000;
    int K = 3000;
    //A is M*N, B is N*K, C is M*K

    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("\nProcesses: %d\n", size);
        printf("Size A: %d %d\nSize B: %d %d\n", M, N, N, K);
    }

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;

    if (rank == 0) {
        A = (double *) malloc(M * N * sizeof(double));
        B = (double *) malloc(N * K * sizeof(double));
        C = (double *) malloc(M * K * sizeof(double));

        for (int i = 0; i < M * N; ++i) {
            A[i] = (double) rand() / RAND_MAX * 300 - 150;
        }

        for (int i = 0; i < N * K; ++i) {
            B[i] = (double) rand() / RAND_MAX * 300 - 150;
        }

    }

    multiply(A, B, C, M, N, K, size, rank);

    free(A);
    free(B);
    free(C);

    MPI_Finalize();

    return 0;
}
