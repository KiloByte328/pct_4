#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// макрос на получение количества элементов
#define NELEMS(x) (sizeof((x)) / sizeof((x)[0]))
// макрос на получение индекса в решётке
#define IND(i, j) ((i) * (nx + 2) + (j))


// находим блок для процесса, если остаток от деления количества на все процессы больше чем номер нашего процесса, то он получает на 1 больше
int get_block_size(int n, int rank, int commsize)
{
    int s = n / commsize;
    return n % commsize > rank ? s : s++;
}

// получаем количество элементов распределеных между процессами
int get_sum_of_prev_blocks(int n, int rank, int commsize)
{
    int rem = n % commsize;
    return n / commsize * rank + ((rank >= rem) ? rem : rank);
}

int main(int argc, char *argv[])
{
    double EPS = 0.001;
    double PI = 3.14159265358979323846;
    int commsize, rank;
    MPI_Init(&argc, &argv);
    double ttotal = -MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // создаём 2д решётку: commsize = px * py; periodic = 0, потому что наша решётка не должна быть периодична, иначе это ломает логику, т.к. у нас всё ограниченно sin
    MPI_Comm cartcomm;
    int dims[2] = {0, 0}, periodic[2] = {0, 0};
    MPI_Dims_create(commsize, 2, dims);
    int px = dims[0];
    int py = dims[1];

    if (px < 2 || py < 2) {
        fprintf(stderr, "Invalid number of processes %d: px %d and py %d must be greater than 1\n",
                commsize, px, py);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // создаём коммуникатор cartcomm, который не периодичен, т.е все обращения вне границ будут обращаться в никуда?
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &cartcomm);
    int coords[2];
    MPI_Cart_coords(cartcomm, rank, 2, coords);
    int rankx = coords[0];
    int ranky = coords[1];

    int rows, cols;

    // проверка на то, что количество строк/столбцов не меньше количества процессов, иначе это ломает логику работы программы
    //а если всё ок, то мы раздаём всем остальным процессам колчества строк и столбцов для дальнейшей инициализации
    if (rank == 0) {
        rows = 1000;
        cols = rows;

        if (rows < py) {
            fprintf(stderr, "Number of rows %d less then number of py processes %d\n", rows, py);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (cols < px) {
            fprintf(stderr, "Number of cols %d less then number of px processes %d\n", cols, px);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        int args[2] = {rows, cols};
        MPI_Bcast(&args, NELEMS(args), MPI_INT, 0, MPI_COMM_WORLD);

    } else {
        int args[2];
        MPI_Bcast(&args, NELEMS(args), MPI_INT, 0, MPI_COMM_WORLD);
        rows = args[0];
        cols = args[1];
    }

    // Выделяем память под локальные 2д решётки с дополнительными теневыми ячейками для строк и столбцов
    int ny = get_block_size(rows, ranky, py);
    int nx = get_block_size(cols, rankx, px);
    double *local_grid = calloc((ny + 2) * (nx + 2), sizeof(*local_grid));
    double *local_newgrid = calloc((ny + 2) * (nx + 2), sizeof(*local_newgrid));

    // заполняем уже известные места в матрице
    // левые и правые столбцы - нулевые
    // нижний(последний процесс) - sin(PI * x) * exp(-PI)
    // врехний(нулевой процесс) - sin(pi * x)
    double dx = 1.0 / (cols - 1.0);
    int sj = get_sum_of_prev_blocks(cols, rankx, px);
    if (ranky == 0) {
        for (int j = 1; j <= nx; j++) {
            double x = dx * (sj + j - 1);
            int ind = IND(0, j);
            local_newgrid[ind] = local_grid[ind] = sin(PI * x);
        }
    }
    if (ranky == py - 1) {
        for (int j = 1; j <= nx; j++) {
            double x = dx * (sj + j - 1);
            int ind = IND(ny + 1, j);
            local_newgrid[ind] = local_grid[ind] = sin(PI * x) * exp(-PI);
        }
    }

    // Узнаём соседей,второй аргумент для смены направления, по горизонтали или по вертикали
    int left, right, top, bottom;
    MPI_Cart_shift(cartcomm, 0, 1, &left, &right);
    MPI_Cart_shift(cartcomm, 1, 1, &top, &bottom);
    //Создаём тип для столбцов
    MPI_Datatype col;
    MPI_Type_vector(ny, 1, nx + 2, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);
    // Создаём тип для строк
    MPI_Datatype row;
    MPI_Type_contiguous(nx, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    MPI_Request reqs[8];
    double thalo = 0;
    double treduce = 0;

    int niters = -1;
    while (1) {
        niters++;
        // Обновляем матрицу
        for (int i = 1; i <= ny; i++) {
            for (int j = 1; j <= nx; j++) {
                local_newgrid[IND(i, j)] =
                    (local_grid[IND(i - 1, j)] + local_grid[IND(i + 1, j)] +
                    local_grid[IND(i, j - 1)] + local_grid[IND(i, j + 1)]) * 0.25;
            }
        }
        // Проверка на заданную точность
        double maxdiff = 0;
        for (int i = 1; i <= ny; i++) {
            for (int j = 1; j <= nx; j++) {
                int ind = IND(i, j);
                maxdiff = fmax(maxdiff, fabs(local_grid[ind] - local_newgrid[ind]));
            }
        }
        // Переносим новую сетку в локальную
        double *p = local_grid;
        local_grid = local_newgrid;
        local_newgrid = p;

        treduce -= MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        treduce += MPI_Wtime();
        if (maxdiff < EPS)
            break;
        
        // Обновление теневых ячеек
        thalo -= MPI_Wtime();
        MPI_Irecv(&local_grid[IND(0, 1)], 1, row, top, 0, cartcomm, &reqs[0]); // top
        MPI_Irecv(&local_grid[IND(ny + 1, 1)], 1, row, bottom, 0, cartcomm, &reqs[1]); // bottom
        MPI_Irecv(&local_grid[IND(1, 0)], 1, col, left, 0, cartcomm, &reqs[2]); // left
        MPI_Irecv(&local_grid[IND(1, nx + 1)], 1, col, right, 0, cartcomm, &reqs[3]); // right
        MPI_Isend(&local_grid[IND(1, 1)], 1, row, top, 0, cartcomm, &reqs[4]); // top
        MPI_Isend(&local_grid[IND(ny, 1)], 1, row, bottom, 0, cartcomm, &reqs[5]); // bottom
        MPI_Isend(&local_grid[IND(1, 1)], 1, col, left, 0, cartcomm, &reqs[6]); // left
        MPI_Isend(&local_grid[IND(1, nx)], 1, col, right, 0, cartcomm, &reqs[7]); // right
        MPI_Waitall(8, reqs, MPI_STATUS_IGNORE);
        thalo += MPI_Wtime();
    }

    MPI_Type_free(&row);
    MPI_Type_free(&col);
    free(local_newgrid);
    free(local_grid);
    ttotal += MPI_Wtime();

    if (rank == 0)
        printf("# Heat 2D (mpi): grid: rows %d, cols %d, procs %d (px %d, py %d)\n", rows, cols, commsize, px, py);

    int namelen;
    char procname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(procname, &namelen);
    printf("# P %4d (%2d, %2d) on %s: grid ny %d nx %d, total %.6f, mpi %.6f (%.2f) = allred %.6f (%.2f) + halo %.6f (%.2f)\n",
        rank, rankx, ranky, procname, ny, nx, ttotal, treduce + thalo, (treduce + thalo) / ttotal,
        treduce, treduce / (treduce + thalo), thalo, thalo / (treduce + thalo));
    
    double prof[3] = {ttotal, treduce, thalo};
    if (rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, prof, NELEMS(prof), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    printf("# procs %d : grid %d %d : niters %d : total time %.6f : mpi time %.6f : allred %.6f : halo %.6f\n",
        commsize, rows, cols, niters, prof[0], prof[1] + prof[2], prof[1], prof[2]);
    } else {
        MPI_Reduce(prof, NULL, NELEMS(prof), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}