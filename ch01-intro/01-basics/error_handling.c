// examples/01-basics/error_handling.c
// 演示 MPI 错误处理：默认错误会终止全部进程，改成 RETURN 后由程序自己处理。
// 编译：mpicc -O2 -o error_handling error_handling.c
// 运行：mpirun -np 2 ./error_handling

#include <mpi.h>
#include <stdio.h>

static void print_error(const char *where, int err) {
    char errstr[MPI_MAX_ERROR_STRING];
    int errlen = 0;
    MPI_Error_string(err, errstr, &errlen);
    printf("%s -> code %d (%s)\n", where, err, errstr);
}

int main(int argc, char **argv) {
    int err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init failed: %d\n", err);
        return 1;
    }

    // 默认 MPI_ERRORS_ARE_FATAL：出错直接终止所有进程。
    // MPI_ERRORS_RETURN：把错误码交还给调用者。MPI_Errhandler_set 已废弃。
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    int rank = -1;
    int size = 0;
    err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (err != MPI_SUCCESS) {
        print_error("MPI_Comm_rank", err);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    err = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (err != MPI_SUCCESS) {
        print_error("MPI_Comm_size", err);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("No error on the success path, rank = %d / %d\n", rank, size);

    // 故意使用非法 rank。正确安装了错误处理器时，这里返回 MPI_ERR_RANK，
    // 进程不会被集体杀掉，后面的检查才能执行。
    int payload = rank;
    err = MPI_Send(&payload, 1, MPI_INT, /*dest=*/size, 0, MPI_COMM_WORLD);
    int saw_rank_error = (err == MPI_ERR_RANK);

    // 非法通信域。MPI_COMM_NULL 不能拿来查询 rank。
    int ignored = 0;
    int err_comm = MPI_Comm_rank(MPI_COMM_NULL, &ignored);
    int saw_comm_error = (err_comm == MPI_ERR_COMM);

    if (rank == 0) {
        print_error("MPI_Send to rank == size", err);
        print_error("MPI_Comm_rank(MPI_COMM_NULL)", err_comm);
        int failed = !(saw_rank_error && saw_comm_error);
        if (failed) {
            fprintf(stderr, "expected MPI_ERR_RANK and MPI_ERR_COMM\n");
        }
        printf("RESULT: %s\n", failed ? "FAIL" : "PASS");
    }

    MPI_Finalize();
    return 0;
}
