// examples/01-basics/error_handling.c
// 演示 MPI 错误处理基本用法
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int err;
    err = MPI_Init(&argc, &argv);
    
    // 默认情况下，任何 MPI 错误都会直接终止整个程序
    // 这里改成返回错误码给我们自己处理
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    
    int rank;
    err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (err != MPI_SUCCESS) {
        char errstr[BUFSIZ];
        int errlen;
        // 获取错误信息字符串
        MPI_Error_string(err, errstr, &errlen);
        fprintf(stderr, "MPI Error: %s\n", errstr);
        // 出错了，终止所有进程
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("No error, rank = %d\n", rank);
    MPI_Finalize();
    return 0;
}
