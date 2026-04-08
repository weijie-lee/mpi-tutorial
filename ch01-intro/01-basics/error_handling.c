// examples/01-basics/error_handling.c
// 演示 MPI 错误处理基本用法
// 默认情况下，MPI 出错会直接终止整个程序
// 我们可以改成让错误返回给我们，自己处理
// 编译：mpicc -O2 -o error_handling error_handling.c
// 运行：mpirun -np 2 ./error_handling

#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int err;  // 存储 MPI 调用返回的错误码

    // 初始化 MPI，返回错误码
    err = MPI_Init(&argc, &argv);
    
    // --------------------------
    // 修改默认错误处理行为
    // 默认：MPI_ERRORS_ARE_FATAL → 任何错误直接终止整个程序
    // 改成：MPI_ERRORS_RETURN → 错误码返回给调用者，我们自己处理
    // 注意：MPI_Errhandler_set 在 MPI-2 之后就废弃了，新版使用 MPI_Comm_set_errhandler
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    
    int rank;
    // 调用 MPI_Comm_rank，获取错误码
    err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // --------------------------
    // 检查错误码，如果不是 MPI_SUCCESS 就是出错了
    if (err != MPI_SUCCESS) {
        // MPI 提供了获取错误信息字符串的函数
        char errstr[BUFSIZ];  // 存储错误信息
        int errlen;            // 错误信息长度
        // 获取错误信息字符串
        MPI_Error_string(err, errstr, &errlen);
        // 打印错误信息
        fprintf(stderr, "MPI Error when calling MPI_Comm_rank: %s\n", errstr);
        // 出错了，调用 MPI_Abort 终止所有进程
        // MPI_Abort 会立即终止整个 communicator 里的所有进程
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 如果走到这里说明没出错，打印正常信息
    printf("No error, rank = %d\n", rank);

    // 正常结束
    MPI_Finalize();
    return 0;
}
