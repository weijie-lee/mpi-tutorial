// examples/03-advanced/derived_type.c
// 演示 MPI 派生数据类型：发送结构体
// 派生数据类型可以让 MPI 直接发送非连续数据/结构体，不需要手动打包
// 编译：mpicc -O2 -o derived_type derived_type.c
// 运行：mpirun -np 2 ./derived_type

#include <mpi.h>
#include <stdio.h>
#include <stddef.h> // for offsetof

// 定义我们自己的结构体
typedef struct {
    int id;         // 整数 id
    double x;       // 坐标 x
    double y;       // 坐标 y
} Particle;

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        fprintf(stderr, "Need at least 2 processes\n");
        MPI_Finalize();
        return 1;
    }

    // --------------------------
    // 步骤一：定义 MPI 派生数据类型，对应我们的 Particle 结构体
    // MPI_Type_create_struct 用来创建任意结构体类型

    int count = 3;                  // 结构体有三个字段
    int blocklengths[3] = {1, 1, 1}; // 每个字段一个元素
    MPI_Datatype types[3] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE}; // 每个字段对应的 MPI 类型
    MPI_Aint offsets[3];              // 每个字段相对于结构体起始地址的偏移（字节）

    // --------------------------
    // 计算偏移，标准做法用 offsetof 宏
    // offsetof(Particle, field_name) 就是 field 在 struct 里的字节偏移
    offsets[0] = offsetof(Particle, id);
    offsets[1] = offsetof(Particle, x);
    offsets[2] = offsetof(Particle, y);

    // --------------------------
    // 创建 MPI 数据类型
    MPI_Datatype MPI_Particle; // 我们新的 MPI 类型
    MPI_Type_create_struct(
        count,          // 多少个块（字段）
        blocklengths,   // 每个块多少元素
        offsets,        // 每个块的偏移
        types,          // 每个块的类型
        &MPI_Particle   // 输出：新类型句柄
    );

    // --------------------------
    // 创建完必须 commit 才能使用
    MPI_Type_commit(&MPI_Particle);

    // --------------------------
    // 现在就可以直接发送 Particle 结构体了
    if (rank == 0) {
        Particle p;
        p.id = 42;
        p.x = 1.5;
        p.y = 2.7;
        printf("Rank 0: sending Particle id=%d, x=%f, y=%f\n", p.id, p.x, p.y);
        // 直接发送一个 MPI_Particle 类型，不需要手动拷贝每个字段
        MPI_Send(&p, 1, MPI_Particle, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        Particle p;
        MPI_Recv(&p, 1, MPI_Particle, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Rank 1: received Particle id=%d, x=%f, y=%f\n", p.id, p.x, p.y);
    }

    // --------------------------
    // 使用完释放类型（程序结束前可以不用，但好习惯）
    MPI_Type_free(&MPI_Particle);

    MPI_Finalize();
    return 0;
}
