#include<stdio.h>
#include<stdlib.h>
__global__ void  mykernel(int* a,int* b,int* c){
    //no code is here
    *c=*a+*b;
}
int main(){
    int a=1;
    int b=9;
    int c;
    int* d_a;
    int* d_b;
    int* d_c;
    cudaMalloc((void**)&d_a,sizeof(int));
    cudaMalloc((void**)&d_b,sizeof(int));
    cudaMalloc((void**)&d_c,sizeof(int));
    cudaMemcpy(d_a,&a,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,&b,sizeof(int),cudaMemcpyHostToDevice);
    mykernel<<<1,1>>>(d_a,d_b,d_c);
    cudaMemcpy(&c,d_c,sizeof(int),cudaMemcpyDeviceToHost);
    printf("the summation is %d",c);
    return 0;
}
/*
#include<stdio.h>
__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>(); 
    printf("hello world from host");
    return 0;
}*/