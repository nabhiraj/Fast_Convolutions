#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<cufftw.h>
#include<cuda_runtime.h>
//#include<fftw3.h>
using namespace std;
void print_res(fftw_complex* com,int n){
    for(int i=0;i<n;i++){
        cout<<com[i][0]<<endl;
    }
}
int main(){
    cout<<"the code is running"<<endl;
    //first we will test one dimention.
    fftw_complex in[10];
    fftw_complex out[10];
    fftw_complex out2[10];
    fftw_complex* d_in;
    fftw_complex* d_out;
    fftw_complex* d_out2;
    cudaMalloc((void**)&d_in,10*sizeof(fftw_complex));
    cudaMalloc((void**)&d_out,10*sizeof(fftw_complex));
    cudaMalloc((void**)&d_out2,10*sizeof(fftw_complex));
    //send the data to the device.
    cudaMemcpy(d_in,&in,10*sizeof(fftw_complex),cudaMemcpyHostToDevice);
    fftw_plan p=fftw_plan_dft_1d(10,d_in,d_out,FFTW_FORWARD,FFTW_ESTIMATE); 
    fftw_execute(p);
    cout<<"input"<<endl;
    print_res(in,10);
    cudaMemcpy(&out,d_out,10*sizeof(fftw_complex),cudaMemcpyDeviceToHost);
    cout<<"forie t"<<endl;
    print_res(out,10);
    fftw_plan p2=fftw_plan_dft_1d(10,d_out,d_out2,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftw_execute(p2);
    //brin out2 back.
    cudaMemcpy(&out2,d_out2,10*sizeof(fftw_complex),cudaMemcpyDeviceToHost);
    for(int i=0;i<10;i++){
        out2[i][0]=out2[i][0]/10;
    }
    cout<<"gettin input back"<<endl;
    print_res(out2,10);
    return 0;
}