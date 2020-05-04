#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<fftw3.h>
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
    fftw_plan p=fftw_plan_dft_1d(10,in,out,FFTW_FORWARD,FFTW_ESTIMATE);   //the plan is ready.
    fftw_execute(p);
    cout<<"everything is getting executed"<<endl;
    print_res(in,10);
    cout<<"changing the game"<<endl;
    print_res(out,10);
    fftw_complex a;
    a[0]=90;
    a[1]=45;
    fftw_complex b;
    b[0]=2;
    b[1]=2;
    cout<<"the real part of c is "<<c[0]<<endl;
    cout<<"the complex part of c is "<<c[1]<<endl;
    return 0;
}