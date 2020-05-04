#include<stdio.h>
#include<stdlib.h>
#include<iostream>
//#include <sys/time.h> 
#define ind_ele(n,i,j) n*(i)+j
#define get_start(l_s,k_s,x) (x-k_s>=0)?x-k_s:(l_s+(x-k_s))
#define inc(l_s,x) (x+1>=l_s)?0:x+1

using namespace std;
double* create_matrix(int n){
    double* mat=new double[n*n];
    return mat;
}
void fill_random(double* mat,int n){
    for(int i=0;i<n*n;i++){
        mat[i]=rand()%10;
    }
}
void fill_constant(double* mat,double c,int n){
    for(int i=0;i<n*n;i++){
        mat[i]=c;
    }
}
void fill_user(double* mat,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<"--> ["<<i<<"]"<<"["<<j<<"]"<<endl;
            cin>>mat[ind_ele(n,i,j)];
        }
    }
}
void print_mat(double* arr,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<arr[ind_ele(n,i,j)]<<" ";
        }
        cout<<endl;
    }
}
//this method is not tested yer
void conv(double* layer,double* kernel,int layer_size,int kernel_size,double* output){
    for(int i=0;i<layer_size;i++){
        for(int j=0;j<layer_size;j++){
            int k_s=(kernel_size-1)/2;
            int l_row_start=get_start(layer_size,k_s,i);
            int l_col_start=get_start(layer_size,k_s,j);
            double sum=0;
            for(int ik=0;ik<kernel_size;ik++){
                l_col_start=get_start(layer_size,k_s,j);
                for(int jk=0;jk<kernel_size;jk++){
                    sum+=kernel[ind_ele(kernel_size,ik,jk)]*layer[ind_ele(layer_size,l_row_start,l_col_start)];
                    l_col_start=inc(layer_size,l_col_start);
                }
                l_row_start=inc(layer_size,l_row_start);
            }
            output[ind_ele(layer_size,i,j)]=sum;
        }
    }
}
//now we have to write the code to flip
//this method is working fine.
double* preProcessKernel(double* plainKernel,int ini_size,int final_size){//i only will do it.
    double* res_mat=create_matrix(final_size);
    fill_constant(res_mat,0,final_size);
    int ini_row=final_size-1;
    int ini_col=final_size-1;
    for(int i=0;i<ini_size;i++){
        ini_col=final_size-1;
        for(int j=0;j<ini_size;j++){
            res_mat[ind_ele(final_size,ini_row,ini_col)]=plainKernel[ind_ele(ini_size,i,j)];
            ini_col=inc(final_size,ini_col);
        }
        ini_row=inc(final_size,ini_row);
    }
    return res_mat;
}
//we will assume the kernel is already prerpssed
void fft_conv(double* layer,double* kernel,int layer_size,int kernel_size,double* output){
    //chalenging code comes here
}
void messaure_normal_time(){
    int layer_length=512;
    double* layer=create_matrix(layer_length);
    fill_random(layer,layer_length);
    double* output=create_matrix(layer_length);
    for(int kernel_length=3;kernel_length<512;kernel_length+=2){
        double* kernel=create_matrix(kernel_length);
        fill_random(kernel,kernel_length);
        clock_t start_time;
        clock_t end_time;
        start_time=clock();
        for(int i=0;i<10;i++){
            conv(layer,kernel,layer_length,kernel_length,output);
        }
        end_time=clock();
        double clock_taken=double(end_time - start_time);
        clock_taken=clock_taken/10;
        double time_taken=clock_taken/double(CLOCKS_PER_SEC);
        cout<<kernel_length<<" "<<time_taken<<endl;
        delete kernel;
    }
}
int main(){
    //messaure_normal_time();
    //testing teh preprocessing of the ekrnel
    double* kernel=create_matrix(3);
    fill_user(kernel,3);
    cout<<"the kernel filter looks like this"<<endl;
    print_mat(kernel,3);
    double* processed_kernel=preProcessKernel(kernel,3,5);
    cout<<"after preprocessing the kernel filter looks like this"<<endl;
    print_mat(processed_kernel,5);

    return 0;
}