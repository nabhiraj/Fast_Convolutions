#include<stdio.h>
#include<stdlib.h>
#include<iostream>
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
void fill_user(double* mat,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<"--> ["<<i<<"]"<<"["<<j<<"]"<<endl;
            cin>>mat[ind_ele(n,i,j)];
        }
    }
}
void print_mat(double* arr,int n){
    cout<<"entering inside the printing method"<<endl;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<arr[ind_ele(n,i,j)]<<" ";
        }
        cout<<endl;
    }
    cout<<"outside the printing method"<<endl;
}
//this method is not tested yer
void conv(double* layer,double* kernel,int layer_size,int kernel_size,double* output){
    for(int i=0;i<layer_size;i++){
        for(int j=0;j<layer_size;j++){
            //these operation are not fully connvected
            //should we implemented in macro.
            int k_s=(kernel_size-1)/2;
            int l_row_start=get_start(layer_size,k_s,i);
            int l_col_start=get_start(layer_size,k_s,j);
            cout<<i<<" "<<j<<endl;
            cout<<"the row start is at "<<l_row_start<<endl;
            cout<<"the col start is at "<<l_col_start<<endl;
            cout<<endl;
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
int main(){
    //this is the master test stroke.
    int l_n=5;
    int k_n=3;
    double* layer=create_matrix(l_n);
    fill_user(layer,l_n);
    double* kernel=create_matrix(k_n);
    fill_user(kernel,k_n);
    double* output=create_matrix(l_n);
    conv(layer,kernel,l_n,k_n,output);
    cout<<"all the processing is done"<<endl;
    cout<<"the layer matrix is "<<endl;
    print_mat(layer,l_n);
    cout<<"the kernel matrix is "<<endl;
    print_mat(kernel,k_n);
    cout<<"the output matrix is "<<endl;
    print_mat(output,l_n);
    return 0;
}