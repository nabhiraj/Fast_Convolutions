#include<iostream>
#include<stdio.h>
#include<stdlib.h>
using namespace std;
void my2d_arrayPrint(int** arr,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<arr[i][j]<<" ";
        }
        cout<<endl;
    }
}
int** create_matrix(int n){
    int** arr=new int*[n];
    for(int i=0;i<n;i++){
        arr[i]=new int[n];
        for(int j=0;j<n;j++){
            arr[i][j]=0;
        }
    }
    return arr;
}
int unit_conv(int** layer,int row_start,int col_start,int** kernel,int kernel_size){
    int sum=0;
    for(int i=0;i<kernel_size;i++){//loop for row
        for(int j=0;j<kernel_size;j++){//loop for col
            sum+=(layer[row_start+i][col_start+j])*(kernel[i][j]);
        }
    }
    return sum;
}

int** basic_conv_operation(int** layer,int** kernel,int layer_size,int kernel_size){
    int sub_row_start=0;
    int sub_col_start=0;
    int sub_start=0;
    int sub_end=layer_size-kernel_size+1;
    int** res_mat=create_matrix(layer_size-kernel_size+1);
    for(sub_row_start=sub_start;sub_row_start<sub_end;sub_row_start++){
        for(sub_col_start=sub_start;sub_col_start<sub_end;sub_col_start++){
            res_mat[sub_row_start][sub_col_start]=unit_conv(layer,sub_row_start,sub_col_start,kernel,kernel_size);
        }
    }
    return res_mat;
}
int res_size(int layer_size,int kernel_size){
    return layer_size-kernel_size+1;
}
void fill_matrix_user(int** arr,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<"value at mat["<<i<<","<<j<<"]"<<endl;
            cin>>arr[i][j];
        }
    }
}
void fill_matrix_user_random(int** arr,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            //cin>>arr[i][j];
            arr[i][j]=rand()%10;
        }
    }
}
int main(){
    int layer_size;//dimention of the convolution layer.
    cout<<"enter layer size"<<endl;
    cin>>layer_size;
    cout<<"enter kernel size"<<endl;
    int kernel_size;
    cin>>kernel_size;
    int** layer_mat=create_matrix(layer_size);
    int** kernel_mat=create_matrix(kernel_size);
    fill_matrix_user_random(layer_mat,layer_size);
    fill_matrix_user_random(kernel_mat,kernel_size);
    cout<<"the layer matrix is "<<endl;
    my2d_arrayPrint(layer_mat,layer_size);
    cout<<"the kernel matrix is "<<endl;
    my2d_arrayPrint(kernel_mat,kernel_size);
    int** res=basic_conv_operation(layer_mat,kernel_mat,layer_size,kernel_size);
    cout<<"the size of the result matrix is "<<res_size(layer_size,kernel_size);
    cout<<"the result matrix is "<<endl;
    my2d_arrayPrint(res,res_size(layer_size,kernel_size));
    return 0;
}