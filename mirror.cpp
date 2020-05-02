#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#define ind_ele(n,i,j) n*(i)+j
//the above macro works fine.
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
int main(){
    //first we need to create the matrix in the continuous way.
    //how to access a row major order matrix in a continuous way.
    //this accessing has to be put as a macro.
    //B + w * (N * (i-1) + j-1)
    double* arr=create_matrix(10);
    fill_user(arr,10);
    print_mat(arr,10);
    cout<<arr[ind_ele(10,3,3)]<<endl;
    return 0;
}