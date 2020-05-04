#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include <complex.h>
#include<fftw3.h>
#define ind_ele(n,i,j) n*(i)+j
#define get_start(l_s,k_s,x) (x-k_s>=0)?x-k_s:(l_s+(x-k_s))
#define inc(l_s,x) (x+1>=l_s)?0:x+1

using namespace std;
void flip_col(float* mat,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n/2;j++){
            float temp;
            temp=mat[ind_ele(n,i,j)];
            mat[ind_ele(n,i,j)]=mat[ind_ele(n,i,n-1-j)];
            mat[ind_ele(n,i,n-1-j)]=temp;
        }
    }
}
void flip_row(float* mat,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n/2;j++){
            float temp;
            temp=mat[ind_ele(n,j,i)];
            mat[ind_ele(n,j,i)]=mat[ind_ele(n,n-j-1,i)];
            mat[ind_ele(n,n-j-1,i)]=temp;
        }
    }
}
float* create_matrix(int n){
    float* mat=new float[n*n];
    return mat;
}
void fill_random(float* mat,int n){
    for(int i=0;i<n*n;i++){
        mat[i]=rand()%10;
    }
}
void fill_constant(float* mat,float c,int n){
    for(int i=0;i<n*n;i++){
        mat[i]=c;
    }
}
void fill_user(float* mat,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<"--> ["<<i<<"]"<<"["<<j<<"]"<<endl;
            cin>>mat[ind_ele(n,i,j)];
        }
    }
}
void print_mat(float* arr,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<arr[ind_ele(n,i,j)]<<" ";
        }
        cout<<endl;
    }
}
void print_comp_mat(fftwf_complex* arr,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<"("<<arr[ind_ele(n,i,j)][0]<<","<<arr[ind_ele(n,i,j)][1]<<") ";
        }
        cout<<endl;
    
}
}
void conv(float* layer,float* kernel,int layer_size,int kernel_size,float* output){
    flip_row(kernel,kernel_size);
    flip_col(kernel,kernel_size);
    for(int i=0;i<layer_size;i++){
        for(int j=0;j<layer_size;j++){
            int k_s=(kernel_size-1)/2;
            int l_row_start=get_start(layer_size,k_s,i);
            int l_col_start=get_start(layer_size,k_s,j);
            float sum=0;
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
float* preProcessKernel(float* plainKernel,int ini_size,int final_size){//i only will do it.
    flip_col(plainKernel,ini_size);
    flip_row(plainKernel,ini_size);
    float* res_mat=create_matrix(final_size);
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
//this method is not tested yet.
void mul(fftwf_complex* A,fftwf_complex* B,fftwf_complex* output,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            output[ind_ele(n,i,j)][0]=(((A[ind_ele(n,i,j)][0])*(B[ind_ele(n,i,j)][0]))-((A[ind_ele(n,i,j)][1])*(B[ind_ele(n,i,j)][1])));  
            output[ind_ele(n,i,j)][1]=(((A[ind_ele(n,i,j)][0])*(B[ind_ele(n,i,j)][1]))+((A[ind_ele(n,i,j)][1])*(B[ind_ele(n,i,j)][0])));
        }
    }
}
void scale_comp(fftwf_complex* mat,int n){
    int nn=n*n;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            mat[ind_ele(n,i,j)][0]=mat[ind_ele(n,i,j)][0]/nn;
        }
    }
}
//routine to convert a float matrix to a complex matrix.
fftwf_complex* get_complex_representation(float* mat,int n){
    fftwf_complex* res=new fftwf_complex[n*n];
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            res[ind_ele(n,i,j)][0]=mat[ind_ele(n,i,j)];
            res[ind_ele(n,i,j)][1]=0;
        }
    }
    return res;
}
//this method is not tested yet.
//there is mistake in inverse forirer transform.
void fft_conv(float* layer,float* kernel,int filter_size,fftwf_complex* output){
    fftwf_complex* fft_layer;
    fft_layer=(fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*filter_size*filter_size);
    fftwf_complex* fft_kernel;
    fft_kernel=(fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*filter_size*filter_size);
    fftwf_plan get_layer=fftwf_plan_dft_2d(filter_size,filter_size,get_complex_representation(layer,filter_size),fft_layer,FFTW_FORWARD,FFTW_ESTIMATE);
    fftwf_plan get_kernel=fftwf_plan_dft_2d(filter_size,filter_size,get_complex_representation(kernel,filter_size),fft_kernel,FFTW_FORWARD,FFTW_ESTIMATE);
    //now we need to multiply.
    fftwf_complex* fft_mul=(fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*filter_size*filter_size);
    fftwf_execute(get_layer);
    fftwf_execute(get_kernel);
    cout<<"fft of the layer is "<<endl;
    print_comp_mat(fft_layer,filter_size);
    cout<<"fft of the kernel is "<<endl;
    print_comp_mat(fft_kernel,filter_size);
    mul(fft_layer,fft_kernel,fft_mul,filter_size);
    cout<<"their multiplication is "<<endl;
    print_comp_mat(fft_mul,filter_size);
    fftwf_plan get_realMul=fftwf_plan_dft_2d(filter_size,filter_size,fft_mul,output,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftwf_execute(get_realMul);
    scale_comp(output,filter_size);
}
void messaure_normal_time(){
    int layer_length=512;
    float* layer=create_matrix(layer_length);
    fill_random(layer,layer_length);
    float* output=create_matrix(layer_length);
    for(int kernel_length=3;kernel_length<512;kernel_length+=2){
        float* kernel=create_matrix(kernel_length);
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
        double time_taken=clock_taken/float(CLOCKS_PER_SEC);
        cout<<kernel_length<<" "<<time_taken<<endl;
        delete kernel;
    }
}
int main(){
    //testing the flipping operation.
    float* l=create_matrix(5);
    float* k=create_matrix(3);
    cout<<"enter the elements of the layler"<<endl;
    fill_user(l,5);
    cout<<"enter the elemnts of the kernel"<<endl;
    fill_user(k,3);
    float* ck=preProcessKernel(k,3,5);
    //float* output=create_matrix(5);
    fftwf_complex* output=new fftwf_complex[5*5];
    float* output2=create_matrix(5);
    fft_conv(l,ck,5,output);
    conv(l,k,5,3,output2);
    cout<<"the convolution operation gives"<<endl;
    print_comp_mat(output,5);
    print_mat(output2,5);
    return 0;
}


//layer=[[1,1,1,2,1],[8,3,2,6,2],[4,5,6,3,3],[1,1,1,1,1],[3,3,3,4,5]]
//kernel=[[1,1,2],[3,1,2],[4,5,5]]