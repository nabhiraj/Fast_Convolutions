#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include <complex.h>
//#include<fftw3.h>
#include<cufftw.h>
#include<math.h>
#define ind_ele(n,i,j) n*(i)+j
#define get_start(l_s,k_s,x) (x-k_s>=0)?x-k_s:(l_s+(x-k_s))
#define inc(l_s,x) (x+1>=l_s)?0:x+1
#define NT 2
using namespace std;
void flip_col(double* mat,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n/2;j++){
            double temp;
            temp=mat[ind_ele(n,i,j)];
            mat[ind_ele(n,i,j)]=mat[ind_ele(n,i,n-1-j)];
            mat[ind_ele(n,i,n-1-j)]=temp;
        }
    }
}
void flip_row(double* mat,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n/2;j++){
            double temp;
            temp=mat[ind_ele(n,j,i)];
            mat[ind_ele(n,j,i)]=mat[ind_ele(n,n-j-1,i)];
            mat[ind_ele(n,n-j-1,i)]=temp;
        }
    }
}
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
        //cout<<"inside iteration number "<<i<<endl;
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
void print_comp_mat(fftwf_complex* arr,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            cout<<"("<<arr[ind_ele(n,i,j)][0]<<","<<arr[ind_ele(n,i,j)][1]<<") ";
        }
        cout<<endl;
    
}
}
void write_to_real_mat(fftwf_complex* arr,double* arr_real,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            arr_real[ind_ele(n,i,j)]=arr[ind_ele(n,i,j)][0];
        }
    }
}





void conv(double* layer,double* kernel,int layer_size,int kernel_size,double* output){
    flip_row(kernel,kernel_size);
    flip_col(kernel,kernel_size);
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
double* preProcessKernel(double* plainKernel,int ini_size,int final_size){
    flip_col(plainKernel,ini_size);
    flip_row(plainKernel,ini_size);
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
double* preProcessKernel_static(double* plainKernel,double* res_mat,int ini_size,int final_size){
    flip_col(plainKernel,ini_size);
    flip_row(plainKernel,ini_size);
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

fftwf_complex* get_complex_representation(double* mat,int n){
    fftwf_complex* res=new fftwf_complex[n*n];
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            res[ind_ele(n,i,j)][0]=mat[ind_ele(n,i,j)];
            res[ind_ele(n,i,j)][1]=0;
        }
    }
    return res;
}
//------------------------------ al kernel methods ---------------------------------------------------
__global__ void to_real_p(fftwf_complex* arr,double* arr_real,int n){
    int tid=threadIdx.x;
    int row_per_thread=n/NT;
    int start_row=tid*row_per_thread;//start row is included.
    int end_row=start_row+row_per_thread;//end row is not included.
    if(tid==NT-1){//last threads flow
        end_row=n;
    }
    for(int i=start_row;i<end_row;i++){
        for(int j=0;j<n;j++){
            arr_real[ind_ele(n,i,j)]=arr[ind_ele(n,i,j)][0];
        }
    }
}
__global__ void scale_p(fftwf_complex* mat,int n){
    int nn=n*n;
    int tid=threadIdx.x;
    int row_per_thread=n/NT;
    int start_row=tid*row_per_thread;//start row is included.
    int end_row=start_row+row_per_thread;//end row is not included.
    if(tid==NT-1){//last threads flow
        end_row=n;
    }
    for(int i=start_row;i<end_row;i++){
        for(int j=0;j<n;j++){
            mat[ind_ele(n,i,j)][0]=mat[ind_ele(n,i,j)][0]/nn;
        }
    }
}
__global__ void conv_complex(double* src,fftwf_complex* des,int n){
    //NT
    //printf("starting the complex converstion\n");
    int tid=threadIdx.x;
    int row_per_thread=n/NT;
    int start_row=tid*row_per_thread;//start row is included.
    int end_row=start_row+row_per_thread;//end row is not included.
    //printf("start row is %d\n",start_row);
    //printf("end row is %d\n",end_row);
    if(tid==NT-1){//last threads flow
        end_row=n;
    }
    for(int i=start_row;i<end_row;i++){
        for(int j=0;j<n;j++){
            //printf("%d %d %d\n",tid,i,j);
            des[ind_ele(n,i,j)][0]=src[ind_ele(n,i,j)];
        }
    }
}

__global__ void poin_mul_parallel(fftwf_complex* A,fftwf_complex* B,fftwf_complex* output,int n){
    int tid=threadIdx.x;
    int row_per_thread=n/NT;
    int start_row=tid*row_per_thread;//start row is included.
    int end_row=start_row+row_per_thread;//end row is not included.
    if(tid==NT-1){//last threads flow
        end_row=n;
    }
    for(int i=start_row;i<end_row;i++){
        for(int j=0;j<n;j++){
            //c[ind_ele(n,i,j)]=a[ind_ele(n,i,j)]*b[ind_ele(n,i,j)];
            output[ind_ele(n,i,j)][0]=(((A[ind_ele(n,i,j)][0])*(B[ind_ele(n,i,j)][0]))-((A[ind_ele(n,i,j)][1])*(B[ind_ele(n,i,j)][1])));  
            output[ind_ele(n,i,j)][1]=(((A[ind_ele(n,i,j)][0])*(B[ind_ele(n,i,j)][1]))+((A[ind_ele(n,i,j)][1])*(B[ind_ele(n,i,j)][0])));
        }
    }
}
__global__ void sumP(double* arr,double* ac_sum,int n){
    double sum=0;
    int tid=threadIdx.x;
    int row_per_thread=n/NT;
    __shared__ double cache[NT];
    int start_row=tid*row_per_thread;//start row is included.
    int end_row=start_row+row_per_thread;//end row is not included.
    if(tid==NT-1){//last threads flow
        end_row=n;
    }
    for(int i=start_row;i<end_row;i++){
        for(int j=0;j<n;j++){
            sum+=arr[ind_ele(n,i,j)];
        }
    }
    cache[tid]=sum;
    __syncthreads();
    *ac_sum=0;
    if(tid==0){
        for(int i=0;i<NT;i++){
            *ac_sum+=cache[i];
        }
    }
}
__global__ void fill_constant_parallel(double* grad,double temp,int n){
    int tid=threadIdx.x;
    int row_per_thread=n/NT;
    int start_row=tid*row_per_thread;//start row is included.
    int end_row=start_row+row_per_thread;//end row is not included.
    if(tid==NT-1){//last threads flow
        end_row=n;
    }
    for(int i=start_row;i<end_row;i++){
        for(int j=0;j<n;j++){
            grad[ind_ele(n,i,j)]=temp;
        }
    }
}
__global__ void updatekernel(double* kernel,double* kernel_grad,double lr,int n){
    int tid=threadIdx.x;
    int row_per_thread=n/NT;
    int start_row=tid*row_per_thread;//start row is included.
    int end_row=start_row+row_per_thread;//end row is not included.
    if(tid==NT-1){//last threads flow
        end_row=n;
    }
    for(int i=start_row;i<end_row;i++){
        for(int j=0;j<n;j++){
            kernel[ind_ele(n,i,j)]=kernel[ind_ele(n,i,j)]-(lr*kernel_grad[ind_ele(n,i,j)]);
        }
    }
}
struct convolution_enviorment{
    //things required for forward propogation
    double* layer;//d
    double* kernel;//d
    double* d_layer;//d
    double* d_kernel;//d
    fftwf_complex* d_comp_layer;//d
    fftwf_complex* d_comp_kernel;//d
    fftwf_complex* d_fft_layer;//d
    fftwf_complex* d_fft_kernel;//d
    fftwf_complex* d_fft_mul;//d
    fftwf_complex* d_mul;//d
    double* d_real_mul;//d
    double* d_predicted;//d 
    double* predicted;//d
    int filter_size;
    fftwf_plan get_layer;
    fftwf_plan get_kernel;
    fftwf_plan getBack_mul;

    double* d_grad;//d
    fftwf_complex* d_comp_grad;//d
    fftwf_complex* d_comp_fft_grad;//dd
    fftwf_complex* d_comp_fft_kernel_grad;//d
    fftwf_complex* d_comp_kernel_grad;//d
    double* d_kernel_grad;
    fftwf_plan get_gradient;
    fftwf_plan getBackkernel_gradient;
    double learning_rate;
};
/*
void insialize_backProp(convolution_enviorment* ce,double lr){
    int tt=ce->filter_size;
    tt=tt*tt;
    ce->fft_gradient = new fftwf_complex[tt];
    ce->fft_kernel_gradient=new fftwf_complex[tt];
    ce->kernel_gradient=new fftwf_complex[tt];
    //ce->aux_mat=new double[tt];
    //ce->aux_mat2=new double[tt];
    ce->kernel_gradient_real=new double[tt];
    ce->gradients=new double[tt];
    ce->learning_rate=lr;
}*/
convolution_enviorment create_convolution_enviormentParellel(int fs){
    convolution_enviorment ce;
    ce.filter_size=fs;
    ce.layer=new double[fs*fs];
    ce.kernel=new double[fs*fs];
    cudaMalloc((void**)&ce.d_layer,sizeof(double)*fs*fs);             //things may get pointerise.
    cudaMalloc((void**)&ce.d_kernel,sizeof(double)*fs*fs);
    cudaMalloc((void**)&ce.d_fft_layer,sizeof(fftwf_complex)*fs*fs);
    cudaMalloc((void**)&ce.d_fft_kernel,sizeof(fftwf_complex)*fs*fs);
    cudaMalloc((void**)&ce.d_fft_mul,sizeof(fftwf_complex)*fs*fs);
    cudaMalloc((void**)&ce.d_mul,sizeof(fftwf_complex)*fs*fs);
    cudaMalloc((void**)&ce.d_comp_kernel,sizeof(fftwf_complex)*fs*fs);
    cudaMalloc((void**)&ce.d_comp_layer,sizeof(fftwf_complex)*fs*fs);
    cudaMalloc((void**)&ce.d_real_mul,sizeof(double)*fs*fs);
    cudaMalloc((void**)&ce.d_predicted,sizeof(double));
    ce.predicted=new double();
    *ce.predicted=10;
    return ce;
}
void ini_bp(convolution_enviorment* ce,double le){
    int tt=ce->filter_size;
    tt=tt*tt;
    cudaMalloc((void**)&(ce->d_grad),sizeof(double)*tt);
    cudaMalloc((void**)&(ce->d_kernel_grad),sizeof(double)*tt);
    cudaMalloc((void**)&(ce->d_comp_grad),sizeof(fftwf_complex)*tt);
    cudaMalloc((void**)&(ce->d_comp_fft_grad),sizeof(fftwf_complex)*tt);
    cudaMalloc((void**)&(ce->d_comp_fft_kernel_grad),sizeof(fftwf_complex)*tt);
    cudaMalloc((void**)&(ce->d_comp_kernel_grad),sizeof(fftwf_complex)*tt);
    ce->learning_rate=le;
}
void backPropogate(convolution_enviorment ce,double actual_value){
    //cout<<"mse error is "<<(*(ce.predicted)-actual_value)*(*(ce.predicted)-actual_value)<<endl;
    //cout<<"entering backprog"<<endl;
    double temp=2*(*(ce.predicted)-actual_value);
    fill_constant_parallel<<<1,NT>>>(ce.d_grad,temp,ce.filter_size);
    cudaDeviceSynchronize();
    //cout<<"filling of constant done"<<endl;
    conv_complex<<<1,NT>>>(ce.d_grad,ce.d_comp_grad,ce.filter_size);
    cudaDeviceSynchronize();
    //cout<<"convertion into complex"<<endl;
    ce.get_gradient=fftwf_plan_dft_2d(ce.filter_size,ce.filter_size,ce.d_comp_grad,ce.d_comp_fft_grad,FFTW_FORWARD,FFTW_ESTIMATE);
    fftwf_execute(ce.get_gradient);
    poin_mul_parallel<<<1,NT>>>(ce.d_comp_fft_grad,ce.d_fft_layer,ce.d_comp_fft_kernel_grad,ce.filter_size);
    cudaDeviceSynchronize();
    ce.getBackkernel_gradient=fftwf_plan_dft_2d(ce.filter_size,ce.filter_size,ce.d_comp_fft_kernel_grad,ce.d_comp_kernel_grad,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftwf_execute(ce.getBackkernel_gradient);
    //we need to scale this.
    scale_p<<<1,NT>>>(ce.d_comp_kernel_grad,ce.filter_size);
    cudaDeviceSynchronize();
    to_real_p<<<1,NT>>>(ce.d_comp_kernel_grad,ce.d_kernel_grad,ce.filter_size);
    cudaDeviceSynchronize();
    updatekernel<<<1,NT>>>(ce.d_kernel,ce.d_kernel_grad,ce.learning_rate,ce.filter_size);
    cudaDeviceSynchronize();
}
void fixKernel(convolution_enviorment ce){
    cudaMemcpy(ce.d_kernel,ce.kernel,ce.filter_size*ce.filter_size,cudaMemcpyHostToDevice);
}
void forwardPass_parallel(convolution_enviorment ce){
    //cout<<"starting the routine"<<endl;
    int fs=ce.filter_size*ce.filter_size;
    cudaMemcpy(ce.d_layer,ce.layer,fs,cudaMemcpyHostToDevice);
    //cudaMemcpy(ce.d_kernel,ce.kernel,fs,cudaMemcpyHostToDevice);
    //cout<<"senf the kernel and layer to the device"<<endl;
    conv_complex<<<1,NT>>>(ce.d_layer,ce.d_comp_layer,ce.filter_size);
    conv_complex<<<1,NT>>>(ce.d_kernel,ce.d_comp_kernel,ce.filter_size);
    //cout<<"convertin them to complex"<<endl;
    ce.get_layer=fftwf_plan_dft_2d(ce.filter_size,ce.filter_size,ce.d_comp_layer,ce.d_fft_layer,FFTW_FORWARD,FFTW_ESTIMATE);
    ce.get_kernel=fftwf_plan_dft_2d(ce.filter_size,ce.filter_size,ce.d_comp_kernel,ce.d_fft_kernel,FFTW_FORWARD,FFTW_ESTIMATE);
    fftwf_execute(ce.get_layer);
    fftwf_execute(ce.get_kernel);    
    cudaDeviceSynchronize();
    //cout<<"doing the fft transform"<<endl;
    poin_mul_parallel<<<1,NT>>>(ce.d_fft_kernel,ce.d_fft_layer,ce.d_fft_mul,ce.filter_size);
    cudaDeviceSynchronize();
    //cout<<"done point wise multiplication"<<endl;
    ce.getBack_mul=fftwf_plan_dft_2d(ce.filter_size,ce.filter_size,ce.d_fft_mul,ce.d_mul,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftwf_execute(ce.getBack_mul);
    cudaDeviceSynchronize();
    //cout<<"inverse fft done"<<endl;
    scale_p<<<1,NT>>>(ce.d_mul,ce.filter_size);   
    cudaDeviceSynchronize();
    //cout<<"scalling is done"<<endl;
    to_real_p<<<1,NT>>>(ce.d_mul,ce.d_real_mul,ce.filter_size);
    cudaDeviceSynchronize();
    //cout<<"convertion to real done"<<endl;
    sumP<<<1,NT>>>(ce.d_real_mul,ce.d_predicted,ce.filter_size);
    cudaMemcpy(ce.predicted,ce.d_predicted,sizeof(double),cudaMemcpyDeviceToHost);
    //cout<<"summation doen and copy to host done"<<endl;
}
void trainIteration(convolution_enviorment ce){
    forwardPass_parallel(ce);
    backPropogate(ce,30);
}

void mat_mul(double* a,double* b,double* res,int n){
    for (int i = 0; i<n;i++) {
        for (int j =0; j<n;j++) {
            for (int k=0;k<n;k++) {
                res[ind_ele(n,i,j)]+=a[ind_ele(n,i,k)]*b[ind_ele(n,k,j)];
            }
        }
    }
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

//before we need to set the kernel we need to preprocess teh kernel
//we need to set the layer
//data set is not coded in it yet



int main(){
    cout<<"enter layer size"<<endl;
    int l_s;
    cin>>l_s;
    int k_s;
    cout<<"enter kernel size"<<endl;
    cin>>k_s;
    if(k_s%2==0){
        cout<<"kernel should be odd"<<endl;
        return 0;
    }
    convolution_enviorment ce=create_convolution_enviormentParellel(l_s);
    ini_bp(&ce,0.00000000001);
    double* k=create_matrix(k_s);
    fill_random(ce.layer,l_s);
    fill_constant(k,20,k_s);
    preProcessKernel_static(k,ce.kernel,k_s,l_s);
    fixKernel(ce);

    clock_t start_time;
    clock_t end_time;
    start_time=clock();
    for(int i=0;i<120;i++)
    trainIteration(ce);
    end_time=clock();
    double clock_taken=double(end_time - start_time);
    double time_taken=clock_taken/double(CLOCKS_PER_SEC);
    cout<<"the time taken is "<<time_taken<<endl;
    return 0;
}
//layer=[[1,1,1,2,1],[8,3,2,6,2],[4,5,6,3,3],[1,1,1,1,1],[3,3,3,4,5]]
//kernel=[[1,1,2],[3,1,2],[4,5,5]]
