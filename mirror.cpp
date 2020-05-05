#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include <complex.h>
#include<fftw3.h>
#include<math.h>
#define ind_ele(n,i,j) n*(i)+j
#define get_start(l_s,k_s,x) (x-k_s>=0)?x-k_s:(l_s+(x-k_s))
#define inc(l_s,x) (x+1>=l_s)?0:x+1
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
        mat[i]=rand()%2;
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
//the kernel which it takes is a preprocessed kernel.
void fft_conv(double* layer,double* kernel,int filter_size,fftwf_complex* output){
    fftwf_complex* fft_layer;
    fftwf_complex* fft_kernel;
    fft_layer=(fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*filter_size*filter_size);
    fft_kernel=(fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*filter_size*filter_size);
    fftwf_complex* fft_mul=(fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*filter_size*filter_size);
    fftwf_plan get_layer=fftwf_plan_dft_2d(filter_size,filter_size,get_complex_representation(layer,filter_size),fft_layer,FFTW_FORWARD,FFTW_ESTIMATE);
    fftwf_plan get_kernel=fftwf_plan_dft_2d(filter_size,filter_size,get_complex_representation(kernel,filter_size),fft_kernel,FFTW_FORWARD,FFTW_ESTIMATE);
    fftwf_execute(get_layer);
    fftwf_execute(get_kernel);
    mul(fft_layer,fft_kernel,fft_mul,filter_size);
    fftwf_plan get_realMul=fftwf_plan_dft_2d(filter_size,filter_size,fft_mul,output,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftwf_execute(get_realMul);
    scale_comp(output,filter_size);
    //distruction of plan object are not done yet in this code.
}

//here kernel is preprocessed kernel.
//we have to inisialize back propogation seperatly.
struct convolution_enviorment{
    fftwf_complex* fft_layer;//both
    fftwf_complex* fft_kernel;//both
    fftwf_complex* fft_mul;//forword
    fftwf_complex* mul;//forword
    fftwf_complex* fft_gradient;//backward done
    fftwf_complex* fft_kernel_gradient;//backward done
    fftwf_complex* kernel_gradient;//backward done.
    double* real_mul;//right now activation function is not used. //we can apply activation on the same memory space.
    double* layer;//both
    double* kernel;//both
    int filter_size;//both
    fftwf_plan get_layer;//forward
    fftwf_plan get_kernel;//forward
    fftwf_plan getBack_mul;//forward
    fftwf_plan get_gradient;//backward  done
    fftwf_plan getBackkernel_gradient;//backward done
    double* aux_mat;//back done
    double* aux_mat2;//back done
    double* kernel_gradient_real;//back done
    double* gradients;//back done
    double predicted;//back done
    double learning_rate;//back done
};
void insialize_backProp(convolution_enviorment* ce,double lr){
    int tt=ce->filter_size;
    tt=tt*tt;
    ce->fft_gradient = new fftwf_complex[tt];
    ce->fft_kernel_gradient=new fftwf_complex[tt];
    ce->kernel_gradient=new fftwf_complex[tt];
    ce->aux_mat=new double[tt];
    ce->aux_mat2=new double[tt];
    ce->kernel_gradient_real=new double[tt];
    ce->gradients=new double[tt];
    ce->learning_rate=lr;
}
convolution_enviorment create_convolution_enviorment(int fs){
    convolution_enviorment myenv;
    myenv.filter_size=fs;
    myenv.layer=create_matrix(fs);
    myenv.kernel=create_matrix(fs);
    myenv.real_mul=create_matrix(fs);
    myenv.fft_kernel=new fftwf_complex[fs*fs];
    myenv.fft_layer=new fftwf_complex[fs*fs];
    myenv.fft_mul=new fftwf_complex[fs*fs];
    myenv.mul=new fftwf_complex[fs*fs];
    //myenv.aux_mat=create_matrix(myenv.filter_size);
    return myenv;
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
void mat_one_minus_this(double* a,double* res,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            res[ind_ele(n,i,j)]=1-a[ind_ele(n,i,j)];
        }
    }
}

void updatekernel(convolution_enviorment ce){
    int n=ce.filter_size;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            ce.kernel[ind_ele(n,i,j)]-=(ce.learning_rate*ce.kernel_gradient_real[ind_ele(n,i,j)]);
        }
    }
}
void relu_derivative(double* in,double* out,int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(in[ind_ele(n,i,j)]>0){
                out[ind_ele(n,i,j)]=1;
            }else{
                out[ind_ele(n,i,j)]=0;
            }
        }
    }
}
//code to update the kernel is not included in the backprop.
void backPropogate(convolution_enviorment ce,double actual_value){
    double temp=2*(ce.predicted-actual_value);
    fill_constant(ce.gradients,temp,ce.filter_size);
    ce.get_gradient=fftwf_plan_dft_2d(ce.filter_size,ce.filter_size,get_complex_representation(ce.gradients,ce.filter_size),ce.fft_gradient,FFTW_FORWARD,FFTW_ESTIMATE);
    fftwf_execute(ce.get_gradient);
    mul(ce.fft_gradient,ce.fft_layer,ce.fft_kernel_gradient,ce.filter_size);
    ce.getBackkernel_gradient=fftwf_plan_dft_2d(ce.filter_size,ce.filter_size,ce.fft_kernel_gradient,ce.kernel_gradient,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftwf_execute(ce.getBackkernel_gradient);
    scale_comp(ce.kernel_gradient,ce.filter_size);
    write_to_real_mat(ce.kernel_gradient,ce.kernel_gradient_real,ce.filter_size);
}



//activation functioin needs to added here.
void forwardPass_convolution_enviorment(convolution_enviorment ce){
    ce.get_layer=fftwf_plan_dft_2d(ce.filter_size,ce.filter_size,get_complex_representation(ce.layer,ce.filter_size),ce.fft_layer,FFTW_FORWARD,FFTW_ESTIMATE);
    ce.get_kernel=fftwf_plan_dft_2d(ce.filter_size,ce.filter_size,get_complex_representation(ce.kernel,ce.filter_size),ce.fft_kernel,FFTW_FORWARD,FFTW_ESTIMATE);
    fftwf_execute(ce.get_layer);
    fftwf_execute(ce.get_kernel);
    mul(ce.fft_layer,ce.fft_kernel,ce.fft_mul,ce.filter_size);
    ce.getBack_mul=fftwf_plan_dft_2d(ce.filter_size,ce.filter_size,ce.fft_mul,ce.mul,FFTW_BACKWARD,FFTW_ESTIMATE);
    fftwf_execute(ce.getBack_mul);
    scale_comp(ce.mul,ce.filter_size);
    write_to_real_mat(ce.mul,ce.real_mul,ce.filter_size);
}

//this method is incomplete.
void destroy_convolutional_enviorment(convolution_enviorment ce){
    delete[] ce.fft_kernel;
    delete[] ce.fft_layer;
    delete[] ce.fft_mul;
    delete[] ce.mul;
    delete[] ce.layer;
    delete[] ce.kernel;
    delete[] ce.real_mul;
    //and destroy other plans.
} 


//method will get changes a lot
double psudoFullyConnected(double* output,int n){
    double sum=0;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            sum+=output[ind_ele(n,i,j)];
        }
    }
    return sum;
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
void trainIteration(convolution_enviorment ce){
    forwardPass_convolution_enviorment(ce);
    //cout<<"output after forward propogation is "<<endl;
    //print_mat(ce.real_mul,ce.filter_size);
    ce.predicted=psudoFullyConnected(ce.real_mul,ce.filter_size);
    cout<<"the predicted value is "<<ce.predicted<<endl;
    cout<<"the error is "<<(ce.predicted-20)*(ce.predicted-20)<<endl;
    backPropogate(ce,20);
    updatekernel(ce);
}
int main(){
    double* k=create_matrix(3);
    fill_random(k,3);
    convolution_enviorment ce=create_convolution_enviorment(5);
    fill_random(ce.layer,ce.filter_size);
    insialize_backProp(&ce,0.0001);
    preProcessKernel_static(k,ce.kernel,3,5);
    for(int i=0;i<10;i++)
    trainIteration(ce);
    return 0;
}






//layer=[[1,1,1,2,1],[8,3,2,6,2],[4,5,6,3,3],[1,1,1,1,1],[3,3,3,4,5]]
//kernel=[[1,1,2],[3,1,2],[4,5,5]]
/*

convolution_enviorment ce=create_convolution_enviorment(5);
    double* kernel=create_matrix(3);
    cout<<"enter the value in the layer"<<endl;
    fill_user(ce.layer,5);
    cout<<"enter the value in the kernel"<<endl;
    fill_user(kernel,3);
    preProcessKernel_static(kernel,ce.kernel,3,5);
    cout<<"doing the forward pass"<<endl;
    forwardPass_convolution_enviorment(ce);
    cout<<"the output generated by the forward pass is "<<endl;
    print_mat(ce.real_mul,ce.filter_size);
    print_comp_mat(ce.mul,ce.filter_size);
    return 0;









double* l=create_matrix(5);
    double* k=create_matrix(3);
    cout<<"enter the elements of the layler"<<endl;
    fill_user(l,5);
    cout<<"enter the elemnts of the kernel"<<endl;
    fill_user(k,3);
    double* ck=preProcessKernel(k,3,5);
    fftwf_complex* output=new fftwf_complex[5*5];
    double* output2=create_matrix(5);
    fft_conv(l,ck,5,output);
    conv(l,k,5,3,output2);
    cout<<"the convolution operation gives"<<endl;
    print_comp_mat(output,5);
    print_mat(output2,5);

*/