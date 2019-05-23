#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define n 50
int main()
{
    int k=0;
    double x[n],y[n],z[n],t,fz,fm; //fz,fm分别为迭代公式分子分母
    x[0]=0.5;  //初值
    printf("x[0]=%f\n",x[0]);
    do
   {
       y[k]=log(2-pow(x[k],2.0) );
       z[k]=log(2-pow(y[k],2.0) );
       fz=x[k]*z[k]-pow(y[k],2.0);
       fm=z[k]-2*y[k]+x[k];
       x[k+1]=fz/fm;
       t=fabs(x[k+1]-x[k]);k++;
       printf("x[%d]=%f\n",k,x[k]);
   }while(t>1e-5);

    return 0;
}