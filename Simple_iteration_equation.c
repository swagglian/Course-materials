#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main()
{
    int k,n=100;
    double x[n];
    double df;
    x[0]=0.5;   //设初值为0.5
    for(k=0;fabs(x[k+1]-x[k])>pow(10,-5);k++)//循环直到精度为pow(10,-5)
   {
       x[k+1]=log(2- pow(x[k],2.0) );
       df=(2*x[k])/(2-pow(x[k],2));
       if(fabs(df)>0.75) break;  //判断收敛性
       printf("x[%d]=%f\n",k,x[k]);
   }
    return 0;
}