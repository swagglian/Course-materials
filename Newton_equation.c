#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define n 50
int main()
{
    int k=0;
    double x[n],f,df,tx,tf;
    x[0]=0.5;
    printf("x[0]=%f\n",x[0]);
   for(k=0;k<=n;k++)
   {
       f = exp(x[k])+pow(x[k],2)-2;
       df = exp(x[k])+2*x[k];
       x[k+1] = x[k]-f/df;
       tx = fabs(x[k+1]-x[k]);
       tf = fabs(f);
       printf("x[%d]=%f\n",k,x[k]);
       if(tx<pow(10,-5)||tf<pow(10,-5)) break;

   }
  if(k>n)
        printf("迭代失败。\n");
  else
    printf("x[%d]=%f\n",k+1,x[k+1]);
    printf("f(x[%d])=%f\n",k,f);
    printf("迭代次数:%d。",k);
    return 0;
}