#include<stdio.h>#include<math.h>int main(void) 
{
	float A[6][6],b[6];
	float x[6] = {0};
	//第k+1次迭代的结果
	float xx[6] = {0};
	//第k次迭代的结果
	int size = 6,n=6,m=6,max=100;
	//最大迭代次数
	float residual = 0.0;
	float sum = 0.0,dis = 0.0,dif = 1.0;
	//相邻迭代的结果差
	float eps = 1.0e-4;
	//迭代精度
	/* 输入矩阵A */
	printf( "Please enter the array A:n" );
	for (  int i = 0; i < n; i++ )
	{
		for (int j = 0; j < m; j++ )
		{
			scanf( "%f", &A[i][j] );
		}
	}
	/* 输入矩阵b */
	printf( "Please enter the array b:n" );
	for (  int i = 0; i < n; i++ )
	{
		scanf( "%f", &b[i] );
	}
 for(int k=1;(k<Max)&&(dif>eps);k++)
 {
  dif = 0.0;
  printf("n第%d次迭代的结果：n",k);
  for(int i=0;i<size;i++)
  {
   for(int j=0;j<size;j++)
   {
    if(i!=j)
    {
     sum +=A[i][j]*xx[j];
    }
   }
   x[i] = (b[i]-sum)/A[i][i];
   sum = 0.0;
  }
  residual=0.0;
  //计算相邻迭代的结果差
  for(int m=0;m<size;m++)
  {
   dis=fabs(x[m]-xx[m]);
   if(dis>residual)
    residual=dis;
  }
  dif=residual;
  //打印第k次的结果
  for(int i=0;i<size;i++)
  {
   printf("%12.4f ",x[i]);
   xx[i]=x[i];
  }
 }
  printf("n迭代计算的结果为：n");for(int k=0;k<size;k++)
  {
   printf("%12.4f ",xx[k]);
  }
  printf("n"