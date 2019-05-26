#include <math.h>#include <stdio.h>#include <strings.h>#include <stdlib.h>int main()
{
	int n = 5, m = 5, i, j, t, k, r;

	float	max_a, max_s, B;
	float	A[n][m], b[n], a, s[n], sum;
	float	L[n][m], U[n][m], PA[n][m], P[n][n];
	float	x[n], y[n];
	/*	 * a,B：行交换变量  i，j：计步器	 * max_x 记录最大值 a主元素最大 s分解值最大	 * t：行记录 k,q:角标	 * sum：累计求和	 */


	/* 输入矩阵A */
	printf( "Please enter the array A:\n" );
	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < m; j++ )
		{
			scanf( "%f", &A[i][j] );
		}
	}
	printf( "array A:\n" );
	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			printf( "%f  ", A[i][j] );
		}
		printf( "\n" );
	}
	/* 输入矩阵b */
	printf( "Please enter the array b:\n" );
	for ( i = 0; i < n; i++ )
	{
		scanf( "%f", &b[i] );
	}
	printf( "array b:\n" );
	for ( i = 0; i < n; i++ )
	{
		printf( "%f ", b[i] );
		printf( "\n" );
	}

	/* 输入单位阵P */
	for ( i = 0; i < n; i++ )
		for ( j = 0; j < n; j++ )
		{
			P[i][j] = 0;
			P[i][i] = 1;
		}
	printf( "array P:\n" );
	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			printf( "%f  ", P[i][j] );
		}
		printf( "\n" );
	}

	/* 第一行主元交换 */
	max_a = A[0][0], t = 0;
	for ( i = 1; i < n; i++ )
	{
		if ( A[i][0] > max_a )          /* 寻找第1列中最大值 */
		{
			t = i; max_a = A[i][0]; /* 记录行和数值 */
		}
		printf( "max is %d,%f\n", t, max_a );
	}
	for ( j = 0; j < m; j++ )               /* 交换系数矩阵行 */
	{
		a	= A[0][j];
		A[0][j] = A[t][j];
		A[t][j] = a;
	}
	printf( "array A:\n" );
	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			printf( "%f  ", A[i][j] );
		}
		printf( "\n" );
	}

	for ( j = 0; j < n; j++ ) /* 交换单位矩阵行 */
	{
		a	= P[0][j];
		P[0][j] = P[t][j];
		P[t][j] = a;
	}
	printf( "array P:\n" );
	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			printf( "%f  ", P[i][j] );
		}
		printf( "\n" );
	}

	/* 交换常数项行 */

	B	= b[0];
	b[0]	= b[t];
	b[t]	= B;

	printf( "array b:\n" );
	for ( i = 0; i < n; i++ )
	{
		printf( "%f  ", b[i] );
		printf( "\n" );
	}


	/* 计算L[n][0],U[0][m] */
	for ( j = 0; j < n; j++ )
	{
		U[0][j] = A[0][j];
	}
	for ( i = 1; i < n; i++ )
	{
		L[i][0] = A[i][0] / U[0][0];
		L[i][i] = 1;
	}

	/* 紧凑储存 */
	for ( i = 0; i < n; i++ )
	{
		PA[i][0] = L[i][0];
	}
	for ( j = 0; j < n; j++ )
	{
		PA[0][j] = U[0][j];
	}
	for ( i = 1; i < n; i++ )
		for ( j = 1; j < n; j++ )
		{
			PA[i][j] = A[i][j];
		}
	printf( "第一次分解，array PA:\n" ); /* 输出第一步PA[i][j] */
	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			printf( "%f  ", PA[i][j] );
		}
		printf( "\n" );
	}
/* 以上正确 */


	/* 第r步分解交换主元 */
	for ( r = 1; r < n; r++ )               /* 第r步 */
	{
		for ( i = r; i < n; i++ )       /* 计算S[k]-S[n] */
		{
			sum = 0;
			for ( k = 0; k <= r - 1; k++ )
			{
				sum += L[i][k] * U[k][r];
			}
			s[i] = PA[i][r] - sum;
			printf( "分步计算的值s[%d] is %f .\n", i, s[i] );
		}
		max_s = fabs( s[r] );           /* 比较s[i]的最大值 */
		for ( i = r + 1; i < n; i++ )
		{
			if ( fabs( s[i] ) > max_s )
			{
				max_s	= fabs( s[i] );
				t	= i;    /* 记录最大值所在行 */
			}
		}
		printf( "s[%d]为最大值是：%f \n", t, max_s );

		/* 交换紧凑储存行的顺序 */
		for ( j = 0; j < n; j++ )
		{
			a		= PA[r][j];
			PA[r][j]	= PA[t][j];
			PA[t][j]	= a;
		}
		printf( "第%d步交换后的，array PA:\n", r ); /* 输出交换后的PA[i][j] */
		for ( i = 0; i < n; i++ )
		{
			for ( j = 0; j < n; j++ )
			{
				printf( "%f  ", PA[i][j] );
			}
			printf( "\n" );
		}
		/* 交换单位矩阵行的顺序 */
		for ( j = 0; j < n; j++ )
		{
			a	= P[r][j];
			P[r][j] = P[t][j];
			P[t][j] = a;
		}
		printf( "第%d步交换后的，array P:\n", r ); /* 输出交换后的P[i][j] */
		for ( i = 0; i < n; i++ )
		{
			for ( j = 0; j < n; j++ )
			{
				printf( "%f  ", P[i][j] );
			}
			printf( "\n" );
		}

		/* 交换系数矩阵行的顺序 */

		B	= b[r];
		b[r]	= b[t];
		b[t]	= B;

		printf( "第%d步交换后的，array b:\n", r ); /* 输出交换后的b[r] */
		for ( i = 0; i < n; i++ )
		{
			printf( "%f  ", b[i] );
			printf( "\n" );
		}

		/* 计算U和L */
		for ( i = r; i < n; i++ ) /* 计算U的第r行元素 */
		{
			sum = 0;
			for ( k = 0; k <= r - 1; k++ )
			{
				sum = sum + PA[r][k] * PA[k][i];
			}
			U[r][i] = PA[r][i] - sum;
			printf( "u=%f,PA=%f,sum=%f\n", U[r][i], PA[r][i], sum );
		}
		for ( i = r + 1; i < n; i++ ) /* 计算L的第r列元素 */
		{
			sum = 0;
			for ( k = 0; k <= r - 1; k++ )
			{
				sum	= sum + PA[i][k] * PA[k][r];
				s[i]	= PA[i][r] - sum;
			}
			L[i][r] = s[i] / U[r][r];
		}


		/* 储存U，L到P */
		for ( i = r; i < n; i++ )
		{
			PA[r][i]	= U[r][i];
			PA[i + 1][r]	= L[i + 1][r];
		}
		printf( "第%d步分解，array PA:\n", r + 1 ); /* 输出第r步PA[r][i] */
		for ( k = 0; k < n; k++ )
		{
			for ( j = 0; j < n; j++ )
			{
				printf( "%f  ", PA[k][j] );
			}
			printf( "\n" );
		}
	}

	/* 输出L，U，x，y; *//* 输出上三角矩阵U[n][n] */
	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			if ( j >= i )
			{
				U[i][j] = PA[i][j];
			}else
				U[i][j] = 0;
		}
	}

	printf( "array U:\n" );
	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			printf( "%f  ", U[i][j] );
		}
		printf( "\n" );
	}
/* 输出下三角矩阵L[n][n] */
	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			if ( j < i )
			{
				L[i][j] = PA[i][j];
			}
			if ( j == i )
			{
				L[i][i] = 1;
			}
			if ( j > i )
			{
				L[i][j] = 0;
			}
		}
	}

	printf( "array L:\n" );
	for ( i = 0; i < n; i++ )
	{
		for ( j = 0; j < n; j++ )
		{
			printf( "%f  ", L[i][j] );
		}
		printf( "\n" );
	}

	/* L*y=b；U*X=y; */
	y[0] = b[0];
	for ( i = 0; i < n; i++ )
	{
		sum = 0;
		for ( k = 0; k < i; k++ )
		{
			sum += L[i][k] * y[k];
		}
		y[i] = b[i] - sum / L[i][i];
	}
	printf( "array y is:\n" );
	for ( i = 0; i < n; i++ )
	{
		printf( "%3.4f ", y[i] );
		printf( "\n" );
	}


	x[4] = y[4] / U[4][4];
	for ( i = 3; i >= 0; i-- )
	{
		sum = 0;
		for ( k = i + 1; k < n; k++ )
		{
			sum += U[i][k] * x[k];
		}
		x[i] = (y[i] - sum) / U[i][i];
	}
	printf( "array x is:\n" );
	for ( i = 0; i < n; i++ )
	{
		printf( "%3.4f ", x[i] );
		printf( "\n" );
	}
	return(0);
}