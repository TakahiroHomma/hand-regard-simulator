
/* how to compile : gcc -o countLL0_m countLL0_m.c */

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
  int val;
  int sum[240000];
  int cnt,n;
  /*  int interval = 1000000;
      int step = 20000;*/
  int interval = 500000;
  int step = 10000; 
  /*  int interval = 5000000;
      int step = 100000;*/
  div_t d;
  FILE *fp;
  FILE *fpp;

  if((fp = fopen("findLL0tr", "r")) == NULL)
    {
      printf("cannot find findLL0tr");
      exit(1);
    }

  /*  if((fp = fopen("findLL0tr", "r")) == NULL)
    {
      printf("cannot find find0Ltr");
      exit(1);
      } */

  if((fpp = fopen("countoutLL0.csv", "w")) == NULL)
    {
      printf("cannot find countoutLL0.csv");
      exit(1);
    }


  for(cnt = 0; fscanf(fp,"%d", &val) != EOF; cnt++)
    {
      d = div(cnt,interval);      

      if((0<=d.rem) && (d.rem<=(step-1)))
	{ 
	  sum[d.quot] += val;
	}

    }

  for(n=1; n<= d.quot; n++)
    {
      fprintf(fpp,"%10d, %6.1lf\n",interval/2+(n-1)*interval
	      ,(100.0*(sum[n-1]+sum[n])/(2*step)));

	      /* fprintf(fpp,"%10d, %10d\n",
		 (d.quot-1)*interval,sum);*/
    }
 
  fclose(fp);
  fclose(fpp);
}

