/* how to compile : gcc -o changecsv changecsv.c */

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
  int val;
  int val11;
  int cnt;
  FILE *fp;
  FILE *fpp;

  if((fp = fopen("timehistory", "r")) == NULL)
    {
      printf("cannot find timehistory");
      exit(1);
    }
  if((fpp = fopen("out_timehisitory.csv", "w")) == NULL)
    {
      printf("cannot find timehistory.csv");
      exit(1);
    }

  for(cnt=1;fscanf(fp,"%d",&val)!=EOF;cnt++)
    {
      fprintf(fpp,"%10d, %6.1lf\n",cnt*1000000,100.0*val/100000);
    }

  fclose(fp);
  fclose(fpp);
}

