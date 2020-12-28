
/* how to compile : gcc -o counttestLL counttestLL.c */

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
  int val;
  int sum0 = 0;
  int sum1 = 0;
  int cnt;
  int flag = 0;
  FILE *fp0;
  /*  FILE *fp1;*/
  FILE *fpp;

  if((fp0 = fopen("findLL0", "r")) == NULL)
    {
      printf("cannot find findLL0");
      exit(1);
    }
  /*  if((fp1 = fopen("find1", "r")) == NULL)
    {
      printf("cannot find find1");
      exit(1);
      }*/
  if((fpp = fopen("out_counttestLL", "w")) == NULL)
    {
      printf("cannot find out_counttestLL");
      exit(1);
    }


  for( cnt = 0; fscanf(fp0,"%d", &val) != EOF; cnt++ )
    {
      sum0 += val;
    }

  /*  for( cnt = 0; fscanf(fp1,"%d", &val) != EOF; cnt++ )
    {
      sum1 += val;
      }*/

  /*  fprintf(fpp,"0: %10d\n1: %10d\n",sum0,sum1);*/
  fprintf(fpp,"%10d\n",sum0);

  fclose(fp0);
  /*  fclose(fp1);*/
  fclose(fpp);
}

