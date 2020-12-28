/* how to compile : gcc -o select_traj select_traj.c */
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
  long int start=84900001;
  long int end  =84900500;
  FILE *fpL,*fpLout;
  FILE *fpR,*fpRout;
  char sL[80],sR[80];
  int cnt;

  if((fpL = fopen("trajLF0tr.csv", "r")) == NULL)
    {
      printf("cannot find trajLF0tr.csv");
      exit(1);
    }
  if((fpR = fopen("trajRT0tr.csv", "r")) == NULL)
    {
      printf("cannot find trajRT0tr.csv");
      exit(1);
    }
  if((fpLout = fopen("select_trajLF0tr.csv", "w")) == NULL)
    {
      printf("cannot find select_trajLF0tr.csv");
      exit(1);
    }
  if((fpRout = fopen("select_trajRT0tr.csv", "w")) == NULL)
    {
      printf("cannot find select_trajRT0tr.csv");
      exit(1);
    }

  for( cnt = 1; fgets(sL,80,fpL) != NULL && fgets(sR,80,fpR) != NULL; cnt++ )
    {
      if((start <= cnt) && (cnt <= end))
	{ 
	  fputs(sL,fpLout);
	  fputs(sR,fpRout);
	}
    }
  
  fclose(fpL);
  fclose(fpLout);
  fclose(fpR);
  fclose(fpRout);
}

