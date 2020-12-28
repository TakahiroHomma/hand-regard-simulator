/* how to compile : gcc -o set_zero_cell set_zero_cell.c */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define max_units 294    

FILE *fp;
FILE *fpp;
FILE *fp2;
FILE *fp3;
FILE *fp4;
FILE *fp5;

char dmy[256],ss[256];

int 
/*number inputs */
in_mod,
/*number outputs */
  out_mod,
/*number hidden units */
  hid_mod, 
/*number input and hidden units */
  hi_in_mod,
/* number of all units */
  ges_mod;

int countend,nblock;
int i,j,k,iii,jjj,kkk,v,r,s,t,icount;
int sum;
double 
/* weight matrix */
  W_mod[max_units][max_units],
/*  W_mod_1_4[max_units][max_units],*/

/* change */
  W_mod_cell[max_units][max_units],

/*  W_mod_all[max_units][max_units],*/

  Yk_mod_new[max_units],
/* old activation for all units */
  Yk_mod_old[max_units],
/* dyk / dwij derivatives of units with respect to weights */
  Pk_ijm_mod[max_units][max_units][max_units], 
/* old values of dyk / dwij derivatives of units with respect to weights */
  Pk_ijm_mod_o[max_units][max_units][max_units];

double ra[max_units];
int cell_assembly[max_units];

double tmp;

int main(void)
{
  in_mod = 225;
  out_mod = 8;
  hid_mod = 48;
  in_mod++;
  in_mod = in_mod+out_mod;   /* add collorary discharge */
  in_mod = in_mod+4;         
  /* add proprioceptive estimates of hands position(x,y) */
  
  hi_in_mod = in_mod+hid_mod;
  ges_mod = hi_in_mod+out_mod;

  countend = hid_mod+out_mod;

  if((fp = fopen("ra_short", "r")) == NULL)
    {
      printf("cannot find ra_short");
      exit(1);
    }
  if((fp2 = fopen("number_file", "r")) == NULL)
    {
      printf("cannot find number_file");
      exit(1);
    }
  if((fp3 = fopen("file_weight", "r")) == NULL)
    {
      printf("cannot find file_weight");
      exit(1);
    }
  if((fp4 = fopen("cell_file_weight", "w")) == NULL)
    {
      printf("cannot find cell_file_weight");
      exit(1);
    }
  if((fp5 = fopen("cell_assembly_weight.csv", "a")) == NULL)
    {
      printf("cannot find cell_assembly_weight.csv");
      exit(1);
    }
  if((fpp = fopen("log_file", "w")) == NULL)
    {
      printf("cannot find log_file");
      exit(1);
    }

  /* read wei number */
  fscanf(fp2,"%d", &icount);

  fprintf(fpp,"%10d\n",icount);

  /* count block number */
  nblock=0;
  while( !feof(fp))
    {
      for(jjj=1; jjj<=(countend+4)*100; jjj++)
	{
	  fgets(ss,256,fp);
	}
      nblock++;
    }

  nblock--;
  
  rewind(fp);

  /* find cell assembly units */
  for(iii=1; iii<=nblock; iii++)
    {
      for(k=1; k<=countend; k++)
	{
	  cell_assembly[k] = 0;
	}
      for(jjj=1; jjj<=100; jjj++)
	{
	  fgets(ss, 256, fp);

	  fgets(ss, 256, fp);

	  fgets(ss, 256, fp);

	  fgets(ss, 256, fp);

	  for(kkk=1; kkk<=countend; kkk++)
	    {
	      fscanf(fp,"%9d%lf",&k,&ra[kkk]);
	    }
	  
	  fgets(dmy,256,fp);

	  if(((iii-2) == icount) && (jjj == 1))
	    {
	      for(k=1; k<=hid_mod; k++)
		{
		  if(ra[k] >= 0.899)
		    {
		      cell_assembly[k] = 1;
		      fprintf (fpp,"k=%d cell_assembly=%d\n",k,cell_assembly[k]);
		    }
		}

	      /* read weight file */
	      for (i=in_mod; i<ges_mod; i++)
		{
		  for (j=0; j<ges_mod; j++)
		    {
		      fscanf(fp3,"(%d,%d): %lf ", &r, &s, &W_mod[i][j]);
		      W_mod_cell[i][j] = W_mod[i][j];
		    }
		  fscanf(fp3,"\n");
		}

	      for (i=0;i<ges_mod;i++)
		{
		  fscanf(fp3,"i:%d %lf ",&r,&Yk_mod_new[i]);
		  Yk_mod_old[i] = Yk_mod_new[i];
		}
	      fscanf(fp3,"\n");
	      
	      for (i=in_mod;i<ges_mod;i++)
		{
		  for (j=in_mod;j<ges_mod;j++)
		    {      
		      for (v=0;v<ges_mod;v++)
			{
			  fscanf(fp3,"i:%d j:%d v:%d %lf ",&r,&s,&t,&Pk_ijm_mod[i][j][v]);
			  Pk_ijm_mod_o[i][j][v] = Pk_ijm_mod[i][j][v];
			}
		    }
		}
	      fscanf(fp3,"\n");

	      /* display weights of cell assembly units */ 
	      for(i=1; i<=hid_mod; i++)
		{
		  for(j=1; j<=hid_mod; j++)
		    {
		      if((cell_assembly[i] == 1) && (cell_assembly[j] == 1))
			{
			  if(fabs(W_mod[i+in_mod-1][j+in_mod-1]) >= 0.000001)
			    {
			      fprintf(fp5,"icount=,%10d, (%.2d,%.2d):, %f\n",icount,i,j,W_mod[i+in_mod-1][j+in_mod-1]);
			    }
			}
		    }
		}

	      /* input weights of cell assembly units into zero */ 
	      for(k=1; k<=hid_mod; k++)
		{
		  if(cell_assembly[k] == 1)
		    {
		      /* change */
		      for(j=0; j<in_mod; j++)
			{
			  W_mod_cell[k+in_mod-1][j] = 0.0;
			}

		      for (i=in_mod;i<hi_in_mod;i++)
			{
			  if(cell_assembly[i-in_mod+1] == 0)
			    {
			      W_mod_cell[i][k+in_mod-1] = 0.0;
			    }
			}

		      for(j=in_mod; j<hi_in_mod; j++)
			{
			  if(cell_assembly[j-in_mod+1] == 0)
			    {
			      W_mod_cell[k+in_mod-1][j] = 0.0;
			    }
			}

		      for (i=hi_in_mod;i<ges_mod;i++)
			{		
			  W_mod_cell[i][k+in_mod-1] = 0.0;
			}
		    }
		}

	      /* write weight file */
	      for (i=in_mod;i<ges_mod;i++)
		{
		  for (j=0;j<ges_mod;j++)
		    {
		      fprintf(fp4,"(%.2d,%.2d): %f ",i,j,W_mod_cell[i][j]);
		    }
		  fprintf(fp4,"\n");
		}

	      for (i=0;i<ges_mod;i++)
		{
		  fprintf(fp4,"i:%d %f ",i,Yk_mod_new[i]);
		}
	      fprintf(fp4,"\n");

	      for (i=in_mod;i<ges_mod;i++)
		{
		  for (j=in_mod;j<ges_mod;j++)
		    {      
		      for (v=0;v<ges_mod;v++)
			{
			  fprintf(fp4,"i:%d j:%d v:%d %f ",i,j,v,Pk_ijm_mod[i][j][v]);
			}
		    }
		}

	      fprintf(fp4,"\n");
	    }
	}
    }


  fclose(fp);

  fclose(fp2);
  fclose(fp3);
  fclose(fp4);
  fclose(fp5);
  fclose(fpp);
}

