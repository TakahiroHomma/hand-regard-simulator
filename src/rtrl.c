#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define max_units 294    
/* number inputs(in "rtrlpars.txt")+bias("1")
  +number of corollary discharge(=number outputs(in "rtrlpars.txt"))
  +4(proprioceptive estimates of hands position(x,y))
  +number hidden units(in "rtrlpars.txt")+number outputs (in "rtrlpars.txt") */
#define vision_bias 226     /* number inputs (in "rtrlpars.txt")+bias(1) */
#define max_outputs 8      /* number outputs (in "rtrlpars.txt") */
#define max_sequence_length 1
#define max_training_size 1
#define max_test_size 1

/* modified in 02/06/2010 */

#define MAX_AREA 31             /* length of side of field,*/
                                /* where hand and others move */
#define NUM_OF_OBJ_TRAIN 2      /* number of others in training */ 
#define NUM_OF_OBJ_TEST 2       /* number of others in test */   
#define UNIT_RANGE 8            /* width of expression for RA[] in AVS */
#define INTERVAL_OF_SWEEPS_TRAIN 1000 /* interval of resetting the position */
                                     /* of hand and others in training*/
#define INTERVAL_OF_SWEEPS_TEST 1000 /* interval of resetting the position */
                                    /* of hand and others in test*/
#define INTERVAL_OF_AVS 1000000  /* interval of outputting AVS data */

#define AVS_RANGE 100           /* output AVS data for (AVS_RANGE)sweeps */
                                /* every (INTERVAL_OF_AVS)sweeps */ 
#define INVARIANT_STEPS 70      /* when RA[] of certain unit */
                                /* does not change, output 1.0 */
                                /* for the unit in AVS data.*/
				/* Currently, it is not used */ 
#define interval_of_backward_pass 10 /* interval of training */                 
#define INTERVAL_OF_RESTART 1000000
                                /* interval of outputting restart */
                                /* file wei*.par */

FILE *fp1,*fp2;
char *outfile,*weightfile;
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
  ges_mod,
  /* current sequence of the training set */
  examples,
  /* random number generator init */
  ran_sta,
  /* performing a test on a test  set? if ((epoch%test_aus==0)||(class_err<min_fehl)) */
  test_aus,
  min_fehl,
  /* write the weight matrix after w_out epochs */
  w_out,
  /* should the net still be learning --> learn=1 */
  learn,
  /* is stop to learn set? --> stop_learn=1 */
  stop_learn,
  /* max. number of trials to be performed */
  maxtrials,
  /* reset after each sequence? */
  sequ_reset,
  training_size,
  test_size,
  /* max. number of epochs before learning stop*/
  maxepoch,
  /* bias1==1 --> units biased ? */
  bias1,
  /* is target provided for the current input */
  targ,
  /* element in a sequence of the test set */
  element_t,
  /* number of sequences in test set */
  example_t,
  /* misclassifications per epoch */
  class_err,
  /* when to stop learning: wrong classifications per epoch < stop_class */
  stop_class,
  /* epochs to be learned after stop learning is set */
  ext_epochs,
  /* 1 if the current sequence has been processed correctly so far */
  seq_cor,
  /* 1 if end of the test set */
  test_end,
  /* weight_up == 1 then weight update */
  weight_up,
  /* w_up if w_up == 1 then weight update per sequence otherwise per epoch*/
  w_up,
  /* element in a sequence of the training set */
  element,
  /* number of sequences in training set */
  example,
  /* number of sequences seen by the net */
  numb_seq,
  /* number of epochs seen by the net */
  epoch,
  /* length of the training sequences */
  length[max_training_size],
  /* length of the test sequences */
  length_t[max_test_size];

double 
  /* weight matrix */
  W_mod[max_units][max_units],
  /* contribution to update of weight matrix */
  DW[max_units][max_units],
  /* new activation for all units */
  Yk_mod_new[max_units],
  /* old activation for all units */
  Yk_mod_old[max_units],
  /* dyk / dwij derivatives of units with respect to weights */
  Pk_ijm_mod[max_units][max_units][max_units], 
  /* old values of dyk / dwij derivatives of units with respect to weights */
  Pk_ijm_mod_o[max_units][max_units][max_units],
  /* learing rate */
  alpha, 
  /* interval of weight initialization is init_range*[-1,1] */
  init_range, 
  /* current target */
  target_a[max_units],
/* matrix for storing the target per training element per
     seqence element per output component */
  tar[max_training_size][max_sequence_length][max_outputs],
/* matrix for storing the input per training element per
   seqence element per input component */
  inp[max_training_size][max_sequence_length][vision_bias+max_outputs+4],
  /* matrix for storing the target per test element per
seqence element per output component */
  tar_t[max_test_size][max_sequence_length][max_outputs],
  /* matrix for storing the input per test element per
seqence element per input component */
  inp_t[max_test_size][max_sequence_length][vision_bias+max_outputs+4],
  /* MSE per epoch */
  epoch_err,
  /* MSE per sequence */
  seq_err,
  /* when to stop learning: MSE per epoch < stop_mse */
  stop_mse,
  /* error for output units */
  error[max_units],
  /* max. allowed error per sequence for correct classification */
  seq_max; 

/* modified in 02/06/2010 */
int mm=0;
int nn=0;
int iii,jjj,ix,iy,tmp_elem,tmp_node_x,tmp_node_y;
int nx=0;
int num=0;
int tmp_position;
int change_sweeps=0;
int change_sweeps_flag;
int recognition;
int node_start,node_max;
int judge_invariant;
int NUM_OF_OBJ;
int INTERVAL_OF_SWEEPS;
int find_flag;
div_t d1,dnode,delem;
double right,left;
double copy;
int
  length_field,
  half_length_field;
int
  min_field_x,
  min_field_y,
  max_field_x,
  max_field_y,
  nearest_obj,
  number_of_inputs;
double
  distance,
  min_distance,
  field_avs[MAX_AREA*MAX_AREA+1];
int
  self_leg_0,
  other,
  other2,
  other3,
  other_leg_0,
  hand, 
  self_leg_0x,
  self_leg_0y,
  other_leg_0x,
  other_leg_0y,
  handx,
  handy,
  training_flag_0,
  sweeps_debug,
  speed,
  speed1,
  which_side,
  which_dir,
  which_dir1,
  which_dir2,
  which_dir3,
  which_dir4;

int test_flag,restart_flag;
int file_number;
int 
  obj_x[vision_bias],
  obj_y[vision_bias],
  obj_x_tmp[vision_bias],
  obj_y_tmp[vision_bias],
  obj_old_x[vision_bias],
  obj_old_y[vision_bias],
  center_field_x,
  center_field_y,
  center_field_x_tmp,
  center_field_y_tmp,
  check_obj[vision_bias],
  check_obj_old[vision_bias],
  elem_field[(MAX_AREA+1)*(MAX_AREA+1)+1],
  elem_unit[4*(max_units - vision_bias - max_outputs - 4)+1];

double
  INPUTS[vision_bias],
  flag_step[max_outputs],
  INPUTS_init[vision_bias];

int sweeps,max_sweeps;

double tmp_hand_x,tmp_hand_y,tmp_hand_1_x,tmp_hand_1_y;

FILE *fp3;  /* read restart file -> weight_in() */

FILE *ffind0;
FILE *ffind0tr;
FILE *ffind1;
FILE *ffind1tr;
FILE *ffind2;
FILE *ffind2tr;
FILE *ffindL0;
FILE *ffindL0tr;
FILE *ffindL1;
FILE *ffindL1tr;
FILE *ffindL2;
FILE *ffindL2tr;
FILE *ffindLL0;
FILE *ffindLL0tr;
FILE *ffindLL1;
FILE *ffindLL1tr;
FILE *ffindLL2;
FILE *ffindLL2tr;
FILE *ftrajLF0;
FILE *ftrajLF0tr;
FILE *ftrajRT0;
FILE *ftrajRT0tr;
FILE *ftrajLF1;
FILE *ftrajLF1tr;
FILE *ftrajRT1;
FILE *ftrajRT1tr;
FILE *ftrajLF2;
FILE *ftrajLF2tr;
FILE *ftrajRT2;
FILE *ftrajRT2tr;
FILE *fcheck4;
FILE *fcheck4atr;
FILE *fcheck4btr;
FILE *fcheck4ctr;
FILE *fcheck4dtr;
FILE *fcheck5;
FILE *fcheck5atr;
FILE *fcheck5btr;
FILE *fcheck5ctr;
FILE *fcheck5dtr;
FILE *fchecka;
FILE *fcheckatr;
FILE *fcheckb;
FILE *fcheckbtr;
FILE *fcheckc;
FILE *fcheckctr;
FILE *fcheckd;
FILE *fcheckdtr;
FILE *ferrortest;
FILE *ferrortr;

/* the output and weight files for 20 trials */
static char
*outf[] = {           "outa.txt",
		      "outb.txt",
		      "outc.txt",
		      "outd.txt",
		      "oute.txt",
		      "outf.txt",
		      "outg.txt",
		      "outh.txt",
		      "outi.txt",
		      "outj.txt",
		      "outk.txt",
		      "outl.txt",
		      "outm.txt",
		      "outn.txt",
		      "outo.txt",
		      "outp.txt",
		      "outq.txt",
		      "outr.txt",
		      "outs.txt",
		      "outt.txt"         },

  *weig[] = {           "wei0.par",
			"wei1.par",
			"wei2.par",
			"wei3.par",
			"wei4.par",
			"wei5.par",
			"wei6.par",
			"wei7.par",
			"wei8.par",
			"wei9.par",
			"wei10.par",
			"wei11.par",
			"wei12.par",
			"wei13.par",
			"wei14.par",
			"wei15.par",
			"wei16.par",
			"wei17.par",
			"wei18.par",
			"wei19.par",
			"wei20.par",
			"wei21.par",
			"wei22.par",
			"wei23.par",
			"wei24.par",
			"wei25.par",
			"wei26.par",
			"wei27.par",
			"wei28.par",
			"wei29.par",
			"wei30.par",
			"wei31.par",
			"wei32.par",
			"wei33.par",
			"wei34.par",
			"wei35.par",
			"wei36.par",
			"wei37.par",
			"wei38.par",
			"wei39.par",
			"wei40.par",
			"wei41.par",
			"wei42.par",
			"wei43.par",
			"wei44.par",
			"wei45.par",
			"wei46.par",
			"wei47.par",
			"wei48.par",
			"wei49.par",
			"wei50.par",
			"wei51.par",
			"wei52.par",
			"wei53.par",
			"wei54.par",
			"wei55.par",
			"wei56.par",
			"wei57.par",
			"wei58.par",
			"wei59.par",
			"wei60.par",
			"wei61.par",
			"wei62.par",
			"wei63.par",
			"wei64.par",
			"wei65.par",
			"wei66.par",
			"wei67.par",
			"wei68.par",
			"wei69.par",
			"wei70.par",
			"wei71.par",
			"wei72.par",
			"wei73.par",
			"wei74.par",
			"wei75.par",
			"wei76.par",
			"wei77.par",
			"wei78.par",
			"wei79.par",
			"wei80.par",
			"wei81.par",
			"wei82.par",
			"wei83.par",
			"wei84.par",
			"wei85.par",
			"wei86.par",
			"wei87.par",
			"wei88.par",
			"wei89.par",
			"wei90.par",
			"wei91.par",
			"wei92.par",
			"wei93.par",
			"wei94.par",
			"wei95.par",
			"wei96.par",
			"wei97.par",
			"wei98.par",
			"wei99.par",
			"wei100.par",
			"wei101.par",
			"wei102.par",
			"wei103.par",
			"wei104.par",
			"wei105.par",
			"wei106.par",
			"wei107.par",
			"wei108.par",
			"wei109.par",
			"wei110.par",
			"wei111.par",
			"wei112.par",
			"wei113.par",
			"wei114.par",
			"wei115.par",
			"wei116.par",
			"wei117.par",
			"wei118.par",
			"wei119.par",
			"wei120.par",
			"wei121.par",
			"wei122.par",
			"wei123.par",
			"wei124.par",
			"wei125.par",
			"wei126.par",
			"wei127.par",
			"wei128.par",
			"wei129.par",
			"wei130.par",
			"wei131.par",
			"wei132.par",
			"wei133.par",
			"wei134.par",
			"wei135.par",
			"wei136.par",
			"wei137.par",
			"wei138.par",
			"wei139.par",
			"wei140.par",
			"wei141.par",
			"wei142.par",
			"wei143.par",
			"wei144.par",
			"wei145.par",
			"wei146.par",
			"wei147.par",
			"wei148.par",
			"wei149.par",
			"wei150.par",
			"wei151.par",
			"wei152.par",
			"wei153.par",
			"wei154.par",
			"wei155.par",
			"wei156.par",
			"wei157.par",
			"wei158.par",
			"wei159.par",
			"wei160.par",
			"wei161.par",
			"wei162.par",
			"wei163.par",
			"wei164.par",
			"wei165.par",
			"wei166.par",
			"wei167.par",
			"wei168.par",
			"wei169.par",
			"wei170.par",
			"wei171.par",
			"wei172.par",
			"wei173.par",
			"wei174.par",
			"wei175.par",
			"wei176.par",
			"wei177.par",
			"wei178.par",
			"wei179.par",
			"wei180.par",
			"wei181.par",
			"wei182.par",
			"wei183.par",
			"wei184.par",
			"wei185.par",
			"wei186.par",
			"wei187.par",
			"wei188.par",
			"wei189.par",
			"wei190.par",
			"wei191.par",
			"wei192.par",
			"wei193.par",
			"wei194.par",
			"wei195.par",
			"wei196.par",
			"wei197.par",
			"wei198.par",
			"wei199.par",
			"wei200.par",
			"wei201.par",
			"wei202.par",
			"wei203.par",
			"wei204.par",
			"wei205.par",
			"wei206.par",
			"wei207.par",
			"wei208.par",
			"wei209.par",
			"wei210.par",
			"wei211.par",
			"wei212.par",
			"wei213.par",
			"wei214.par",
			"wei215.par",
			"wei216.par",
			"wei217.par",
			"wei218.par",
			"wei219.par",
			"wei220.par",
			"wei221.par",
			"wei222.par",
			"wei223.par",
			"wei224.par",
			"wei225.par",
			"wei226.par",
			"wei227.par",
			"wei228.par",
			"wei229.par",
			"wei230.par",
			"wei231.par",
			"wei232.par",
			"wei233.par",
			"wei234.par",
			"wei235.par",
			"wei236.par",
			"wei237.par",
			"wei238.par",
			"wei239.par",
			"wei240.par",
			"wei241.par",
			"wei242.par",
			"wei243.par",
			"wei244.par",
			"wei245.par",
			"wei246.par",
			"wei247.par",
			"wei248.par",
			"wei249.par",
			"wei250.par",
			"wei251.par",
			"wei252.par",
			"wei253.par",
			"wei254.par",
			"wei255.par",
			"wei256.par",
			"wei257.par",
			"wei258.par",
			"wei259.par",
			"wei260.par",
			"wei261.par",
			"wei262.par",
			"wei263.par",
			"wei264.par",
			"wei265.par",
			"wei266.par",
			"wei267.par",
			"wei268.par",
			"wei269.par",
			"wei270.par",
			"wei271.par",
			"wei272.par",
			"wei273.par",
			"wei274.par",
			"wei275.par",
			"wei276.par",
			"wei277.par",
			"wei278.par",
			"wei279.par",
			"wei280.par",
			"wei281.par",
			"wei282.par",
			"wei283.par",
			"wei284.par",
			"wei285.par",
			"wei286.par",
			"wei287.par",
			"wei288.par",
			"wei289.par",
			"wei290.par",
			"wei291.par",
			"wei292.par",
			"wei293.par",
			"wei294.par",
			"wei295.par",
			"wei296.par",
			"wei297.par",
			"wei298.par",
			"wei299.par",
			"wei300.par",
			"wei301.par",
			"wei302.par",
			"wei303.par",
			"wei304.par",
			"wei305.par",
			"wei306.par",
			"wei307.par",
			"wei308.par",
			"wei309.par",
			"wei310.par",
			"wei311.par",
			"wei312.par",
			"wei313.par",
			"wei314.par",
			"wei315.par",
			"wei316.par",
			"wei317.par",
			"wei318.par",
			"wei319.par",
			"wei320.par",
			"wei321.par",
			"wei322.par",
			"wei323.par",
			"wei324.par",
			"wei325.par",
			"wei326.par",
			"wei327.par",
			"wei328.par",
			"wei329.par",
			"wei330.par",
			"wei331.par",
			"wei332.par",
			"wei333.par",
			"wei334.par",
			"wei335.par",
			"wei336.par",
			"wei337.par",
			"wei338.par",
			"wei339.par",
			"wei340.par",
			"wei341.par",
			"wei342.par",
			"wei343.par",
			"wei344.par",
			"wei345.par",
			"wei346.par",
			"wei347.par",
			"wei348.par",
			"wei349.par",
			"wei350.par",
			"wei351.par",
			"wei352.par",
			"wei353.par",
			"wei354.par",
			"wei355.par",
			"wei356.par",
			"wei357.par",
			"wei358.par",
			"wei359.par",
			"wei360.par",
			"wei361.par",
			"wei362.par",
			"wei363.par",
			"wei364.par",
			"wei365.par",
			"wei366.par",
			"wei367.par",
			"wei368.par",
			"wei369.par",
			"wei370.par",
			"wei371.par",
			"wei372.par",
			"wei373.par",
			"wei374.par",
			"wei375.par",
			"wei376.par",
			"wei377.par",
			"wei378.par",
			"wei379.par",
			"wei380.par",
			"wei381.par",
			"wei382.par",
			"wei383.par",
			"wei384.par",
			"wei385.par",
			"wei386.par",
			"wei387.par",
			"wei388.par",
			"wei389.par",
			"wei390.par",
			"wei391.par",
			"wei392.par",
			"wei393.par",
			"wei394.par",
			"wei395.par",
			"wei396.par",
			"wei397.par",
			"wei398.par",
			"wei399.par",
			"wei400.par",
			"wei401.par",
			"wei402.par",
			"wei403.par",
			"wei404.par",
			"wei405.par",
			"wei406.par",
			"wei407.par",
			"wei408.par",
			"wei409.par",
			"wei410.par",
			"wei411.par",
			"wei412.par",
			"wei413.par",
			"wei414.par",
			"wei415.par",
			"wei416.par",
			"wei417.par",
			"wei418.par",
			"wei419.par",
			"wei420.par",
			"wei421.par",
			"wei422.par",
			"wei423.par",
			"wei424.par",
			"wei425.par",
			"wei426.par",
			"wei427.par",
			"wei428.par",
			"wei429.par",
			"wei430.par",
			"wei431.par",
			"wei432.par",
			"wei433.par",
			"wei434.par",
			"wei435.par",
			"wei436.par",
			"wei437.par",
			"wei438.par",
			"wei439.par",
			"wei440.par",
			"wei441.par",
			"wei442.par",
			"wei443.par",
			"wei444.par",
			"wei445.par",
			"wei446.par",
			"wei447.par",
			"wei448.par",
			"wei449.par",
			"wei450.par",
			"wei451.par",
			"wei452.par",
			"wei453.par",
			"wei454.par",
			"wei455.par",
			"wei456.par",
			"wei457.par",
			"wei458.par",
			"wei459.par",
			"wei460.par",
			"wei461.par",
			"wei462.par",
			"wei463.par",
			"wei464.par",
			"wei465.par",
			"wei466.par",
			"wei467.par",
			"wei468.par",
			"wei469.par",
			"wei470.par",
			"wei471.par",
			"wei472.par",
			"wei473.par",
			"wei474.par",
			"wei475.par",
			"wei476.par",
			"wei477.par",
			"wei478.par",
			"wei479.par",
			"wei480.par",
			"wei481.par",
			"wei482.par",
			"wei483.par",
			"wei484.par",
			"wei485.par",
			"wei486.par",
			"wei487.par",
			"wei488.par",
			"wei489.par",
			"wei490.par",
			"wei491.par",
			"wei492.par",
			"wei493.par",
			"wei494.par",
			"wei495.par",
			"wei496.par",
			"wei497.par",
			"wei498.par",
			"wei499.par",
			"wei500.par",
			"wei501.par",
			"wei502.par",
			"wei503.par",
			"wei504.par",
			"wei505.par",
			"wei506.par",
			"wei507.par",
			"wei508.par",
			"wei509.par",
			"wei510.par",
			"wei511.par",
			"wei512.par",
			"wei513.par",
			"wei514.par",
			"wei515.par",
			"wei516.par",
			"wei517.par",
			"wei518.par",
			"wei519.par",
			"wei520.par",
			"wei521.par",
			"wei522.par",
			"wei523.par",
			"wei524.par",
			"wei525.par",
			"wei526.par",
			"wei527.par",
			"wei528.par",
			"wei529.par",
			"wei530.par",
			"wei531.par",
			"wei532.par",
			"wei533.par",
			"wei534.par",
			"wei535.par",
			"wei536.par",
			"wei537.par",
			"wei538.par",
			"wei539.par",
			"wei540.par",
			"wei541.par",
			"wei542.par",
			"wei543.par",
			"wei544.par",
			"wei545.par",
			"wei546.par",
			"wei547.par",
			"wei548.par",
			"wei549.par",
			"wei550.par",
			"wei551.par",
			"wei552.par",
			"wei553.par",
			"wei554.par",
			"wei555.par",
			"wei556.par",
			"wei557.par",
			"wei558.par",
			"wei559.par",
			"wei560.par",
			"wei561.par",
			"wei562.par",
			"wei563.par",
			"wei564.par",
			"wei565.par",
			"wei566.par",
			"wei567.par",
			"wei568.par",
			"wei569.par",
			"wei570.par",
			"wei571.par",
			"wei572.par",
			"wei573.par",
			"wei574.par",
			"wei575.par",
			"wei576.par",
			"wei577.par",
			"wei578.par",
			"wei579.par",
			"wei580.par",
			"wei581.par",
			"wei582.par",
			"wei583.par",
			"wei584.par",
			"wei585.par",
			"wei586.par",
			"wei587.par",
			"wei588.par",
			"wei589.par",
			"wei590.par",
			"wei591.par",
			"wei592.par",
			"wei593.par",
			"wei594.par",
			"wei595.par",
			"wei596.par",
			"wei597.par",
			"wei598.par",
			"wei599.par",
			"wei600.par",
			"wei601.par",
			"wei602.par",
			"wei603.par",
			"wei604.par",
			"wei605.par",
			"wei606.par",
			"wei607.par",
			"wei608.par",
			"wei609.par",
			"wei610.par",
			"wei611.par",
			"wei612.par",
			"wei613.par",
			"wei614.par",
			"wei615.par",
			"wei616.par",
			"wei617.par",
			"wei618.par",
			"wei619.par",
			"wei620.par",
			"wei621.par",
			"wei622.par",
			"wei623.par",
			"wei624.par",
			"wei625.par",
			"wei626.par",
			"wei627.par",
			"wei628.par",
			"wei629.par",
			"wei630.par",
			"wei631.par",
			"wei632.par",
			"wei633.par",
			"wei634.par",
			"wei635.par",
			"wei636.par",
			"wei637.par",
			"wei638.par",
			"wei639.par",
			"wei640.par",
			"wei641.par",
			"wei642.par",
			"wei643.par",
			"wei644.par",
			"wei645.par",
			"wei646.par",
			"wei647.par",
			"wei648.par",
			"wei649.par",
			"wei650.par",
			"wei651.par",
			"wei652.par",
			"wei653.par",
			"wei654.par",
			"wei655.par",
			"wei656.par",
			"wei657.par",
			"wei658.par",
			"wei659.par",
			"wei660.par",
			"wei661.par",
			"wei662.par",
			"wei663.par",
			"wei664.par",
			"wei665.par",
			"wei666.par",
			"wei667.par",
			"wei668.par",
			"wei669.par",
			"wei670.par",
			"wei671.par",
			"wei672.par",
			"wei673.par",
			"wei674.par",
			"wei675.par",
			"wei676.par",
			"wei677.par",
			"wei678.par",
			"wei679.par",
			"wei680.par",
			"wei681.par",
			"wei682.par",
			"wei683.par",
			"wei684.par",
			"wei685.par",
			"wei686.par",
			"wei687.par",
			"wei688.par",
			"wei689.par",
			"wei690.par",
			"wei691.par",
			"wei692.par",
			"wei693.par",
			"wei694.par",
			"wei695.par",
			"wei696.par",
			"wei697.par",
			"wei698.par",
			"wei699.par",
			"wei700.par",
			"wei701.par",
			"wei702.par",
			"wei703.par",
			"wei704.par",
			"wei705.par",
			"wei706.par",
			"wei707.par",
			"wei708.par",
			"wei709.par",
			"wei710.par",
			"wei711.par",
			"wei712.par",
			"wei713.par",
			"wei714.par",
			"wei715.par",
			"wei716.par",
			"wei717.par",
			"wei718.par",
			"wei719.par",
			"wei720.par",
			"wei721.par",
			"wei722.par",
			"wei723.par",
			"wei724.par",
			"wei725.par",
			"wei726.par",
			"wei727.par",
			"wei728.par",
			"wei729.par",
			"wei730.par",
			"wei731.par",
			"wei732.par",
			"wei733.par",
			"wei734.par",
			"wei735.par",
			"wei736.par",
			"wei737.par",
			"wei738.par",
			"wei739.par",
			"wei740.par",
			"wei741.par",
			"wei742.par",
			"wei743.par",
			"wei744.par",
			"wei745.par",
			"wei746.par",
			"wei747.par",
			"wei748.par",
			"wei749.par",
			"wei750.par",
			"wei751.par",
			"wei752.par",
			"wei753.par",
			"wei754.par",
			"wei755.par",
			"wei756.par",
			"wei757.par",
			"wei758.par",
			"wei759.par",
			"wei760.par",
			"wei761.par",
			"wei762.par",
			"wei763.par",
			"wei764.par",
			"wei765.par",
			"wei766.par",
			"wei767.par",
			"wei768.par",
			"wei769.par",
			"wei770.par",
			"wei771.par",
			"wei772.par",
			"wei773.par",
			"wei774.par",
			"wei775.par",
			"wei776.par",
			"wei777.par",
			"wei778.par",
			"wei779.par",
			"wei780.par",
			"wei781.par",
			"wei782.par",
			"wei783.par",
			"wei784.par",
			"wei785.par",
			"wei786.par",
			"wei787.par",
			"wei788.par",
			"wei789.par",
			"wei790.par",
			"wei791.par",
			"wei792.par",
			"wei793.par",
			"wei794.par",
			"wei795.par",
			"wei796.par",
			"wei797.par",
			"wei798.par",
			"wei799.par",
			"wei800.par",
			"wei801.par",
			"wei802.par",
			"wei803.par",
			"wei804.par",
			"wei805.par",
			"wei806.par",
			"wei807.par",
			"wei808.par",
			"wei809.par",
			"wei810.par",
			"wei811.par",
			"wei812.par",
			"wei813.par",
			"wei814.par",
			"wei815.par",
			"wei816.par",
			"wei817.par",
			"wei818.par",
			"wei819.par",
			"wei820.par",
			"wei821.par",
			"wei822.par",
			"wei823.par",
			"wei824.par",
			"wei825.par",
			"wei826.par",
			"wei827.par",
			"wei828.par",
			"wei829.par",
			"wei830.par",
			"wei831.par",
			"wei832.par",
			"wei833.par",
			"wei834.par",
			"wei835.par",
			"wei836.par",
			"wei837.par",
			"wei838.par",
			"wei839.par",
			"wei840.par",
			"wei841.par",
			"wei842.par",
			"wei843.par",
			"wei844.par",
			"wei845.par",
			"wei846.par",
			"wei847.par",
			"wei848.par",
			"wei849.par",
			"wei850.par",
			"wei851.par",
			"wei852.par",
			"wei853.par",
			"wei854.par",
			"wei855.par",
			"wei856.par",
			"wei857.par",
			"wei858.par",
			"wei859.par",
			"wei860.par",
			"wei861.par",
			"wei862.par",
			"wei863.par",
			"wei864.par",
			"wei865.par",
			"wei866.par",
			"wei867.par",
			"wei868.par",
			"wei869.par",
			"wei870.par",
			"wei871.par",
			"wei872.par",
			"wei873.par",
			"wei874.par",
			"wei875.par",
			"wei876.par",
			"wei877.par",
			"wei878.par",
			"wei879.par",
			"wei880.par",
			"wei881.par",
			"wei882.par",
			"wei883.par",
			"wei884.par",
			"wei885.par",
			"wei886.par",
			"wei887.par",
			"wei888.par",
			"wei889.par",
			"wei890.par",
			"wei891.par",
			"wei892.par",
			"wei893.par",
			"wei894.par",
			"wei895.par",
			"wei896.par",
			"wei897.par",
			"wei898.par",
			"wei899.par",
			"wei900.par",
			"wei901.par",
			"wei902.par",
			"wei903.par",
			"wei904.par",
			"wei905.par",
			"wei906.par",
			"wei907.par",
			"wei908.par",
			"wei909.par",
			"wei910.par",
			"wei911.par",
			"wei912.par",
			"wei913.par",
			"wei914.par",
			"wei915.par",
			"wei916.par",
			"wei917.par",
			"wei918.par",
			"wei919.par",
			"wei920.par",
			"wei921.par",
			"wei922.par",
			"wei923.par",
			"wei924.par",
			"wei925.par",
			"wei926.par",
			"wei927.par",
			"wei928.par",
			"wei929.par",
			"wei930.par",
			"wei931.par",
			"wei932.par",
			"wei933.par",
			"wei934.par",
			"wei935.par",
			"wei936.par",
			"wei937.par",
			"wei938.par",
			"wei939.par",
			"wei940.par",
			"wei941.par",
			"wei942.par",
			"wei943.par",
			"wei944.par",
			"wei945.par",
			"wei946.par",
			"wei947.par",
			"wei948.par",
			"wei949.par",
			"wei950.par",
			"wei951.par",
			"wei952.par",
			"wei953.par",
			"wei954.par",
			"wei955.par",
			"wei956.par",
			"wei957.par",
			"wei958.par",
			"wei959.par",
			"wei960.par",
			"wei961.par",
			"wei962.par",
			"wei963.par",
			"wei964.par",
			"wei965.par",
			"wei966.par",
			"wei967.par",
			"wei968.par",
			"wei969.par",
			"wei970.par",
			"wei971.par",
			"wei972.par",
			"wei973.par",
			"wei974.par",
			"wei975.par",
			"wei976.par",
			"wei977.par",
			"wei978.par",
			"wei979.par",
			"wei980.par",
			"wei981.par",
			"wei982.par",
			"wei983.par",
			"wei984.par",
			"wei985.par",
			"wei986.par",
			"wei987.par",
			"wei988.par",
			"wei989.par",
			"wei990.par",
			"wei991.par",
			"wei992.par",
			"wei993.par",
			"wei994.par",
			"wei995.par",
			"wei996.par",
			"wei997.par",
			"wei998.par",
			"wei999.par",
			"wei1000.par",
			"wei1001.par",
			"wei1002.par",
			"wei1003.par",
			"wei1004.par",
			"wei1005.par",
			"wei1006.par",
			"wei1007.par",
			"wei1008.par",
			"wei1009.par",
			"wei1010.par",
			"wei1011.par",
			"wei1012.par",
			"wei1013.par",
			"wei1014.par",
			"wei1015.par",
			"wei1016.par",
			"wei1017.par",
			"wei1018.par",
			"wei1019.par",
			"wei1020.par",
			"wei1021.par",
			"wei1022.par",
			"wei1023.par",
			"wei1024.par",
			"wei1025.par",
			"wei1026.par",
			"wei1027.par",
			"wei1028.par",
			"wei1029.par",
			"wei1030.par",
			"wei1031.par",
			"wei1032.par",
			"wei1033.par",
			"wei1034.par",
			"wei1035.par",
			"wei1036.par",
			"wei1037.par",
			"wei1038.par",
			"wei1039.par",
			"wei1040.par",
			"wei1041.par",
			"wei1042.par",
			"wei1043.par",
			"wei1044.par",
			"wei1045.par",
			"wei1046.par",
			"wei1047.par",
			"wei1048.par",
			"wei1049.par",
			"wei1050.par",
			"wei1051.par",
			"wei1052.par",
			"wei1053.par",
			"wei1054.par",
			"wei1055.par",
			"wei1056.par",
			"wei1057.par",
			"wei1058.par",
			"wei1059.par",
			"wei1060.par",
			"wei1061.par",
			"wei1062.par",
			"wei1063.par",
			"wei1064.par",
			"wei1065.par",
			"wei1066.par",
			"wei1067.par",
			"wei1068.par",
			"wei1069.par",
			"wei1070.par",
			"wei1071.par",
			"wei1072.par",
			"wei1073.par",
			"wei1074.par",
			"wei1075.par",
			"wei1076.par",
			"wei1077.par",
			"wei1078.par",
			"wei1079.par",
			"wei1080.par",
			"wei1081.par",
			"wei1082.par",
			"wei1083.par",
			"wei1084.par",
			"wei1085.par",
			"wei1086.par",
			"wei1087.par",
			"wei1088.par",
			"wei1089.par",
			"wei1090.par",
			"wei1091.par",
			"wei1092.par",
			"wei1093.par",
			"wei1094.par",
			"wei1095.par",
			"wei1096.par",
			"wei1097.par",
			"wei1098.par",
			"wei1099.par",
			"wei1100.par",
			"wei1101.par",
			"wei1102.par",
			"wei1103.par",
			"wei1104.par",
			"wei1105.par",
			"wei1106.par",
			"wei1107.par",
			"wei1108.par",
			"wei1109.par",
			"wei1110.par",
			"wei1111.par",
			"wei1112.par",
			"wei1113.par",
			"wei1114.par",
			"wei1115.par",
			"wei1116.par",
			"wei1117.par",
			"wei1118.par",
			"wei1119.par",
			"wei1120.par",
			"wei1121.par",
			"wei1122.par",
			"wei1123.par",
			"wei1124.par",
			"wei1125.par",
			"wei1126.par",
			"wei1127.par",
			"wei1128.par",
			"wei1129.par",
			"wei1130.par",
			"wei1131.par",
			"wei1132.par",
			"wei1133.par",
			"wei1134.par",
			"wei1135.par",
			"wei1136.par",
			"wei1137.par",
			"wei1138.par",
			"wei1139.par",
			"wei1140.par",
			"wei1141.par",
			"wei1142.par",
			"wei1143.par",
			"wei1144.par",
			"wei1145.par",
			"wei1146.par",
			"wei1147.par",
			"wei1148.par",
			"wei1149.par",
			"wei1150.par",
			"wei1151.par",
			"wei1152.par",
			"wei1153.par",
			"wei1154.par",
			"wei1155.par",
			"wei1156.par",
			"wei1157.par",
			"wei1158.par",
			"wei1159.par",
			"wei1160.par",
			"wei1161.par",
			"wei1162.par",
			"wei1163.par",
			"wei1164.par",
			"wei1165.par",
			"wei1166.par",
			"wei1167.par",
			"wei1168.par",
			"wei1169.par",
			"wei1170.par",
			"wei1171.par",
			"wei1172.par",
			"wei1173.par",
			"wei1174.par",
			"wei1175.par",
			"wei1176.par",
			"wei1177.par",
			"wei1178.par",
			"wei1179.par",
			"wei1180.par",
			"wei1181.par",
			"wei1182.par",
			"wei1183.par",
			"wei1184.par",
			"wei1185.par",
			"wei1186.par",
			"wei1187.par",
			"wei1188.par",
			"wei1189.par",
			"wei1190.par",
			"wei1191.par",
			"wei1192.par",
			"wei1193.par",
			"wei1194.par",
			"wei1195.par",
			"wei1196.par",
			"wei1197.par",
			"wei1198.par",
			"wei1199.par",
			"wei1200.par",
			"wei1201.par",
			"wei1202.par",
			"wei1203.par",
			"wei1204.par",
			"wei1205.par",
			"wei1206.par",
			"wei1207.par",
			"wei1208.par",
			"wei1209.par",
			"wei1210.par",
			"wei1211.par",
			"wei1212.par",
			"wei1213.par",
			"wei1214.par",
			"wei1215.par",
			"wei1216.par",
			"wei1217.par",
			"wei1218.par",
			"wei1219.par",
			"wei1220.par",
			"wei1221.par",
			"wei1222.par",
			"wei1223.par",
			"wei1224.par",
			"wei1225.par",
			"wei1226.par",
			"wei1227.par",
			"wei1228.par",
			"wei1229.par",
			"wei1230.par",
			"wei1231.par",
			"wei1232.par",
			"wei1233.par",
			"wei1234.par",
			"wei1235.par",
			"wei1236.par",
			"wei1237.par",
			"wei1238.par",
			"wei1239.par",
			"wei1240.par",
			"wei1241.par",
			"wei1242.par",
			"wei1243.par",
			"wei1244.par",
			"wei1245.par",
			"wei1246.par",
			"wei1247.par",
			"wei1248.par",
			"wei1249.par",
			"wei1250.par",
			"wei1251.par",
			"wei1252.par",
			"wei1253.par",
			"wei1254.par",
			"wei1255.par",
			"wei1256.par",
			"wei1257.par",
			"wei1258.par",
			"wei1259.par",
			"wei1260.par",
			"wei1261.par",
			"wei1262.par",
			"wei1263.par",
			"wei1264.par",
			"wei1265.par",
			"wei1266.par",
			"wei1267.par",
			"wei1268.par",
			"wei1269.par",
			"wei1270.par",
			"wei1271.par",
			"wei1272.par",
			"wei1273.par",
			"wei1274.par",
			"wei1275.par",
			"wei1276.par",
			"wei1277.par",
			"wei1278.par",
			"wei1279.par",
			"wei1280.par",
			"wei1281.par",
			"wei1282.par",
			"wei1283.par",
			"wei1284.par",
			"wei1285.par",
			"wei1286.par",
			"wei1287.par",
			"wei1288.par",
			"wei1289.par",
			"wei1290.par",
			"wei1291.par",
			"wei1292.par",
			"wei1293.par",
			"wei1294.par",
			"wei1295.par",
			"wei1296.par",
			"wei1297.par",
			"wei1298.par",
			"wei1299.par",
			"wei1300.par",
			"wei1301.par",
			"wei1302.par",
			"wei1303.par",
			"wei1304.par",
			"wei1305.par",
			"wei1306.par",
			"wei1307.par",
			"wei1308.par",
			"wei1309.par",
			"wei1310.par",
			"wei1311.par",
			"wei1312.par",
			"wei1313.par",
			"wei1314.par",
			"wei1315.par",
			"wei1316.par",
			"wei1317.par",
			"wei1318.par",
			"wei1319.par",
			"wei1320.par",
			"wei1321.par",
			"wei1322.par",
			"wei1323.par",
			"wei1324.par",
			"wei1325.par",
			"wei1326.par",
			"wei1327.par",
			"wei1328.par",
			"wei1329.par",
			"wei1330.par",
			"wei1331.par",
			"wei1332.par",
			"wei1333.par",
			"wei1334.par",
			"wei1335.par",
			"wei1336.par",
			"wei1337.par",
			"wei1338.par",
			"wei1339.par",
			"wei1340.par",
			"wei1341.par",
			"wei1342.par",
			"wei1343.par",
			"wei1344.par",
			"wei1345.par",
			"wei1346.par",
			"wei1347.par",
			"wei1348.par",
			"wei1349.par",
			"wei1350.par",
			"wei1351.par",
			"wei1352.par",
			"wei1353.par",
			"wei1354.par",
			"wei1355.par",
			"wei1356.par",
			"wei1357.par",
			"wei1358.par",
			"wei1359.par",
			"wei1360.par",
			"wei1361.par",
			"wei1362.par",
			"wei1363.par",
			"wei1364.par",
			"wei1365.par",
			"wei1366.par",
			"wei1367.par",
			"wei1368.par",
			"wei1369.par",
			"wei1370.par",
			"wei1371.par",
			"wei1372.par",
			"wei1373.par",
			"wei1374.par",
			"wei1375.par",
			"wei1376.par",
			"wei1377.par",
			"wei1378.par",
			"wei1379.par",
			"wei1380.par",
			"wei1381.par",
			"wei1382.par",
			"wei1383.par",
			"wei1384.par",
			"wei1385.par",
			"wei1386.par",
			"wei1387.par",
			"wei1388.par",
			"wei1389.par",
			"wei1390.par",
			"wei1391.par",
			"wei1392.par",
			"wei1393.par",
			"wei1394.par",
			"wei1395.par",
			"wei1396.par",
			"wei1397.par",
			"wei1398.par",
			"wei1399.par",
			"wei1400.par",
			"wei1401.par",
			"wei1402.par",
			"wei1403.par",
			"wei1404.par",
			"wei1405.par",
			"wei1406.par",
			"wei1407.par",
			"wei1408.par",
			"wei1409.par",
			"wei1410.par",
			"wei1411.par",
			"wei1412.par",
			"wei1413.par",
			"wei1414.par",
			"wei1415.par",
			"wei1416.par",
			"wei1417.par",
			"wei1418.par",
			"wei1419.par",
			"wei1420.par",
			"wei1421.par",
			"wei1422.par",
			"wei1423.par",
			"wei1424.par",
			"wei1425.par",
			"wei1426.par",
			"wei1427.par",
			"wei1428.par",
			"wei1429.par",
			"wei1430.par",
			"wei1431.par",
			"wei1432.par",
			"wei1433.par",
			"wei1434.par",
			"wei1435.par",
			"wei1436.par",
			"wei1437.par",
			"wei1438.par",
			"wei1439.par",
			"wei1440.par",
			"wei1441.par",
			"wei1442.par",
			"wei1443.par",
			"wei1444.par",
			"wei1445.par",
			"wei1446.par",
			"wei1447.par",
			"wei1448.par",
			"wei1449.par",
			"wei1450.par",
			"wei1451.par",
			"wei1452.par",
			"wei1453.par",
			"wei1454.par",
			"wei1455.par",
			"wei1456.par",
			"wei1457.par",
			"wei1458.par",
			"wei1459.par",
			"wei1460.par",
			"wei1461.par",
			"wei1462.par",
			"wei1463.par",
			"wei1464.par",
			"wei1465.par",
			"wei1466.par",
			"wei1467.par",
			"wei1468.par",
			"wei1469.par",
			"wei1470.par",
			"wei1471.par",
			"wei1472.par",
			"wei1473.par",
			"wei1474.par",
			"wei1475.par",
			"wei1476.par",
			"wei1477.par",
			"wei1478.par",
			"wei1479.par",
			"wei1480.par",
			"wei1481.par",
			"wei1482.par",
			"wei1483.par",
			"wei1484.par",
			"wei1485.par",
			"wei1486.par",
			"wei1487.par",
			"wei1488.par",
			"wei1489.par",
			"wei1490.par",
			"wei1491.par",
			"wei1492.par",
			"wei1493.par",
			"wei1494.par",
			"wei1495.par",
			"wei1496.par",
			"wei1497.par",
			"wei1498.par",
			"wei1499.par",
			"wei1500.par",
			"wei1501.par",
			"wei1502.par",
			"wei1503.par",
			"wei1504.par",
			"wei1505.par",
			"wei1506.par",
			"wei1507.par",
			"wei1508.par",
			"wei1509.par",
			"wei1510.par",
			"wei1511.par",
			"wei1512.par",
			"wei1513.par",
			"wei1514.par",
			"wei1515.par",
			"wei1516.par",
			"wei1517.par",
			"wei1518.par",
			"wei1519.par",
			"wei1520.par",
			"wei1521.par",
			"wei1522.par",
			"wei1523.par",
			"wei1524.par",
			"wei1525.par",
			"wei1526.par",
			"wei1527.par",
			"wei1528.par",
			"wei1529.par",
			"wei1530.par",
			"wei1531.par",
			"wei1532.par",
			"wei1533.par",
			"wei1534.par",
			"wei1535.par",
			"wei1536.par",
			"wei1537.par",
			"wei1538.par",
			"wei1539.par",
			"wei1540.par",
			"wei1541.par",
			"wei1542.par",
			"wei1543.par",
			"wei1544.par",
			"wei1545.par",
			"wei1546.par",
			"wei1547.par",
			"wei1548.par",
			"wei1549.par",
			"wei1550.par",
			"wei1551.par",
			"wei1552.par",
			"wei1553.par",
			"wei1554.par",
			"wei1555.par",
			"wei1556.par",
			"wei1557.par",
			"wei1558.par",
			"wei1559.par",
			"wei1560.par",
			"wei1561.par",
			"wei1562.par",
			"wei1563.par",
			"wei1564.par",
			"wei1565.par",
			"wei1566.par",
			"wei1567.par",
			"wei1568.par",
			"wei1569.par",
			"wei1570.par",
			"wei1571.par",
			"wei1572.par",
			"wei1573.par",
			"wei1574.par",
			"wei1575.par",
			"wei1576.par",
			"wei1577.par",
			"wei1578.par",
			"wei1579.par",
			"wei1580.par",
			"wei1581.par",
			"wei1582.par",
			"wei1583.par",
			"wei1584.par",
			"wei1585.par",
			"wei1586.par",
			"wei1587.par",
			"wei1588.par",
			"wei1589.par",
			"wei1590.par",
			"wei1591.par",
			"wei1592.par",
			"wei1593.par",
			"wei1594.par",
			"wei1595.par",
			"wei1596.par",
			"wei1597.par",
			"wei1598.par",
			"wei1599.par",
			"wei1600.par",
			"wei1601.par",
			"wei1602.par",
			"wei1603.par",
			"wei1604.par",
			"wei1605.par",
			"wei1606.par",
			"wei1607.par",
			"wei1608.par",
			"wei1609.par",
			"wei1610.par",
			"wei1611.par",
			"wei1612.par",
			"wei1613.par",
			"wei1614.par",
			"wei1615.par",
			"wei1616.par",
			"wei1617.par",
			"wei1618.par",
			"wei1619.par",
			"wei1620.par",
			"wei1621.par",
			"wei1622.par",
			"wei1623.par",
			"wei1624.par",
			"wei1625.par",
			"wei1626.par",
			"wei1627.par",
			"wei1628.par",
			"wei1629.par",
			"wei1630.par",
			"wei1631.par",
			"wei1632.par",
			"wei1633.par",
			"wei1634.par",
			"wei1635.par",
			"wei1636.par",
			"wei1637.par",
			"wei1638.par",
			"wei1639.par",
			"wei1640.par",
			"wei1641.par",
			"wei1642.par",
			"wei1643.par",
			"wei1644.par",
			"wei1645.par",
			"wei1646.par",
			"wei1647.par",
			"wei1648.par",
			"wei1649.par",
			"wei1650.par",
			"wei1651.par",
			"wei1652.par",
			"wei1653.par",
			"wei1654.par",
			"wei1655.par",
			"wei1656.par",
			"wei1657.par",
			"wei1658.par",
			"wei1659.par",
			"wei1660.par",
			"wei1661.par",
			"wei1662.par",
			"wei1663.par",
			"wei1664.par",
			"wei1665.par",
			"wei1666.par",
			"wei1667.par",
			"wei1668.par",
			"wei1669.par",
			"wei1670.par",
			"wei1671.par",
			"wei1672.par",
			"wei1673.par",
			"wei1674.par",
			"wei1675.par",
			"wei1676.par",
			"wei1677.par",
			"wei1678.par",
			"wei1679.par",
			"wei1680.par",
			"wei1681.par",
			"wei1682.par",
			"wei1683.par",
			"wei1684.par",
			"wei1685.par",
			"wei1686.par",
			"wei1687.par",
			"wei1688.par",
			"wei1689.par",
			"wei1690.par",
			"wei1691.par",
			"wei1692.par",
			"wei1693.par",
			"wei1694.par",
			"wei1695.par",
			"wei1696.par",
			"wei1697.par",
			"wei1698.par",
			"wei1699.par",
			"wei1700.par",
			"wei1701.par",
			"wei1702.par",
			"wei1703.par",
			"wei1704.par",
			"wei1705.par",
			"wei1706.par",
			"wei1707.par",
			"wei1708.par",
			"wei1709.par",
			"wei1710.par",
			"wei1711.par",
			"wei1712.par",
			"wei1713.par",
			"wei1714.par",
			"wei1715.par",
			"wei1716.par",
			"wei1717.par",
			"wei1718.par",
			"wei1719.par",
			"wei1720.par",
			"wei1721.par",
			"wei1722.par",
			"wei1723.par",
			"wei1724.par",
			"wei1725.par",
			"wei1726.par",
			"wei1727.par",
			"wei1728.par",
			"wei1729.par",
			"wei1730.par",
			"wei1731.par",
			"wei1732.par",
			"wei1733.par",
			"wei1734.par",
			"wei1735.par",
			"wei1736.par",
			"wei1737.par",
			"wei1738.par",
			"wei1739.par",
			"wei1740.par",
			"wei1741.par",
			"wei1742.par",
			"wei1743.par",
			"wei1744.par",
			"wei1745.par",
			"wei1746.par",
			"wei1747.par",
			"wei1748.par",
			"wei1749.par",
			"wei1750.par",
			"wei1751.par",
			"wei1752.par",
			"wei1753.par",
			"wei1754.par",
			"wei1755.par",
			"wei1756.par",
			"wei1757.par",
			"wei1758.par",
			"wei1759.par",
			"wei1760.par",
			"wei1761.par",
			"wei1762.par",
			"wei1763.par",
			"wei1764.par",
			"wei1765.par",
			"wei1766.par",
			"wei1767.par",
			"wei1768.par",
			"wei1769.par",
			"wei1770.par",
			"wei1771.par",
			"wei1772.par",
			"wei1773.par",
			"wei1774.par",
			"wei1775.par",
			"wei1776.par",
			"wei1777.par",
			"wei1778.par",
			"wei1779.par",
			"wei1780.par",
			"wei1781.par",
			"wei1782.par",
			"wei1783.par",
			"wei1784.par",
			"wei1785.par",
			"wei1786.par",
			"wei1787.par",
			"wei1788.par",
			"wei1789.par",
			"wei1790.par",
			"wei1791.par",
			"wei1792.par",
			"wei1793.par",
			"wei1794.par",
			"wei1795.par",
			"wei1796.par",
			"wei1797.par",
			"wei1798.par",
			"wei1799.par",
			"wei1800.par",
			"wei1801.par",
			"wei1802.par",
			"wei1803.par",
			"wei1804.par",
			"wei1805.par",
			"wei1806.par",
			"wei1807.par",
			"wei1808.par",
			"wei1809.par",
			"wei1810.par",
			"wei1811.par",
			"wei1812.par",
			"wei1813.par",
			"wei1814.par",
			"wei1815.par",
			"wei1816.par",
			"wei1817.par",
			"wei1818.par",
			"wei1819.par",
			"wei1820.par",
			"wei1821.par",
			"wei1822.par",
			"wei1823.par",
			"wei1824.par",
			"wei1825.par",
			"wei1826.par",
			"wei1827.par",
			"wei1828.par",
			"wei1829.par",
			"wei1830.par",
			"wei1831.par",
			"wei1832.par",
			"wei1833.par",
			"wei1834.par",
			"wei1835.par",
			"wei1836.par",
			"wei1837.par",
			"wei1838.par",
			"wei1839.par",
			"wei1840.par",
			"wei1841.par",
			"wei1842.par",
			"wei1843.par",
			"wei1844.par",
			"wei1845.par",
			"wei1846.par",
			"wei1847.par",
			"wei1848.par",
			"wei1849.par",
			"wei1850.par",
			"wei1851.par",
			"wei1852.par",
			"wei1853.par",
			"wei1854.par",
			"wei1855.par",
			"wei1856.par",
			"wei1857.par",
			"wei1858.par",
			"wei1859.par",
			"wei1860.par",
			"wei1861.par",
			"wei1862.par",
			"wei1863.par",
			"wei1864.par",
			"wei1865.par",
			"wei1866.par",
			"wei1867.par",
			"wei1868.par",
			"wei1869.par",
			"wei1870.par",
			"wei1871.par",
			"wei1872.par",
			"wei1873.par",
			"wei1874.par",
			"wei1875.par",
			"wei1876.par",
			"wei1877.par",
			"wei1878.par",
			"wei1879.par",
			"wei1880.par",
			"wei1881.par",
			"wei1882.par",
			"wei1883.par",
			"wei1884.par",
			"wei1885.par",
			"wei1886.par",
			"wei1887.par",
			"wei1888.par",
			"wei1889.par",
			"wei1890.par",
			"wei1891.par",
			"wei1892.par",
			"wei1893.par",
			"wei1894.par",
			"wei1895.par",
			"wei1896.par",
			"wei1897.par",
			"wei1898.par",
			"wei1899.par",
			"wei1900.par",
			"wei1901.par",
			"wei1902.par",
			"wei1903.par",
			"wei1904.par",
			"wei1905.par",
			"wei1906.par",
			"wei1907.par",
			"wei1908.par",
			"wei1909.par",
			"wei1910.par",
			"wei1911.par",
			"wei1912.par",
			"wei1913.par",
			"wei1914.par",
			"wei1915.par",
			"wei1916.par",
			"wei1917.par",
			"wei1918.par",
			"wei1919.par",
			"wei1920.par",
			"wei1921.par",
			"wei1922.par",
			"wei1923.par",
			"wei1924.par",
			"wei1925.par",
			"wei1926.par",
			"wei1927.par",
			"wei1928.par",
			"wei1929.par",
			"wei1930.par",
			"wei1931.par",
			"wei1932.par",
			"wei1933.par",
			"wei1934.par",
			"wei1935.par",
			"wei1936.par",
			"wei1937.par",
			"wei1938.par",
			"wei1939.par",
			"wei1940.par",
			"wei1941.par",
			"wei1942.par",
			"wei1943.par",
			"wei1944.par",
			"wei1945.par",
			"wei1946.par",
			"wei1947.par",
			"wei1948.par",
			"wei1949.par",
			"wei1950.par",
			"wei1951.par",
			"wei1952.par",
			"wei1953.par",
			"wei1954.par",
			"wei1955.par",
			"wei1956.par",
			"wei1957.par",
			"wei1958.par",
			"wei1959.par",
			"wei1960.par",
			"wei1961.par",
			"wei1962.par",
			"wei1963.par",
			"wei1964.par",
			"wei1965.par",
			"wei1966.par",
			"wei1967.par",
			"wei1968.par",
			"wei1969.par",
			"wei1970.par",
			"wei1971.par",
			"wei1972.par",
			"wei1973.par",
			"wei1974.par",
			"wei1975.par",
			"wei1976.par",
			"wei1977.par",
			"wei1978.par",
			"wei1979.par",
			"wei1980.par",
			"wei1981.par",
			"wei1982.par",
			"wei1983.par",
			"wei1984.par",
			"wei1985.par",
			"wei1986.par",
			"wei1987.par",
			"wei1988.par",
			"wei1989.par",
			"wei1990.par",
			"wei1991.par",
			"wei1992.par",
			"wei1993.par",
			"wei1994.par",
			"wei1995.par",
			"wei1996.par",
			"wei1997.par",
			"wei1998.par",
			"wei1999.par"         };

long random();
void srandom();

int seprand(k)
     int k;
{
  long l;
  int f;
  l = random();
  f = l % k;
  return(f);
}

void reset_net()
{
  int i,j,v;

  for (i=0;i<ges_mod;i++)
    {
      Yk_mod_new[i]=0.5;
      Yk_mod_old[i]=0.5;
    }
  for (i=in_mod;i<ges_mod;i++)
    for (j=in_mod;j<ges_mod;j++)
      for (v=0;v<ges_mod;v++)
	{
	  Pk_ijm_mod[i][j][v]=0; 
	  Pk_ijm_mod_o[i][j][v]=0;
	}
}

void set_input_t()
{
  int i,j,k;
  double max;
  if (bias1==0)
    {
      for (i=0;i<in_mod;i++)
	{
	  Yk_mod_new[i]=inp_t[example_t][element_t][i];
	  Yk_mod_old[i]=Yk_mod_new[i];
	}
    }
  else
    {
      for (i=0;i<in_mod-1;i++)
	{
	  Yk_mod_new[i]=inp_t[example_t][element_t][i];
	  Yk_mod_old[i]=Yk_mod_new[i];
	}
      Yk_mod_new[in_mod-1]=1.0;
      Yk_mod_old[in_mod-1]=1.0;
    }

  max=0;
  for (k=hi_in_mod,j=0;k<ges_mod;k++,j++)
    {
      target_a[j]=tar_t[example_t][element_t][j];
      if (fabs(tar_t[example_t][element_t][j])>max)
	max=fabs(tar_t[example_t][element_t][j]);
    }
  targ=1;
  if (max>1.0)
    targ=0;
}


void execute_act_test()
{
  set_input_t();
  element_t++;
  if (element_t>length_t[example_t])
    {
      element_t=0;
      example_t++;
      seq_cor=1;
      seq_err=0;
      if (sequ_reset==1)
	reset_net();
      if (example_t>test_size-1)
	{
	  test_end=1;
	}
      set_input_t();
    }
}

void forward_pass()
{
  int i,j;
  double sum;
  
#pragma omp parallel for private(i,j,sum)     
  for (i=in_mod;i<ges_mod;i++)
    {
      sum = 0;
      for (j=0;j<ges_mod;j++)
	sum += W_mod[i][j]*Yk_mod_old[j];
      Yk_mod_new[i] = 1/(1+exp(-sum));
    };
}

void comp_err() 
{
  int k,j,maxout;
  double err,max;

  /* MSE */
  for (k=hi_in_mod,j=0;k<ges_mod;k++,j++)
    {
      err=  error[j]*error[j];
    };
  seq_err+=err;
  epoch_err+=err;

  /* Maximal error output */

  max=0;
  for (k=hi_in_mod,j=0;k<ges_mod;k++,j++)
    {
      if (fabs(error[j])>max)
	{
	  max=fabs(error[j]);
	  maxout=j;
	}
    };
  if ((seq_cor==1)&&(max>seq_max))
    {
      seq_cor=0;
      class_err++;
    }


  /* output error */
  if(test_flag)
    {
      fprintf (ferrortest, "%10d, %6.2lf",sweeps,sqrt(err/(double)out_mod));
      for (k=hi_in_mod,j=0;k<ges_mod;k++,j++)
	{
	  fprintf (ferrortest, " ,%6.2lf",error[j]);
	}
      fprintf (ferrortest, "\n");
    }
  else
    {
      if(sweeps % INTERVAL_OF_AVS == 5)
	{
	  fprintf (ferrortr, "%10d, %6.2lf",(sweeps-5),sqrt(err/(double)out_mod));
	  for (k=hi_in_mod,j=0;k<ges_mod;k++,j++)
	    {
	      fprintf (ferrortr, " ,%6.2lf",error[j]);
	    }
	  fprintf (ferrortr, "\n");
	}	      
    }
}

void test()
{
  int i,k,j;
  element_t=0;
  example_t=0;
  epoch_err=0;
  class_err=0;
  seq_cor=1;
  seq_err=0;
  test_end=0;

  while (test_end==0)
    {

      /* executing the environment
	 and setting the input
	 */
      execute_act_test();

      /* forward pass */

      forward_pass();


      if (targ==1) /* only if target for this input */
	{
	  /* compute error */

	  for (k=hi_in_mod,j=0;k<ges_mod;k++,j++)
	    {
	      error[j]=  target_a[j] - Yk_mod_new[k];
	    };

	  /* Training error */

	  comp_err();
	}

      /* set old activations */
      for (i=0;i<ges_mod;i++)
	{
	  Yk_mod_old[i] = Yk_mod_new[i];
        }

    }

  fp1=fopen(outfile, "a");
  fprintf(fp1,"TEST: epochs:%d sequences:%d\n",epoch+1,numb_seq);
  fprintf(fp1,"TEST: MSE:%.4f\n",epoch_err/(1.0*test_size));
  fprintf(fp1,"TEST: misclassifications:%d (out of %d test examples)\n",class_err,test_size);
  fprintf(fp1,"\n");
  fclose(fp1);
}

void output_epoch()
{
  fp1=fopen(outfile, "a");
  fprintf(fp1,"epochs:%d sequences:%d\n",epoch+1,numb_seq);
  fprintf(fp1,"MSE:%.4f\n", epoch_err/(1.0*training_size));
  fprintf(fp1,"misclassifications:%d (out of %d training examples)\n",class_err,training_size);
  fprintf(fp1,"\n");
  fclose(fp1);
}

void weight_out()
{
  int i,j,v;
  fp2 = fopen(weightfile, "w");
  /* fprintf(fp2,"anz:%d\n",numb_seq);*/
  for (i=in_mod;i<ges_mod;i++)
    {
      for (j=0;j<ges_mod;j++)
	/*	fprintf(fp2,"(%.2d,%.2d): %.3f ",i,j,W_mod[i][j]);*/
	fprintf(fp2,"(%.2d,%.2d): %f ",i,j,W_mod[i][j]);
      fprintf(fp2,"\n");
    };

  /* for (i=in_mod;i<ges_mod;i++)
    {
      for (j=0;j<ges_mod;j++)
	fprintf(fp2,"(%.2d,%.2d): %.3f ",i,j,DW[i][j]);
      fprintf(fp2,"\n");
      };*/

  for (i=0;i<ges_mod;i++)
    {
      fprintf(fp2,"i:%d %f ",i,Yk_mod_new[i]);
    }
  fprintf(fp2,"\n");

  for (i=in_mod;i<ges_mod;i++)
    {
      for (j=in_mod;j<ges_mod;j++)
	{      
	  for (v=0;v<ges_mod;v++)
	    {
	      fprintf(fp2,"i:%d j:%d v:%d %f ",i,j,v,Pk_ijm_mod[i][j][v]);
	    }
	}
    }
  fprintf(fp2,"\n");
  fclose(fp2);
}

void weight_out_init()
{
  int i,j,v;
  fp2 = fopen("init_weight","w");
  /* fprintf(fp2,"anz:%d\n",numb_seq);*/
  for (i=in_mod;i<ges_mod;i++)
    {
      for (j=0;j<ges_mod;j++)
	/*	fprintf(fp2,"(%.2d,%.2d): %.3f ",i,j,W_mod[i][j]);*/
	fprintf(fp2,"(%.2d,%.2d): %f ",i,j,W_mod[i][j]);
      fprintf(fp2,"\n");
    };

  /* for (i=in_mod;i<ges_mod;i++)
    {
      for (j=0;j<ges_mod;j++)
	fprintf(fp2,"(%.2d,%.2d): %.3f ",i,j,DW[i][j]);
      fprintf(fp2,"\n");
      };*/

  for (i=0;i<ges_mod;i++)
    {
      fprintf(fp2,"i:%d %f ",i,Yk_mod_new[i]);
    }
  fprintf(fp2,"\n");

  for (i=in_mod;i<ges_mod;i++)
    {
      for (j=in_mod;j<ges_mod;j++)
	{      
	  for (v=0;v<ges_mod;v++)
	    {
	      fprintf(fp2,"i:%d j:%d v:%d %f ",i,j,v,Pk_ijm_mod[i][j][v]);
	    }
	}
    }
  fprintf(fp2,"\n");
  fclose(fp2);
}

void weight_in()
{
  int i,j,v,r,s,t;
  fp3 = fopen("restart_file", "r");

  for (i=in_mod;i<ges_mod;i++)
    {
      for (j=0;j<ges_mod;j++)
	{
	  fscanf(fp3,"(%d,%d): %lf ", &r, &s, &W_mod[i][j]);
	}
      fscanf(fp3,"\n");
    };

  /*  for (i=in_mod;i<ges_mod;i++)
    {
      for (j=0;j<ges_mod;j++)
	fscanf(fp3,"(%d,%d): %lf ",&t,&v,DW[i][j]);
      fscanf(fp3,"\n");
      };*/

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

  fscanf(fp3,"\n");
  fclose(fp3);

}

void set_input()
{
  int i,j,k;
  double max;
  if (bias1==0)
    {
      for (i=0;i<in_mod;i++)
	{
	  Yk_mod_new[i]=inp[example][element][i];
	  Yk_mod_old[i]=Yk_mod_new[i];
	}
    }
  else
    {
      for (i=0;i<in_mod-1;i++)
	{
	  Yk_mod_new[i]=inp[example][element][i];
	  Yk_mod_old[i]=Yk_mod_new[i];
	}
      Yk_mod_new[in_mod-1]=1.0;
      Yk_mod_old[in_mod-1]=1.0;
    }

  max=0;
  for (k=hi_in_mod,j=0;k<ges_mod;k++,j++)
    {
      target_a[j]=tar[example][element][j];
      if (fabs(tar[example][element][j])>max)
	max=fabs(tar[example][element][j]);
    }
  /* is there a target for this input */
  targ=1;
  if (max>1.0)
    targ=0;
}

void execute_act()
{
/*   modified in 02/06/2010  */
  int i;
  length_field=(int)sqrt(vision_bias-1);
  half_length_field=(length_field-1)/2;
  int k,j;
  int i_start;
  int which_direction;

  center_field_x = (MAX_AREA-1)/2;
  center_field_y = (MAX_AREA-1)/2;
  min_field_x = center_field_x-half_length_field;
  min_field_y = center_field_y-half_length_field;
  max_field_x = center_field_x+1+half_length_field;
  max_field_y = center_field_y+1+half_length_field;

/* modified in 04/06/2010 */   
  if(test_flag)
    {
      if(sweeps%1000 == 0)
	{
	  fprintf (fchecka, "sweeps=%10d\n",sweeps); 
	}
    }
  else
    {
      if(sweeps%1000 == 0)
	{
	  fprintf (fcheckatr, "sweeps=%10d\n",sweeps); 
	}
    }
  
  if(change_sweeps == 0)
    {
      change_sweeps = INTERVAL_OF_SWEEPS;

      /* initialize positions of hands and others */
      /*  
	obj_x[0] is the X coordinate of the left hand in 2 dimensional space.
	obj_y[0] is the Y coordinate of the left hand in 2 dimensional space.
	obj_x[1] is the X coordinate of the right hand in 2 dimensional space.
	obj_y[1] is the Y coordinate of the right hand in 2 dimensional space.
	obj_x[>1] is the X coordinate of the other in 2 dimensional space.
	obj_y[>1] is the Y coordinate of the other in 2 dimensional space.
      */

      for(nn=0; nn<=NUM_OF_OBJ; nn++)
	{
	  if(nn==0) /* left hand */
	    {
	      if(test_flag) /* test phase */
		{
		  obj_x[nn] = min_field_x;
		  obj_y[nn] = random()%(length_field)+min_field_y;
		}
	      else /* training phase */
		{
		  obj_x[nn] = random()%((MAX_AREA-1)/2);
		  obj_y[nn] = random()%(MAX_AREA);
		}
	      obj_old_x[nn] = obj_x[nn];
	      obj_old_y[nn] = obj_y[nn];
	    }
	  else if(nn==1) /* right hand */
	    {
	      if(test_flag) /* test phase */
		{
		  obj_x[nn] = max_field_x-1;
		  obj_y[nn] = random()%(length_field)+min_field_y;
		}
	      else /* training phase */
		{
		  obj_x[nn] = (random()%((MAX_AREA-1)/2))+(MAX_AREA-1)/2+1;
		  obj_y[nn] = random()%(MAX_AREA);
		}
	      obj_old_x[nn] = obj_x[nn];
	      obj_old_y[nn] = obj_y[nn];
	    }
	  else /* others */
	    {
	      if(test_flag) /* test phase */
		{
		  obj_x[nn] = random()%(length_field)+min_field_x;
		  obj_y[nn] = random()%(length_field)+min_field_y;
		}
	      else /* training phase */
		{
		  obj_x[nn] = random()%(MAX_AREA);
		  obj_y[nn] = random()%(MAX_AREA);
		}
	      obj_old_x[nn] = obj_x[nn];
	      obj_old_y[nn] = obj_y[nn];
	    }
	}
      
      /* check whether hand and others are within MAX_AREA */
      for(nn=0; nn<=NUM_OF_OBJ; nn++)
	{
	  if(obj_x[nn] < 0) obj_x[nn]=0;
	  if(obj_x[nn] >= MAX_AREA) obj_x[nn]=MAX_AREA-1;
	  if(obj_y[nn] < 0) obj_y[nn]=0;
	  if(obj_y[nn] >= MAX_AREA) obj_y[nn]=MAX_AREA-1;
	}
      
    }
  
      /* initialize INPUTS */
      for(nn=0; nn<vision_bias; nn++)
      {
	/*   INPUTS[nn]  = INPUTS_init[nn];*/
	INPUTS[nn]  = 0.0;
      }

      /* initialize field_avs*/
      for(mm=0; mm<=MAX_AREA*MAX_AREA; mm++)
	{
	  field_avs[mm] = 0.3;
	}

      /* update hand position */
      if(change_sweeps != INTERVAL_OF_SWEEPS)
	{
	  /* initialize tmp_hand*  */
	  tmp_hand_x = 0;
	  tmp_hand_y = 0;
	  tmp_hand_1_x = 0;
	  tmp_hand_1_y = 0;

	  /* initialize corollary discharge */
	  for(nn=0; nn<max_outputs; nn++)
	    {
	      flag_step[nn]  = 0.0;
	    }

	  /* calculate motor command (=efference copy) tmp_hand_*. Currently, out_mod =8 and i_start=0 */
	  /* Yk_mod_old[hi_in_mod+0] is output that moves left hand to the right.
	     Yk_mod_old[hi_in_mod+2] is output that moves left hand to the left.
	     Yk_mod_old[hi_in_mod+1] is output that moves left hand upwards.
	     Yk_mod_old[hi_in_mod+3] is output that moves left hand downwards.
	     Yk_mod_old[hi_in_mod+4] is output that moves right hand to the right.
	     Yk_mod_old[hi_in_mod+6] is output that moves right hand to the left.
	     Yk_mod_old[hi_in_mod+5] is output that moves right hand upwards.
	     Yk_mod_old[hi_in_mod+7] is output that moves right hand downwards. */

	  for(i_start=0; i_start < out_mod/8; i_start++)
	    {
	      /* left hand */
	      tmp_hand_x = tmp_hand_x 
		+Yk_mod_old[hi_in_mod+0+i_start*8]
		-Yk_mod_old[hi_in_mod+2+i_start*8];
     
	      tmp_hand_y = tmp_hand_y
		+Yk_mod_old[hi_in_mod+1+i_start*8]          
		-Yk_mod_old[hi_in_mod+3+i_start*8];

	      /* right hand */
	      tmp_hand_1_x = tmp_hand_1_x 
		+Yk_mod_old[hi_in_mod+4+i_start*8]
		-Yk_mod_old[hi_in_mod+6+i_start*8];
	      
	      tmp_hand_1_y = tmp_hand_1_y
		+Yk_mod_old[hi_in_mod+5+i_start*8]          
		-Yk_mod_old[hi_in_mod+7+i_start*8];
	    }

	  /* 1. update hand positions from motor commands */
	  /* 2. simplified forward model 
	     estimate the direction of hand movement(corrolary discharge) from efference copy(tmp_hand_*)  
             flag_step[0] = 0.8; left hand moves 1 square to the right.
             flag_step[2] = 0.8; left hand moves 1 square to the left.
             flag_step[1] = 0.8; left hand moves up 1 square.
             flag_step[3] = 0.8; left hand moves down 1 square.
             flag_step[4] = 0.8; left hand moves 1 square to the right.
             flag_step[6] = 0.8; left hand moves 1 square to the left. 
             flag_step[5] = 0.8; left hand moves up 1 square.
             flag_step[7] = 0.8; left hand moves down 1 square.
	  */

	  if(0.8 <= tmp_hand_x) 
	    {	  
	      obj_x[0] = obj_old_x[0] + 1;
	      flag_step[0] = 0.8;
	    }
	  if((-0.8 < tmp_hand_x)&&(tmp_hand_x < 0.8)) 
	    {
	      obj_x[0] = obj_old_x[0];
	    }
	  if(tmp_hand_x <= -0.8) 
	    {
	      obj_x[0] = obj_old_x[0] - 1;
	      flag_step[2] = 0.8;
	    }
	  if(0.8 <= tmp_hand_y) 
	    {
	      obj_y[0] = obj_old_y[0] + 1;
	      flag_step[1] = 0.8;
	    }
	  if((-0.8 < tmp_hand_y)&&(tmp_hand_y < 0.8)) 
	    {
	      obj_y[0] = obj_old_y[0];
	    }
	  if(tmp_hand_y <= -0.8) 
	    {
	      obj_y[0] = obj_old_y[0] - 1;
	      flag_step[3] = 0.8;
	    }

	  if(0.8 <= tmp_hand_1_x) 
	    {
	      obj_x[1] = obj_old_x[1] + 1;
	      flag_step[4] = 0.8;
	    }
	  if((-0.8 < tmp_hand_1_x)&&(tmp_hand_1_x < 0.8)) 
	    {
	      obj_x[1] = obj_old_x[1];
	    }
	  if(tmp_hand_1_x <= -0.8) 
	    {
	      obj_x[1] = obj_old_x[1] - 1;
	      flag_step[6] = 0.8;
	    }
	  if(0.8 <= tmp_hand_1_y) 
	    {
	      obj_y[1] = obj_old_y[1] + 1;
	      flag_step[5] = 0.8;
	    }
	  if((-0.8 < tmp_hand_1_y)&&(tmp_hand_1_y < 0.8)) 
	    {
	      obj_y[1] = obj_old_y[1];
	    }
	  if(tmp_hand_1_y <= -0.8) 
	    {
	      obj_y[1] = obj_old_y[1] - 1;
	      flag_step[7] = 0.8;
	    }
 
      /* update other position 
	 move other 1 square in a random direction every 50 steps */
	  for(nn=2; nn<=NUM_OF_OBJ; nn++)
	    {
	      if(test_flag)
		{
		  which_dir = random()%2;
		  speed = random()%50;
		  if(which_dir == 0)
		    {
		      if(speed == 0)obj_x[nn]=obj_old_x[nn]-1;
		    }
		  if(which_dir == 1)
		    {
		      if(speed == 0)obj_x[nn]=obj_old_x[nn]+1;
		    }
		  which_dir = random()%2;
		  speed = random()%50;
		  if(which_dir == 0)
		    {
		      if(speed == 0)obj_y[nn]=obj_old_y[nn]-1;
		    }
		  if(which_dir == 1)
		    {
		      if(speed == 0)obj_y[nn]=obj_old_y[nn]+1;
		    }
		}
	      else
		{
		  which_dir = random()%2;
		  speed = random()%50;
		  if(which_dir == 0)
		    {
		      if(speed == 0)obj_x[nn]=obj_old_x[nn]-1;
		    }
		  if(which_dir == 1)
		    {
		      if(speed == 0)obj_x[nn]=obj_old_x[nn]+1;
		    }
		  which_dir = random()%2;
		  speed = random()%50;
		  if(which_dir == 0)
		    {
		      if(speed == 0)obj_y[nn]=obj_old_y[nn]-1;
		    }
		  if(which_dir == 1)
		    {
		      if(speed == 0)obj_y[nn]=obj_old_y[nn]+1;
		    }
		}
	    }
	}

      /* keep hand and others inside MAX_AREA */
      for(nn=0; nn<=NUM_OF_OBJ; nn++)
	{
	  if(obj_x[nn] < 0) obj_x[nn]=0;
	  if(obj_x[nn] >= MAX_AREA) obj_x[nn]=MAX_AREA-1;
	  if(obj_y[nn] < 0) obj_y[nn]=0;
	  if(obj_y[nn] >= MAX_AREA) obj_y[nn]=MAX_AREA-1;
	}
 
      /* keep others inside view field during the test phase*/
      if(test_flag)
	{
	  for(nn=2; nn<=NUM_OF_OBJ; nn++)
	    {
	      if(obj_x[nn] < min_field_x) obj_x[nn]=min_field_x;
	      if(obj_x[nn] > (max_field_x-1)) obj_x[nn]=max_field_x-1;
	      if(obj_y[nn] < min_field_y) obj_y[nn]=min_field_y;
	      if(obj_y[nn] > (max_field_y-1)) obj_y[nn]=max_field_y-1;
	    }
	}

      /* check whether hand and others are within view field */
      for(nn=0; nn<=NUM_OF_OBJ; nn++)
	{
	  if( ( (min_field_x <= obj_x[nn])
                && (obj_x[nn] <= (max_field_x-1)))
	      && ( (min_field_y <= obj_y[nn])
		   && (obj_y[nn] <= (max_field_y-1))))
	    {
	      check_obj[nn] = 1;
	    }
	  else

	    {
	      check_obj[nn] = 0;
	    }
	}
      
      /* find the nearest object to center of view field */
      distance = 0.0;
      nearest_obj = NUM_OF_OBJ+1;
      min_distance = (MAX_AREA+1)*2.0;
      for(nn=0; nn <= NUM_OF_OBJ; nn++)
	{
	  if(((check_obj[nn] == 1) && (check_obj_old[nn] == 1))
	     && (((obj_x[nn]-obj_old_x[nn]) != 0)
		 || ((obj_y[nn]-obj_old_y[nn]) != 0) 
                 || ((obj_x[nn]==center_field_x)&&
                     (obj_old_x[nn]==center_field_x)&&
                     (obj_y[nn]==center_field_y)&&
                     (obj_old_y[nn]==center_field_y)
		     )))
	    {
	      distance 
		= sqrt((obj_x[nn]-center_field_x)
		      *(obj_x[nn]-center_field_x)
		      +(obj_y[nn]-center_field_y)
		      *(obj_y[nn]-center_field_y));
	      if(distance < min_distance)
		{
		  nearest_obj = nn;
		  min_distance = distance;
		}
	    }
	}

      /* initialize target_a */
      for (k=hi_in_mod,j=0;k<ges_mod;k++,j++)
	{
	  target_a[j] = 0.0;
	};

      /* calculate proprioceptively perceived positions(obj_x/y_tmp[nn]) of both hands from hand positions(obj_x/y[nn]) */
      for(nn=0; nn<2; nn++)
	{
	  if(obj_x[nn]<10)
	    {
	      obj_x_tmp[nn]=0;
	    }
	  if(obj_x[nn]>=10 && obj_x[nn]<=20)
	    {
	      obj_x_tmp[nn]=10;
	    }
	  if(obj_x[nn]>20)
	    {
	      obj_x_tmp[nn]=20;
	    }

	  if(obj_y[nn]<10)
	    {
	      obj_y_tmp[nn]=0;
	    }
	  if(obj_y[nn]>=10 && obj_y[nn]<=20)
	    {
	      obj_y_tmp[nn]=10;
	    }
	  if(obj_y[nn]>20)
	    {
	      obj_y_tmp[nn]=20;
	    }
	}

      center_field_x_tmp = 10;
      center_field_y_tmp = 10;

      /* calculate motor command errors */
      /* left hand */
      target_a[0] =  (double)(center_field_x_tmp-obj_x_tmp[0])/center_field_x_tmp/1.0;
      target_a[2] = -(double)(center_field_x_tmp-obj_x_tmp[0])/center_field_x_tmp/1.0;
      target_a[1] =  (double)(center_field_y_tmp-obj_y_tmp[0])/center_field_y_tmp/1.0;
      target_a[3] = -(double)(center_field_y_tmp-obj_y_tmp[0])/center_field_y_tmp/1.0;

      /* right hand */
      target_a[4] =  (double)(center_field_x_tmp-obj_x_tmp[1])/center_field_x_tmp/1.0;
      target_a[6] = -(double)(center_field_x_tmp-obj_x_tmp[1])/center_field_x_tmp/1.0;
      target_a[5] =  (double)(center_field_y_tmp-obj_y_tmp[1])/center_field_y_tmp/1.0;
      target_a[7] = -(double)(center_field_y_tmp-obj_y_tmp[1])/center_field_y_tmp/1.0;

      number_of_inputs = 0;

      /* input data into INPUTS array */
      for(nn=NUM_OF_OBJ; nn>=0; nn--)
	{
	  if(check_obj[nn] == 1)
	    {
	      number_of_inputs = (obj_x[nn]-min_field_x+1)
		+(obj_y[nn]-min_field_y)*length_field;

	      if((((center_field_x-half_length_field/2)<=obj_x[nn])
		  && (obj_x[nn]<=(center_field_x+1+half_length_field/2)))
		 && (((center_field_y-half_length_field/2)<=obj_y[nn])
		     && (obj_y[nn]<=((center_field_x+1+half_length_field/2)))))
		{
		  if(nn == 0 || nn == 1)
		    {
		      INPUTS[number_of_inputs] = 0.5;
		    }
		  else
		    {
		      if(test_flag)
			{
			  INPUTS[number_of_inputs] = 0.5;
			}
		      else
			{
			  INPUTS[number_of_inputs] = nn*0.1;
			}
		    }
		}
	      else /*Change INPUTS[] values when different input values are used for the center 
                     and periphery of the field of view. Currently, the same values are input.  */
		{
		  if(nn == 0 || nn == 1)
		    {
		      INPUTS[number_of_inputs] = 0.5;
		    }
		  else
		    {
		      if(test_flag)
			{
			  INPUTS[number_of_inputs] = 0.5;
			}
		      else
			{
			  INPUTS[number_of_inputs] = nn*0.1;
			}
		    }
		}
	    }
	}

      if (bias1==0) /* Currently, it is not used */
	{
	  for (i=1;i<in_mod;i++)
	    {
	      Yk_mod_new[i-1]=INPUTS[i];
	      Yk_mod_old[i-1]=INPUTS[i];
	    }
	}
      else  /* bias is always used */
	{
	  /* for (i=1;i<in_mod-1;i++)*/
	  for (i=1;i<vision_bias-1;i++)
	    {
	      Yk_mod_new[i-1]=INPUTS[i];
	      Yk_mod_old[i-1]=INPUTS[i];
	    }

	  Yk_mod_new[vision_bias-1]=1.0;
	  Yk_mod_old[vision_bias-1]=1.0;

	  /* input collorary discharge */
	  for(nn=0; nn<max_outputs; nn++)
	    {
	      Yk_mod_new[vision_bias+nn]=flag_step[nn];
	      Yk_mod_old[vision_bias+nn]=flag_step[nn];
	    }
	  
	  /* input proprioceptive estimates of hands position(x,y) */
	  for(nn=0; nn<2; nn++)
	    {
	      Yk_mod_new[vision_bias+out_mod+nn*2]=(double)obj_x_tmp[nn]/MAX_AREA;
	      Yk_mod_old[vision_bias+out_mod+nn*2]=(double)obj_x_tmp[nn]/MAX_AREA;
	      Yk_mod_new[vision_bias+out_mod+nn*2+1]=(double)obj_y_tmp[nn]/MAX_AREA;
	      Yk_mod_old[vision_bias+out_mod+nn*2+1]=(double)obj_y_tmp[nn]/MAX_AREA;
	    }
	}

  /* setting the net input */
      /*  set_input();*/
  /* next element in the sequence */
      /* element++;*/
  /* sequence end */
      /*  if (element>length[example])
	  {*/
      /* first element in a sequence */
      /*  element=0;*/
      /* counting the number of sequences in a epoch */
      /*  example++;*/
      /* counting the number of sequences the net has seen */
      /*  numb_seq++;*/
      /* resetting the error for the current sequence */
      /* seq_cor=1;*/
      /* resetting the error per sequence */
      /*seq_err=0;*/
      /* reset after each sequence? */
      /* if (sequ_reset==1)
	 reset_net();*/
      /* weight update after each sequence? */
      /* if (w_up==1)
	 weight_up=1;*/
      /* end of an epoch ? */
      /* if (example>training_size-1)
	 {*/
	  /* MSE and misclassifications output for training set */
      /* output_epoch();*/
	  /* when to stop learning */
      /* if (stop_learn==0)
	    {
	      if ((epoch_err<stop_mse)||(class_err<stop_class))
		{
		  stop_learn=1;
		  maxepoch=epoch+ext_epochs;
		}
		}*/
	  /* weight update after each epoch? */
      /* if (w_up!=1)
	 weight_up=1;*/
	  /* performing a test on a test  set? */
      /*  if ((epoch%test_aus==0)||(class_err<min_fehl))
	    {
	      reset_net();
	      test();
	      reset_net();
	      seq_cor=1;
	      seq_err=0;
	      }*/
	  /* first sequence in the training set */
      /* example=0;*/
	  /* counting the epochs */
      /*  epoch++;*/
	  /* resetting the error per epoch */
      /*  epoch_err=0;*/
	  /* resetting the misclassifications per epoch */
      /*  class_err=0;*/
	  /* Write weight matrix */
      /*  if (epoch%w_out==0)
	  weight_out();

	  }
	  set_input();
	  }*/
}

void initia()
{
  int i, j;

  example=0;
  epoch=0;
  epoch_err=0;
  class_err=0;
  numb_seq=0;
  seq_cor=1;
  seq_err=0;
  weight_up=0;

/* modified in 04/06/2010 */   

  if(test_flag)
    {
      NUM_OF_OBJ = NUM_OF_OBJ_TEST;
    }
  else
    {
      NUM_OF_OBJ = NUM_OF_OBJ_TRAIN;
    }

  if(test_flag)
    {
      INTERVAL_OF_SWEEPS = INTERVAL_OF_SWEEPS_TEST;
    }
  else
    {
      INTERVAL_OF_SWEEPS = INTERVAL_OF_SWEEPS_TRAIN;
    }
 
  
  /* initialize check_obj, check_obj_old */
  for(nn=0; nn < vision_bias; nn++)
    {
      check_obj[nn] = 0;
      check_obj_old[nn] = 0;
    }
  
  /* weight initialization */
  if(restart_flag == 0)
    {
      for (i=in_mod;i<ges_mod;i++)
	{
	  for (j=0;j<ges_mod;j++)
	    {
	      W_mod[i][j] = (seprand(2000) - 1000);
	      W_mod[i][j] /= 1000.0;
	      W_mod[i][j]*=init_range;
	      DW[i][j]=0;
	    };
	};
      /* reset activations and derivatives */
      reset_net();
    }
  else
    {
      weight_in();
    }

  /* initial value of INPUTS */
  for(i=0; i<vision_bias; i++)
    {
      INPUTS_init[i]  = (double)(random()%11)/10.0;
    }
}






void backward_pass()
{
  int k,i,j,l;
  double sum,kron_y;

#pragma omp parallel for private(i,j,k,l,kron_y,sum)     
  for (i=in_mod;i<ges_mod;i++)
    for (j=0;j<ges_mod;j++)
      for (k=in_mod;k<ges_mod;k++)
	{
	  sum = 0;
	  for (l=in_mod;l<ges_mod;l++)
	    sum += W_mod[k][l]*Pk_ijm_mod_o[l][i][j];
	  if (i==k)
	    kron_y = Yk_mod_old[j];
	  else
	    kron_y = 0;
	  Pk_ijm_mod[k][i][j] = Yk_mod_new[k]*(1-Yk_mod_new[k])*(sum + kron_y);
	  
	};

#pragma omp parallel for private(i,j,k,l,sum)         
  for (i=in_mod;i<ges_mod;i++)
    {
      for (j=0;j<ges_mod;j++)
	{
	  sum = 0;
	  for (k=hi_in_mod,l=0;k<ges_mod;k++,l++)
	    sum += error[l]*Pk_ijm_mod[k][i][j];
	  DW[i][j] += alpha*sum;
	  
	};

    };
  
  for (i=in_mod;i<ges_mod;i++)
    for (j=0;j<ges_mod;j++)
      for (k=in_mod;k<ges_mod;k++)
	Pk_ijm_mod_o[k][i][j] = Pk_ijm_mod[k][i][j];
  
  
}

void weight_update()
{
  int i,j;


  for (i=in_mod;i<ges_mod;i++)
    {
      for (j=0;j<ges_mod;j++)
	{
	  W_mod[i][j] += DW[i][j];
	  DW[i][j] = 0;
	};
    }
 
  /* set W_mod[input][*]=0 to realize the block diagram of 
     learning hand regard (Figure 1(B), Homma,T, Forntiers in Psychology(2018)) */

  for (i=0; i<ges_mod; i++)
    {
       for (j=0; j<ges_mod; j++)
	{
	  if ((in_mod<=i && i<=hi_in_mod-1)&&(hi_in_mod<=j && j<=ges_mod-1))
	    {
	      W_mod[i][j] = 0.0;  /* hidden units <- output units */
	    }
	  if ((hi_in_mod<=i && i<=ges_mod-1)&&(hi_in_mod<=j && j<=ges_mod-1))
	    {
	      W_mod[i][j] = 0.0;  /* output units <- output units */
	    }
	}
       for (j=0; j<in_mod; j++)
	 {
	   if (  0<=i && i<=(in_mod-1))
	     {
	       W_mod[i][j] = 0.0;  /* input units <- input units */
	     }
	   if (hi_in_mod<=i && i<=(ges_mod-1))
	     {
	       W_mod[i][j] = 0.0;   /* output units <- input units */
	     }
	 }
       for (j=vision_bias; j<(vision_bias+out_mod); j++)
	 {
	   if (in_mod+hid_mod/2<=i && i<=(hi_in_mod-1))
	     {
	       W_mod[i][j] = 0.0;  /* ownership <- corollary discharge */
	     }
	 }
       for (j=in_mod; j<(in_mod+hid_mod/2); j++)
	 {
	   if (in_mod+hid_mod/2<=i && i<=(hi_in_mod-1))
	     {
	       W_mod[i][j] = 0.0;  /* ownership <- agency */
	     }
	 }
       for (j=(in_mod+hid_mod/2); j<hi_in_mod; j++)
	 {
	   if (in_mod<=i && i<=(in_mod+hid_mod/2-1))
	     {
	       W_mod[i][j] = 0.0;  /* agency <- ownership */
	     }
	 }
    }
}

void getpars()
{
  int u,v,corr[50];
  char *wrong1[] = {
    "number inputs: ?",
    "number outputs: ?",
    "number hidden units: ?",
    "biased: ?",
    "learning rate: ?",
    "max. error for correct sequence: ?",
    "half interval length for intialization: ?",
    "performing test after ? epochs: ?",
    "performing test after fewer than ? wrong classifications on training set: ?",
    "write weight after ? epochs: ?",
    "max. number of trials: ?",
    "stop learning once MSE per epoch <: ?",
    "stop learning once wrong classifications per epoch <: ?",
    "epochs to be learned after stop learning is set: ?",
    "initialization of random generator: ?",
    "reset the net after each sequence?: ?",
    "weight update after sequence or epoch?: ?",
    "max. number of epochs: ?",
    "size of training set: ?",
    "size of test set: ?", 
    "restart?(yes=1 no=0): ?",
    "test or training?(test=1 training=0): ?",
    "max. number of sweeps: ?" } ;



  FILE *fp5;

  for (u=0;u<50;u++)
    {
      corr[u]=1;
    }

  v=0;
  fp5=fopen("rtrlpars.txt","r");
  corr[v]=fscanf(fp5,"number inputs: %d\n",&in_mod);
  /* are the maximal ranges correct? */
  if (in_mod>vision_bias)
    {
      printf("Program terminated!\n");
      printf("You have to set the constant vision_bias at begin\n");
      printf("of the program file greater or equal %d and then\n",in_mod);
      printf("compile the program again.\n");
      exit(0);
    }
  v++;
  corr[v]=fscanf(fp5,"number outputs: %d\n",&out_mod);
  /* are the maximal ranges correct? */
  if (out_mod>max_outputs)
    {
      printf("Program terminated!\n");
      printf("You have to set the constant max_outputs at begin\n");
      printf("of the program file greater or equal %d and then\n",out_mod);
      printf("compile the program again.\n");
      exit(0);
    }
  v++;
  corr[v]=fscanf(fp5,"number hidden units: %d\n",&hid_mod);
  v++;
  corr[v]=fscanf(fp5,"biased: %d\n",&bias1);
  v++;
  corr[v]=fscanf(fp5,"learning rate: %lf\n",&alpha);
  v++;
  corr[v]=fscanf(fp5,"max. error for correct sequence: %lf\n",&seq_max);
  v++;
  corr[v]=fscanf(fp5,"half interval length for intialization: %lf\n",&init_range);
  v++;
  corr[v]=fscanf(fp5,"performing test after ? epochs: %d\n",&test_aus);
  v++;
  corr[v]=fscanf(fp5,"performing test after fewer than ? wrong classifications on training set: %d\n",&min_fehl);
  v++;
  corr[v]=fscanf(fp5,"write weight after ? epochs: %d\n",&w_out);
  v++;
  corr[v]=fscanf(fp5,"max. number of trials: %d\n",&maxtrials);
  v++;
  corr[v]=fscanf(fp5,"stop learning once MSE per epoch <: %lf\n",&stop_mse);
  v++;
  corr[v]=fscanf(fp5,"stop learning once wrong classifications per epoch <: %d\n",&stop_class);
  v++;
  corr[v]=fscanf(fp5,"epochs to be learned after stop learning is set: %d\n",&ext_epochs);
  v++;
  corr[v]=fscanf(fp5,"initialization of random generator: %d\n",&ran_sta);
  v++;
  corr[v]=fscanf(fp5,"reset the net after each sequence?: %d\n",&sequ_reset);
  v++;
  corr[v]=fscanf(fp5,"weight update after sequence or epoch?: %d\n",&w_up);
  v++;
  corr[v]=fscanf(fp5,"max. number of epochs: %d\n",&maxepoch);
  v++;
  corr[v]=fscanf(fp5,"size of training set: %d\n",&training_size);
  /* are the maximal ranges correct? */
  if (training_size>max_training_size)
    {
      printf("Program terminated!\n");
      printf("You have to set the constant max_training_size at begin\n");
      printf("of the program file greater or equal %d and then\n",training_size);
      printf("compile the program again.\n");
      exit(0);
    }
  v++;
  corr[v]=fscanf(fp5,"size of test set: %d\n",&test_size);
  /* are the maximal ranges correct? */
  if (test_size>max_test_size)
    {
      printf("Program terminated!\n");
      printf("You have to set the constant max_test_size at begin\n");
      printf("of the program file greater or equal %d and then\n",test_size);
      printf("compile the program again.\n");
      exit(0);
    }
  v++;
/* modified in 06/06/2010 */
  corr[v]=fscanf(fp5,"restart?(yes=1 no=0): %d\n",&restart_flag);
  v++;
  corr[v]=fscanf(fp5,"test or training?(test=1 training=0): %d\n",&test_flag);
  v++;
  corr[v]=fscanf(fp5,"max. number of sweeps: %d\n",&max_sweeps);
  v++;

  fclose(fp5);

  for (u=0;u<v;u++)
    {
      if (corr[u]==0)
	{
	  printf("Error in lstmpars.txt at line:\n");
	  printf("%s\n",wrong1[u]);
	  exit(0);
	}
    }
}

void write_info()
{

      printf("Perhaps an error occurred during reading the file\n");
      printf("lstmtest.txt ---> stopping the program.\n");
      printf("There should be %d input components and\n",in_mod);
      printf("%d target components per line. Makes %d real\n",out_mod,in_mod+out_mod);
      printf("values per line.\n");
      printf("An input sequence ends with a line\n");
      printf("where the first value (out of %d values in\n",in_mod+out_mod);
      printf("the line) is greater than 10.0\n");
      printf("The file ends with an extra line\n");
      printf("where the first value (out of %d values in\n",in_mod+out_mod);
      printf("the line) is greater than 10.0\n");
}


void getsets()
{
  FILE *fp3,*fp4;
  int end_seq,elm,end,trains,i;


  fp3=fopen("rtrltrain.txt","r");
  end=0;
  trains=0;
  while (end==0)
    {
      end_seq=0;
      elm=0;
      while (end_seq==0)
	{
	  for (i=0;i<in_mod;i++)
	    {
	      fscanf(fp3,"%lf ",&inp[trains][elm][i]);
	    }
	  for (i=0;i<out_mod;i++)
	    {
	      fscanf(fp3,"%lf ",&tar[trains][elm][i]);
	    }
	  fscanf(fp3,"\n");
	  /* sequence ends with first input component greater 10.0 */
	  if (fabs(inp[trains][elm][0])>10.0)
	    {
	      /* if 2 successive sequences have a first input component greater
		 than 10.0, then training set is finished */ 
	      if (elm==0)
		{
		  end=1;
		}
	      else
		{
		  elm--;
		}
	      end_seq=1;
	    }
	  elm++;
	  if (elm>max_sequence_length)
	    {
	      printf("Program terminated!\n");
	      printf("You have to set the constant max_sequence_length at begin\n");
	      printf("of the program file greater or equal maximal sequence\n");
	      printf("length (>= %d) and then compile the program again.\n",elm);
	      exit(0);
	    }
	}
      length[trains]=elm;
      trains++;
      if (end==1)
	trains--;
    }
  fclose(fp3);
  if (trains!=training_size)
    {
      printf("Training set size in parameter file: %d.\n",training_size);
      printf("Training set size detected: %d.\n",trains);
      write_info();
      exit(0);
    }

  fp4=fopen("rtrltest.txt","r");
  end=0;
  trains=0;
  while (end==0)
    {
      end_seq=0;
      elm=0;
      while (end_seq==0)
	{
	  for (i=0;i<in_mod;i++)
	    {
	      fscanf(fp4,"%lf ",&inp_t[trains][elm][i]);
	    }
	  for (i=0;i<out_mod;i++)
	    {
	      fscanf(fp4,"%lf ",&tar_t[trains][elm][i]);
	    }
	  fscanf(fp3,"\n");
	  if (fabs(inp_t[trains][elm][0])>10.0)
	    {
	      if (elm==0)
		{
		  end=1;
		}
	      else
		{
		  elm--;
		}
	      end_seq=1;
	    }
	  elm++;

	  if (elm>max_sequence_length)
	    {
	      printf("Program terminated!\n");
	      printf("You have to set the constant max_sequence_length at begin\n");
	      printf("of the program file greater or equal maximal sequence\n");
	      printf("length (>= %d) and then compile the program again.\n",elm);
	      exit(0);
	    }
	}
      length_t[trains]=elm;
      trains++;
      if (end==1)
	trains--;
    }
  fclose(fp4);
  if (trains!=test_size)
    {
      printf("Test set size in parameter file: %d.\n",test_size);
      printf("Test set size detected: %d.\n",trains);
      write_info();
      exit(0);
    }
}

/* modified in 04/06/2010 */
void output_result()
{
  /* output trajectory of hand movements(left hand(obj_x/y[0]) and right hand(obj_x/y[1]))
     in 2D coordinates with the origin located at the center of field of view */
  /* left hand*/  
  if(sweeps <= 400000000)
    {
      if(test_flag) 
	{      
	  fprintf (ftrajLF0,"%4d, %4d\n",(obj_x[0]-center_field_x),(obj_y[0]-center_field_y)); 
	}
      else
	{
	  fprintf (ftrajLF0tr,"%4d, %4d\n",(obj_x[0]-center_field_x),(obj_y[0]-center_field_y)); 
	}
    }

  if((400000001 <= sweeps) && (sweeps <= 800000000))
    {
      if(test_flag) 
	{      
	  fprintf (ftrajLF1,"%4d, %4d\n",(obj_x[0]-center_field_x),(obj_y[0]-center_field_y)); 
	}
      else
	{
	  fprintf (ftrajLF1tr,"%4d, %4d\n",(obj_x[0]-center_field_x),(obj_y[0]-center_field_y)); 
	}
    }

  if((800000001 <= sweeps) && (sweeps <= 1200000000))
    {
      if(test_flag) 
	{      
	  fprintf (ftrajLF2,"%4d, %4d\n",(obj_x[0]-center_field_x),(obj_y[0]-center_field_y)); 
	}
      else
	{
	  fprintf (ftrajLF2tr,"%4d, %4d\n",(obj_x[0]-center_field_x),(obj_y[0]-center_field_y)); 
	}
    } 

  /* right hand */
  if(sweeps <= 400000000)
    {
      if(test_flag) 
	{      
	  fprintf (ftrajRT0,"%4d, %4d\n",(obj_x[1]-center_field_x),(obj_y[1]-center_field_y)); 
	}
      else
	{
	  fprintf (ftrajRT0tr,"%4d, %4d\n",(obj_x[1]-center_field_x),(obj_y[1]-center_field_y)); 
	}
    }

  if((400000001 <= sweeps) && (sweeps <= 800000000))
    {
      if(test_flag) 
	{      
	  fprintf (ftrajRT1,"%4d, %4d\n",(obj_x[1]-center_field_x),(obj_y[1]-center_field_y)); 
	}
      else
	{
	  fprintf (ftrajRT1tr,"%4d, %4d\n",(obj_x[1]-center_field_x),(obj_y[1]-center_field_y)); 
	}
    }

  if((800000001 <= sweeps) && (sweeps <= 1200000000))
    {
      if(test_flag) 
	{      
	  fprintf (ftrajRT2,"%4d, %4d\n",(obj_x[1]-center_field_x),(obj_y[1]-center_field_y)); 
	}
      else
	{
	  fprintf (ftrajRT2tr,"%4d, %4d\n",(obj_x[1]-center_field_x),(obj_y[1]-center_field_y)); 
	}
    } 

  /* output "1" when left or right hand enters the area of 
     3 squares up, down, left, and right from the center of
     of the field of view. This output is used to count
     how many times left or right hand has entered this area. */

  if(sweeps <= 400000000)
    {
      if(test_flag) 
	{      
	  if((((center_field_x-3)<=obj_x[0])
	      && (obj_x[0]<=(center_field_x+3)))
	     && (((center_field_y-3)<=obj_y[0])
		 && (obj_y[0]<=((center_field_x+3))))
	     ||
	     (((center_field_x-3)<=obj_x[1])
	      && (obj_x[1]<=(center_field_x+3)))
	     && (((center_field_y-3)<=obj_y[1])
		 && (obj_y[1]<=((center_field_x+3))))
	     )
	    {
	      fprintf (ffindL0,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffindL0,"%4d\n",0); 
	    }
	}
      else
	{
	  if((((center_field_x-3)<=obj_x[0])
	      && (obj_x[0]<=(center_field_x+3)))
	     && (((center_field_y-3)<=obj_y[0])
		 && (obj_y[0]<=((center_field_x+3))))
	     ||
	     (((center_field_x-3)<=obj_x[1])
	      && (obj_x[1]<=(center_field_x+3)))
	     && (((center_field_y-3)<=obj_y[1])
		 && (obj_y[1]<=((center_field_x+3))))
	     )
	    {
	      fprintf (ffindL0tr,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffindL0tr,"%4d\n",0); 
	    }
	}
    }


  if((400000001 <= sweeps) && (sweeps <= 800000000))
    {
      if(test_flag) 
	{      
	  if((((center_field_x-3)<=obj_x[0])
	      && (obj_x[0]<=(center_field_x+3)))
	     && (((center_field_y-3)<=obj_y[0])
		 && (obj_y[0]<=((center_field_x+3))))
	     ||
	     (((center_field_x-3)<=obj_x[1])
	      && (obj_x[1]<=(center_field_x+3)))
	     && (((center_field_y-3)<=obj_y[1])
		 && (obj_y[1]<=((center_field_x+3))))
	     )
	    {
	      fprintf (ffindL1,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffindL1,"%4d\n",0); 
	    }
	}
      else
	{
	  if((((center_field_x-3)<=obj_x[0])
	      && (obj_x[0]<=(center_field_x+3)))
	     && (((center_field_y-3)<=obj_y[0])
		 && (obj_y[0]<=((center_field_x+3))))
	     ||
	     (((center_field_x-3)<=obj_x[1])
	      && (obj_x[1]<=(center_field_x+3)))
	     && (((center_field_y-3)<=obj_y[1])
		 && (obj_y[1]<=((center_field_x+3))))
	     )
	    {
	      fprintf (ffindL1tr,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffindL1tr,"%4d\n",0); 
	    }
	}
    }

  if((800000001 <= sweeps) && (sweeps <= 1200000000))
    {
      if(test_flag) 
	{      
	  if((((center_field_x-3)<=obj_x[0])
	      && (obj_x[0]<=(center_field_x+3)))
	     && (((center_field_y-3)<=obj_y[0])
		 && (obj_y[0]<=((center_field_x+3))))
	     ||
	     (((center_field_x-3)<=obj_x[1])
	      && (obj_x[1]<=(center_field_x+3)))
	     && (((center_field_y-3)<=obj_y[1])
		 && (obj_y[1]<=((center_field_x+3))))
	     )
	    {
	      fprintf (ffindL2,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffindL2,"%4d\n",0); 
	    }
	}
      else
	{
	  if((((center_field_x-3)<=obj_x[0])
	      && (obj_x[0]<=(center_field_x+3)))
	     && (((center_field_y-3)<=obj_y[0])
		 && (obj_y[0]<=((center_field_x+3))))
	     ||
	     (((center_field_x-3)<=obj_x[1])
	      && (obj_x[1]<=(center_field_x+3)))
	     && (((center_field_y-3)<=obj_y[1])
		 && (obj_y[1]<=((center_field_x+3))))
	     )
	    {
	      fprintf (ffindL2tr,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffindL2tr,"%4d\n",0); 
	    }
	}
    }

  /* output "1" when left or right hand enters the area of 
     2 squares up, down, left, and right from the center of
     of the field of view. This output is used to count
     how many times left or right hand has entered this area. */

  if(sweeps <= 400000000)
    {
      if(test_flag) 
	{      
	  if((((center_field_x-2)<=obj_x[0])
	      && (obj_x[0]<=(center_field_x+2)))
	     && (((center_field_y-2)<=obj_y[0])
		 && (obj_y[0]<=((center_field_x+2))))
	     ||
	     (((center_field_x-2)<=obj_x[1])
	      && (obj_x[1]<=(center_field_x+2)))
	     && (((center_field_y-2)<=obj_y[1])
		 && (obj_y[1]<=((center_field_x+2))))
	     )
	    {
	      fprintf (ffindLL0,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffindLL0,"%4d\n",0); 
	    }
	}
      else
	{
	  if((((center_field_x-2)<=obj_x[0])
	      && (obj_x[0]<=(center_field_x+2)))
	     && (((center_field_y-2)<=obj_y[0])
		 && (obj_y[0]<=((center_field_x+2))))
	     ||
	     (((center_field_x-2)<=obj_x[1])
	      && (obj_x[1]<=(center_field_x+2)))
	     && (((center_field_y-2)<=obj_y[1])
		 && (obj_y[1]<=((center_field_x+2))))
	     )
	    {
	      fprintf (ffindLL0tr,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffindLL0tr,"%4d\n",0); 
	    }
	}
    }


  if((400000001 <= sweeps) && (sweeps <= 800000000))
    {
      if(test_flag) 
	{      
	  if((((center_field_x-2)<=obj_x[0])
	      && (obj_x[0]<=(center_field_x+2)))
	     && (((center_field_y-2)<=obj_y[0])
		 && (obj_y[0]<=((center_field_x+2))))
	     ||
	     (((center_field_x-2)<=obj_x[1])
	      && (obj_x[1]<=(center_field_x+2)))
	     && (((center_field_y-2)<=obj_y[1])
		 && (obj_y[1]<=((center_field_x+2))))
	     )
	    {
	      fprintf (ffindLL1,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffindLL1,"%4d\n",0); 
	    }
	}
      else
	{
	  if((((center_field_x-2)<=obj_x[0])
	      && (obj_x[0]<=(center_field_x+2)))
	     && (((center_field_y-2)<=obj_y[0])
		 && (obj_y[0]<=((center_field_x+2))))
	     ||
	     (((center_field_x-2)<=obj_x[1])
	      && (obj_x[1]<=(center_field_x+2)))
	     && (((center_field_y-2)<=obj_y[1])
		 && (obj_y[1]<=((center_field_x+2))))
	     )
	    {
	      fprintf (ffindLL1tr,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffindLL1tr,"%4d\n",0); 
	    }
	}
    }

  if((800000001 <= sweeps) && (sweeps <= 1200000000))
    {
      if(test_flag) 
	{      
	  if((((center_field_x-2)<=obj_x[0])
	      && (obj_x[0]<=(center_field_x+2)))
	     && (((center_field_y-2)<=obj_y[0])
		 && (obj_y[0]<=((center_field_x+2))))
	     ||
	     (((center_field_x-2)<=obj_x[1])
	      && (obj_x[1]<=(center_field_x+2)))
	     && (((center_field_y-2)<=obj_y[1])
		 && (obj_y[1]<=((center_field_x+2))))
	     )
	    {
	      fprintf (ffindLL2,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffindLL2,"%4d\n",0); 
	    }
	}
      else
	{
	  if((((center_field_x-2)<=obj_x[0])
	      && (obj_x[0]<=(center_field_x+2)))
	     && (((center_field_y-2)<=obj_y[0])
		 && (obj_y[0]<=((center_field_x+2))))
	     ||
	     (((center_field_x-2)<=obj_x[1])
	      && (obj_x[1]<=(center_field_x+2)))
	     && (((center_field_y-2)<=obj_y[1])
		 && (obj_y[1]<=((center_field_x+2))))
	     )
	    {
	      fprintf (ffindLL2tr,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffindLL2tr,"%4d\n",0); 
	    }
	}
    }

  /* output "1" when left or right hand enters the area of 
     1 square up, down, left, and right from the center of
     of the field of view. This output is used to count
     how many times left or right hand has entered this area. */

  if(sweeps <= 400000000)
    {
      if(test_flag) 
	{      
	  if((abs(obj_x[0]-center_field_x)<=1) 
	     && (abs(obj_y[0]-center_field_y)<=1)
	     || 
	     (abs(obj_x[1]-center_field_x)<=1) 
	     && (abs(obj_y[1]-center_field_y)<=1)
	     )
	    {
	      fprintf (ffind0,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffind0,"%4d\n",0); 
	    }
	}
      else
	{
	  if((abs(obj_x[0]-center_field_x)<=1) 
	     && (abs(obj_y[0]-center_field_y)<=1)
	     || 
	     (abs(obj_x[1]-center_field_x)<=1) 
	     && (abs(obj_y[1]-center_field_y)<=1)
	     )
	    {
	      fprintf (ffind0tr,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffind0tr,"%4d\n",0); 
	    }
	}
    }

  if((400000001 <= sweeps) && (sweeps <= 800000000))
    {
      if(test_flag) 
	{      
	  if((abs(obj_x[0]-center_field_x)<=1) 
	     && (abs(obj_y[0]-center_field_y)<=1)
	     || 
	     (abs(obj_x[1]-center_field_x)<=1) 
	     && (abs(obj_y[1]-center_field_y)<=1)
	     )
	    {
	      fprintf (ffind1,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffind1,"%4d\n",0); 
	    }
	}
      else
	{
	  if((abs(obj_x[0]-center_field_x)<=1) 
	     && (abs(obj_y[0]-center_field_y)<=1)
	     || 
	     (abs(obj_x[1]-center_field_x)<=1) 
	     && (abs(obj_y[1]-center_field_y)<=1)
	     )
	    {
	      fprintf (ffind1tr,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffind1tr,"%4d\n",0); 
	    }
	}
    }

  if((800000001 <= sweeps) && (sweeps <= 1200000000))
    {
      if(test_flag) 
	{      
	  if((abs(obj_x[0]-center_field_x)<=1) 
	     && (abs(obj_y[0]-center_field_y)<=1)
	     || 
	     (abs(obj_x[1]-center_field_x)<=1) 
	     && (abs(obj_y[1]-center_field_y)<=1)
	     )
	    {
	      fprintf (ffind2,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffind2,"%4d\n",0); 
	    }
	}
      else
	{
	  if((abs(obj_x[0]-center_field_x)<=1) 
	     && (abs(obj_y[0]-center_field_y)<=1)
	     || 
	     (abs(obj_x[1]-center_field_x)<=1) 
	     && (abs(obj_y[1]-center_field_y)<=1)
	     )
	    {
	      fprintf (ffind2tr,"%4d\n",1); 
	    }
	  else
	    {
	      fprintf (ffind2tr,"%4d\n",0); 
	    }
	}
    }

  /* output view_field into eye.inp (for nakp) */
  d1 = div(sweeps,INTERVAL_OF_AVS);
  
  if(sweeps == 1)
    {
      if(test_flag)
	{
	  fprintf (fcheck5, "%8d\n",(max_sweeps-1)); 
	  fprintf (fcheck5, "data\n"); 
	  fprintf (fcheck5, "step1                   1\n"); 
	  fprintf (fcheck5, "%8d%8d\n",
		   (MAX_AREA+1)*(MAX_AREA+1),MAX_AREA*MAX_AREA);
	}
      else
	{
	  fprintf (fcheck5atr, "%8d\n",(max_sweeps/INTERVAL_OF_AVS)*AVS_RANGE); 
	  fprintf (fcheck5atr, "data\n"); 
	  fprintf (fcheck5atr, "step1                1\n"); 
	  fprintf (fcheck5atr, "%8d%8d\n",
		   (MAX_AREA+1)*(MAX_AREA+1),MAX_AREA*MAX_AREA);
	}
      /* nodal data */
      for(iii=1; iii<= (MAX_AREA+1)*(MAX_AREA+1); iii++)
	{
	  dnode = div(iii,(MAX_AREA+1));
	  if(dnode.rem == 0)
	    {
	      tmp_node_x = MAX_AREA;
	      tmp_node_y = dnode.quot-1;
	    }
	  else
	    {
	      tmp_node_x = dnode.rem-1;
	      tmp_node_y = dnode.quot;
	    }
	  
	  if(test_flag)
	    {
	      fprintf (fcheck5, "%6d%8.1lf%8.1lf%8.1lf\n"
		       ,iii,(float)tmp_node_x,(float)tmp_node_y,0.0); 
	    }	  
	  else 
	    {
	      fprintf (fcheck5atr, "%6d%8.1lf%8.1lf%8.1lf\n"
		       ,iii,(float)tmp_node_x,(float)tmp_node_y,0.0); 
	    }	  
	}
      /* element data */
      for(jjj=1; jjj <= MAX_AREA*MAX_AREA; jjj++)
	{
	  delem = div(jjj,MAX_AREA);
	  if(delem.rem != 0)
	    {
	      node_start = jjj + delem.quot;
	    }
	  else
	    {
	      node_start = jjj + (delem.quot-1);
	    }
	  elem_field[node_start] = jjj;
	  
	  if(test_flag)
	    {
	      fprintf (fcheck5, "%9d 1 quad %9d%9d%9d%9d\n"
		       ,jjj,node_start,node_start+1
		       ,node_start+(MAX_AREA+1)+1
		       ,node_start+(MAX_AREA+1));
	    }
	  else
	    {
	      fprintf (fcheck5atr, "%9d 1 quad %9d%9d%9d%9d\n"
		       ,jjj,node_start,node_start+1
		       ,node_start+(MAX_AREA+1)+1
		       ,node_start+(MAX_AREA+1));
	    }
	}
    }
  else /* sweeps != 1 */
    {
      if(test_flag)
	{
	  fprintf (fcheck5, "step%-6d              %6d\n",
		   sweeps,sweeps); 
	}
      else if((1<=d1.rem) && (d1.rem<=AVS_RANGE)) 
	{
	  fprintf (fcheck5atr, "step%-6d              %6d\n",(d1.quot*AVS_RANGE+d1.rem),sweeps); 
	}
    }
  
  /* view field data */
  for(iy=min_field_y; iy<= max_field_y-1; iy++)
    {
      for(ix=min_field_x; ix<= max_field_x-1; ix++)
	{
	  node_start = (MAX_AREA+1)*iy + ix +1;
	  tmp_elem = elem_field[node_start];

	  /*	  if((abs(ix-center_field_x)<=1) && (abs(iy-center_field_y)<=1))*/
	  if((abs(ix-center_field_x)<=2) && (abs(iy-center_field_y)<=2))
	    {
	      field_avs[tmp_elem] = 0.8;
	    }
	  else
	    {
	      field_avs[tmp_elem] = 1.0;
	    }
	}
    }
  /* objects data */
  for(nn=NUM_OF_OBJ; nn>=0; nn--)
    {
      node_start = (MAX_AREA+1)*obj_y[nn]+obj_x[nn]+1;
      tmp_elem = elem_field[node_start];

      if(nn == 0)
	{
	  field_avs[tmp_elem] = 0.7;
	}
      else if(nn == 1)
	{
	  field_avs[tmp_elem] = 0.6;
	}
      else
	{
	  field_avs[tmp_elem] = 0.0;
	}
    }
  /* input view field, objects data */
  if(test_flag)
    {
      fprintf (fcheck5, "0 1\n");
      fprintf (fcheck5, "1 1\n");
      fprintf (fcheck5, "field_avs[], \n");
      for(jjj=1; jjj<=MAX_AREA*MAX_AREA; jjj++)
	{
	  fprintf (fcheck5, "%9d%12.1lf\n"
		   ,jjj,field_avs[jjj]);
	}
    }
  else if((1<=d1.rem) && (d1.rem<=AVS_RANGE)) 
    {
      fprintf (fcheck5atr, "0 1\n");
      fprintf (fcheck5atr, "1 1\n");
      fprintf (fcheck5atr, "field_avs[], \n");
      for(jjj=1; jjj<=MAX_AREA*MAX_AREA; jjj++)
	{
	  fprintf (fcheck5atr, "%9d%12.1lf\n"
		   ,jjj,field_avs[jjj]);
	}
    }
  
  /* output RA[] into ra.inp (for nakp) */
  /* calculate number of nodes */
  delem = div((ges_mod-in_mod),UNIT_RANGE);
  if(delem.rem != 0)
    {
      node_max = (ges_mod-in_mod)+delem.quot+(UNIT_RANGE+1)+1;
    }
  else
    {
      node_max = (ges_mod-in_mod)+delem.quot-1+(UNIT_RANGE+1)+1;
    }
  
  if(sweeps == 1)
    {
      if(test_flag)
	{
	  /* fprintf (fcheck4, "%8d\n",(max_sweeps-1)/1+1); */
	  fprintf (fcheck4, "%8d\n",(max_sweeps-1)); 
	  fprintf (fcheck4, "data\n"); 
	  fprintf (fcheck4, "step1                   1\n"); 
	  fprintf (fcheck4, "%8d%8d\n",node_max,(ges_mod-in_mod));
	}
      else
	{
	  fprintf (fcheck4atr, "%8d\n",(max_sweeps/INTERVAL_OF_AVS)*AVS_RANGE); 
	  fprintf (fcheck4atr, "data\n"); 
	  fprintf (fcheck4atr, "step1                1\n"); 
	  fprintf (fcheck4atr, "%8d%8d\n",node_max,(ges_mod-in_mod));
	}
      /* nodal data */
      for(iii=1; iii<= node_max; iii++)
	{
	  dnode = div(iii,(UNIT_RANGE+1));
	  if(dnode.rem == 0)
	    {
	      tmp_node_x = UNIT_RANGE;
	      tmp_node_y = dnode.quot-1;
	    }
	  else
	    {
	      tmp_node_x = dnode.rem-1;
	      tmp_node_y = dnode.quot;
	    }
	  
	  if(test_flag)
	    {
	      fprintf (fcheck4, "%6d%8.1lf%8.1lf%8.1lf\n"
		       ,iii,(float)tmp_node_x,(float)tmp_node_y,0.0); 
	    }	  
	  else 
	    {
	      fprintf (fcheck4atr, "%6d%8.1lf%8.1lf%8.1lf\n"
		       ,iii,(float)tmp_node_x,(float)tmp_node_y,0.0); 
	    }	  
	}
      /* element data */
      for(jjj=1; jjj <= (ges_mod-in_mod); jjj++)
	{
	  delem = div(jjj,UNIT_RANGE);
	  if(delem.rem != 0)
	    {
	      node_start = jjj + delem.quot;
	    }
	  else
	    {
	      node_start = jjj + (delem.quot-1);
	    }
	      elem_unit[node_start] = jjj;
	      
	      if(test_flag)
		{
		  fprintf (fcheck4, "%9d 1 quad %9d%9d%9d%9d\n"
			   ,jjj,node_start,node_start+1
                           ,node_start+(UNIT_RANGE+1)+1
                           ,node_start+(UNIT_RANGE+1));
		}
	      else
		{
		  fprintf (fcheck4atr, "%9d 1 quad %9d%9d%9d%9d\n"
			   ,jjj,node_start,node_start+1
                           ,node_start+(UNIT_RANGE+1)+1
                           ,node_start+(UNIT_RANGE+1));
		}
	}
    }
  else /* sweeps != 1 */
    {
      if(test_flag)
	{
	  fprintf (fcheck4, "step%-6d              %6d\n",
		   sweeps,sweeps); 
	}
      else if((1<=d1.rem) && (d1.rem<=AVS_RANGE))
	{
	  fprintf (fcheck4atr, "step%-6d              %6d\n",d1.quot*AVS_RANGE+d1.rem,sweeps); 
	}
    }
  
  /* RA[] data */
  if(test_flag)
    {
      fprintf (fcheck4, "0 1\n");
      fprintf (fcheck4, "1 1\n");
      fprintf (fcheck4, "RA[], \n");
      for(iii=in_mod,jjj=1; iii<ges_mod;iii++, jjj++)
	{
	  fprintf (fcheck4, "%9d%12.1lf\n"
		   ,jjj,Yk_mod_old[iii]);
	}
    }
  else if((1<=d1.rem) && (d1.rem<=AVS_RANGE))
    {
      fprintf (fcheck4atr, "0 1\n");
      fprintf (fcheck4atr, "1 1\n");
      fprintf (fcheck4atr, "RA[], \n");
      for(iii=in_mod,jjj=1; iii<ges_mod;iii++, jjj++)
	{
	  fprintf (fcheck4atr, "%9d%12.1lf\n"
		   ,jjj,Yk_mod_old[iii]);
	}
    }
}

void main()
{

  int i, j, k,
    trialnr;



  /* input pars */

  getpars();


/* modified in 04/06/2010 */

  file_number = 0;

  if(test_flag)      
    {
      if((ffind0=fopen("find0","w"))==NULL){
	printf("find0 file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ffind0tr=fopen("find0tr","w"))==NULL){
	printf("find0tr file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)      
    {
      if((ffind1=fopen("find1","w"))==NULL){
	printf("find1 file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ffind1tr=fopen("find1tr","w"))==NULL){
	printf("find1tr file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)      
    {
      if((ffind2=fopen("find2","w"))==NULL){
	printf("find2 file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ffind2tr=fopen("find2tr","w"))==NULL){
	printf("find2tr file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)      
    {
      if((ffindL0=fopen("findL0","w"))==NULL){
	printf("findL0 file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ffindL0tr=fopen("findL0tr","w"))==NULL){
	printf("findL0tr file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)      
    {
      if((ffindL1=fopen("findL1","w"))==NULL){
	printf("findL1 file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ffindL1tr=fopen("findL1tr","w"))==NULL){
	printf("findL1tr file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)      
    {
      if((ffindL2=fopen("findL2","w"))==NULL){
	printf("findL2 file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ffindL2tr=fopen("findL2tr","w"))==NULL){
	printf("findL2tr file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)      
    {
      if((ffindLL0=fopen("findLL0","w"))==NULL){
	printf("findLL0 file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ffindLL0tr=fopen("findLL0tr","w"))==NULL){
	printf("findLL0tr file cannot open\n");
	exit(1);
      } 
    }


  if(test_flag)      
    {
      if((ffindLL1=fopen("findLL1","w"))==NULL){
	printf("findLL1 file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ffindLL1tr=fopen("findLL1tr","w"))==NULL){
	printf("findLL1tr file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)      
    {
      if((ffindLL2=fopen("findLL2","w"))==NULL){
	printf("findLL2 file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ffindLL2tr=fopen("findLL2tr","w"))==NULL){
	printf("findLL2tr file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)      
    {
      if((ftrajLF0=fopen("trajLF0.csv","w"))==NULL){
	printf("trajLF0.csv file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ftrajLF0tr=fopen("trajLF0tr.csv","w"))==NULL){
	printf("trajLF0tr.csv file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)      
    {
      if((ftrajRT0=fopen("trajRT0.csv","w"))==NULL){
	printf("trajRT0.csv file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ftrajRT0tr=fopen("trajRT0tr.csv","w"))==NULL){
	printf("trajRT0tr.csv file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)      
    {
      if((ftrajLF1=fopen("trajLF1.csv","w"))==NULL){
	printf("trajLF1.csv file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ftrajLF1tr=fopen("trajLF1tr.csv","w"))==NULL){
	printf("traj1tr.csv file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)      
    {
      if((ftrajRT1=fopen("trajRT1.csv","w"))==NULL){
	printf("trajRT1.csv file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ftrajRT1tr=fopen("trajRT1tr.csv","w"))==NULL){
	printf("trajRT1tr.csv file cannot open\n");
	exit(1);
      } 
    }
  if(test_flag)      
    {
      if((ftrajLF2=fopen("trajLF2.csv","w"))==NULL){
	printf("trajLF2.csv file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ftrajLF2tr=fopen("trajLF2tr.csv","w"))==NULL){
	printf("traj2tr.csv file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)      
    {
      if((ftrajRT2=fopen("trajRT2.csv","w"))==NULL){
	printf("trajRT2.csv file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ftrajRT2tr=fopen("trajRT2tr.csv","w"))==NULL){
	printf("trajRT2tr.csv file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)      
    {
      if((fcheck4=fopen("ra.inp","w"))==NULL){
	printf("ra.inp file cannot open\n");
	exit(1);
      } 
    }
  else
    {
      if((fcheck4atr=fopen("raatr.inp","w"))==NULL){
	printf("raatr.inp file cannot open\n");
	exit(1);
      }
      if((fcheck4btr=fopen("rabtr.inp","w"))==NULL){
	printf("rabtr.inp file cannot open\n");
	exit(1);
      }
      if((fcheck4ctr=fopen("ractr.inp","w"))==NULL){
	printf("ractr.inp file cannot open\n");
	exit(1);
      }
      if((fcheck4dtr=fopen("radtr.inp","w"))==NULL){
	printf("radtr.inp file cannot open\n");
	exit(1);
      } 
    }

  if(test_flag)
    {      
      if((fcheck5=fopen("eye.inp","w"))==NULL){
	printf("eye.inp file cannot open\n");
	exit(1);
      } 
    }
  else
    {
      if((fcheck5atr=fopen("eyeatr.inp","w"))==NULL){
	printf("eyeatr.inp file cannot open\n");
	exit(1);
      }
      if((fcheck5btr=fopen("eyebtr.inp","w"))==NULL){
	printf("eyebtr.inp file cannot open\n");
	exit(1);
      }
      if((fcheck5ctr=fopen("eyectr.inp","w"))==NULL){
	printf("eyectr.inp file cannot open\n");
	exit(1);
      }
      if((fcheck5dtr=fopen("eyedtr.inp","w"))==NULL){
	printf("eyedtr.inp file cannot open\n");
	exit(1);
      } 
    }
  
  if(test_flag)
    {
      if((fchecka=fopen("checka","w"))==NULL){
	printf("checka file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((fcheckatr=fopen("checkatr","w"))==NULL){
	printf("checkatr file cannot open\n");
	exit(1);
      }
    }

  if(test_flag)
    {
      if((fcheckb=fopen("checkb","w"))==NULL){
	printf("checkb file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((fcheckbtr=fopen("checkbtr","w"))==NULL){	      
	printf("checkbtr file cannot open\n");
	exit(1);
      }
    }
  
  if(test_flag)
    {
      if((fcheckc=fopen("checkc","w"))==NULL){
	printf("checkc file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((fcheckctr=fopen("checkctr","w"))==NULL){
	printf("checkctr file cannot open\n");
	exit(1);
      }
    }
  
  if(test_flag)
    {
      if((fcheckd=fopen("checkd","w"))==NULL){
	printf("checkd file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((fcheckdtr=fopen("checkdtr","w"))==NULL){
	printf("checkdtr file cannot open\n");
	exit(1);
      }
    }
   
  if(test_flag)
    {
      if((ferrortest=fopen("error.csv","w"))==NULL){
	printf("error.csv file cannot open\n");
	exit(1);
      }
    }
  else
    {
      if((ferrortr=fopen("errortr.csv","w"))==NULL){
	printf("errortr.csv file cannot open\n");
	exit(1);
      }
    }

  /* input training set and test set */

  /* getsets();   

  if (maxtrials>20)
  maxtrials=20; */

  if (bias1==1)
    in_mod++;

  in_mod = in_mod+out_mod;   /* add number of collorary discharge(=out_mod) */
  in_mod = in_mod+4;         
      /* add number of proprioceptive estimates of hands position(x,y) */

  hi_in_mod = in_mod+hid_mod;
  ges_mod = hi_in_mod+out_mod;

  if (ges_mod>max_units)
    {
      printf("Program terminated!\n");
      printf("You have to set the constant max_units at begin\n");
      printf("of the program file greater or equal %d and then\n",ges_mod);
      printf("compile the program again.\n");
      exit(0);
    }


  /*  srandom(time(NULL));*/
  srandom(ran_sta);

  /*  for (trialnr=0;trialnr<maxtrials;trialnr++)
    {


      outfile = outf[trialnr];

      weightfile = weig[trialnr];



      fp1 = fopen(outfile, "w");
      fprintf(fp1,"Trial Nr.:%.1d\n",trialnr);
      fclose(fp1);

      fp2 = fopen(weightfile, "w");
      fprintf(fp2,"Trial Nr.:%.1d\n",trialnr);
      fclose(fp2);
  */

  initia();

  examples=0;
  epoch=0;
  
  
  
  stop_learn=0;
  learn = 1;
  
  /*  while (learn == 1)
      {*/

  max_sweeps++;
  for(sweeps = 1; sweeps < max_sweeps; sweeps++)
    {

      /* executing the environment
	 and setting the input
      */
      execute_act();

      /* forward pass */
      forward_pass();
      
      
      /*  if (targ==1)*/ /* only if target for this input */
      /*{*/
	  /* compute error */
	  
       for (k=hi_in_mod,j=0;k<ges_mod;k++,j++)
	    {
	      /* error[j]=  target_a[j] - Yk_mod_new[k];*/
	      error[j]=  target_a[j];
	    };
       
	  /* Training error */
	  
       comp_err();
       /*  } */
      
      
      
      /* backward pass */
       if(test_flag == 0)
	 {
	   if(sweeps%interval_of_backward_pass == 0)
	     {
	       backward_pass();
	     }
	 }
      
      /* set old activations */
      for (i=0;i<ges_mod;i++)
	{
	  Yk_mod_old[i] = Yk_mod_new[i];
	}
      
      
      /* update weights */
      
       if(test_flag == 0)
	 {
	  weight_update();
	 }

       /* if (weight_up==1)
	{
	  weight_up=0;
	  weight_update();
	  }*/
   
      
      /* stop if maxepoch reached */
       /* if (epoch>maxepoch)
	  learn=0;*/
      
      /* modified in 04/06/2010 */
	  /* output result */
      output_result();
      
      /*	}*/

      /* test();*/
      /* }*/
      

      /* output restart file at sweeps = 1 */
      if((!test_flag) && (sweeps==1))
	{
	  weight_out_init();
	}

      /* output restart file every INTERVAL_OF_RESTART */
      if((!test_flag) && (sweeps%INTERVAL_OF_RESTART==0))
	{
	  weightfile = weig[file_number];
	  weight_out();
	  file_number++;
	}
      
      fflush(ffind0); 
      fflush(ffind0tr); 
      fflush(ffind1); 
      fflush(ffind1tr); 
      fflush(ffindL0); 
      fflush(ffindL0tr); 
      fflush(ffindL1); 
      fflush(ffindL1tr); 
      fflush(ffindL2); 
      fflush(ffindL2tr); 
      fflush(ffindLL0); 
      fflush(ffindLL0tr); 
      fflush(ffindLL1); 
      fflush(ffindLL1tr); 
      fflush(ffindLL2); 
      fflush(ffindLL2tr); 
      fflush(ftrajLF0);
      fflush(ftrajLF0tr);
      fflush(ftrajLF1);
      fflush(ftrajLF1tr);
      fflush(ftrajLF2);
      fflush(ftrajLF2tr);
      fflush(ftrajRT0);
      fflush(ftrajRT0tr);
      fflush(ftrajRT1);
      fflush(ftrajRT1tr);
      fflush(ftrajRT2);
      fflush(ftrajRT2tr);
      fflush(fcheck4);
      fflush(fcheck4atr); 
      fflush(fcheck4btr); 
      fflush(fcheck4ctr); 
      fflush(fcheck4dtr); 
      fflush(fcheck4); 
      fflush(fcheck5atr); 
      fflush(fcheck5btr); 
      fflush(fcheck5ctr); 
      fflush(fcheck5dtr); 
      fflush(fcheck5); 
      fflush(fchecka); 
      fflush(fcheckatr); 
      fflush(fcheckb); 
      fflush(fcheckbtr); 
      fflush(fcheckc); 
      fflush(fcheckctr); 
      fflush(fcheckd); 
      fflush(fcheckdtr); 
      fflush(ferrortest); 
      fflush(ferrortr); 
	     
      /* reduce change_sweeps */
      change_sweeps--;

      /* update obj_x,obj_y,check_obj */
      for(nn = 0; nn <= NUM_OF_OBJ; nn++)
	{ 
	  obj_old_x[nn] = obj_x[nn];
	  obj_old_y[nn] = obj_y[nn];
	  check_obj_old[nn] = check_obj[nn];
	}
    }
  
  exit(0);
}






