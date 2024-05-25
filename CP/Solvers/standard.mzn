% Use this editor as a MiniZinc scratch book
% Use this editor as a MiniZinc scratch book
% Use this editor as a MiniZinc scratch book
include "globals.mzn";
int:m;
int:n;

int: dim=(n div m)+2;

set of int:dimension=1..dim;

set of int :C = 1..m;   % Couriers
set of int :I = 1..n;   % Items
set of int :I_ext = 1..(n+1); % D matrix positions
array[C] of int: L;      % Couriers max weight
array[I] of int: S;      % Items weight
array[I_ext,I_ext] of int: D;    % Distance Distributions

array[C] of var 0..sum(S):weights=[sum(j in dimension where ass[j,courier]!=0 ) (S[ass[j,courier]])|courier in C];
array[dimension,C] of var 0..n:ass;
array[C] of var int : Start=[ass[1,courier]| courier in C];
array[C] of var int: Finish=[ ass[(count(i in dimension)(ass[i,courier]!=0)),courier]|courier in C];
array[C] of var int: num_assigned=[(count(i in dimension)(ass[i,courier]!=0)) | courier in C];

constraint forall(courier in C) 
           (   
            ( forall(j,z in dimension where ass[j,courier]==0 /\ z>j) ( ass[z,courier]==0))
           );

array[C] of var int: distance=[ sum( i in 1..num_assigned[courier]-1) (D[ass[i,courier],ass[i+1,courier]])+D[n+1,Start[courier]] + 
              D[Finish[courier],n+1] |courier in C]; 
 predicate bin_packing(int: c,array[int] of var int: bin,array[int] of var int: w) =
   forall( b in lb_array(bin)..ub_array(bin) ) (
     c >= sum ( i in index_set(bin) ) (
       w[i] 
     )   );

constraint forall(courier in C)(
             bin_packing(L[courier],ass[dimension,courier],[if ass[i,courier]!=0 then S[ass[i,courier]] else 0 endif|i in dimension]));
constraint forall(courier in C) (ass[1,courier]!=0);
% constraint forall(courier in C) 
%            (   
%             ( forall(j,z in dimension where ass[j,courier]==0 /\ z>j) ( ass[z,courier]==0))
%            );
           
constraint forall(courier in C, j in 1..dim-1) (
    ass[j, courier] == 0 -> forall(z in j+1..dim) (ass[z, courier] == 0)
);




constraint among(n,ass,I)::domain;

constraint alldifferent_except_0(ass)::domain;



int:low_max = max([D[n+1,i] +D[i,n+1] | i in 1..n]);
int: high = max(D)*((n div m)+1) + max(D[n+1,I_ext])+max(D[I_ext,n+1]);
var low_max..high: max_cap_dist; 
constraint max_cap_dist = max(distance);  

constraint forall(c,d in C where L[c]>=weights[d] /\ L[d]>=weights[c] /\ c>d) ( ass[1,c] > ass[1,d]);


solve :: seq_search([
          
    
               int_search(ass[1..dim div 2,i],dom_w_deg,indomain_random)|i in C
               
        
            ] )

     :: relax_and_reconstruct(array1d(ass),80) 
     :: restart_luby(n)
      %satisfy;
     minimize max_cap_dist ;


