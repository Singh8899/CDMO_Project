include "globals.mzn";
int:m;
int:n;

int:dim = (n div m)+1+2;
set of int:dimension=1..dim;
set of int :C = 1..m;   % Couriers
set of int :I = 1..n;   % Items
set of int :I_ext = 1..(n+1); % D matrix positions

array[C] of int: L;      % Couriers max weight
array[I] of int: S;      % Items weight
array[I_ext,I_ext] of int: D;    % Distance Distributions
int: total_max = sum(j in I_ext, i in 1..2)(D[i,j]);
var 0..total_max : total_dist;

array[I] of int: low_bounds = [D[n+1,i] +D[i,n+1] | i in 1..n] ;

int:low = min(low_bounds);

int: high = sum(i in 1..(n-1-m))(D[i,i+1]) + D[n-m+1,1] + D[n-m,n+1];
array[C] of var low..high: distance;

var int:UB;
var int:LB ;
constraint UB = high;
constraint LB = low;

array[dimension,C] of var I_ext:ass; 

constraint among(n,ass,I);
constraint alldifferent_except(ass,(n+1)..(n+1));

constraint forall(i in C)(ass[1,i]==ass[dim,i]);

constraint forall(i in C)( 
              sum(j in 2..(dim-1) where ass[j,i]!=(n+1))(S[ass[j,i]]) <= L[i]);
              
constraint forall(i in C)( 
              forall(j in 2..(dim-1))(
                ass[j,i]==(n+1) -> ass[j+1,i]==(n+1)));
                
function var int: dist_sum(array[dimension,C] of var I_ext:ass,array[I_ext, I_ext] of int: D,int :courier) =
        let 
        {       
              var int: distance= sum(i in 1..(dim-1) )(D[ass[i,courier],ass[i+1,courier]]);
        } in distance;                

constraint forall(i in C) ( dist_sum(ass,D,i) == distance[i]); 

constraint total_dist==sum(i in C)(distance[i]);   

constraint symmetry_breaking_constraint(forall(i,j in C where (i>j /\ L[i]==L[j]) ) (      
                                           (ass[2,i] >= ass[2,j])));
%                                         /\(distance[i] >= distance[j])));   

                                               
% var int :max_distance= max(distance);
% var int :min_distance= min(distance);
% var float: weight_max_distance=0.5;
% var float: weight_total_distance=0.8;
% var int:max_min_dist=max_distance-min_distance;
% var float:objective_function=weight_max_distance*max_min_dist+weight_total_distance*total_dist;


%array[C] of var 0..sum(S):weights;

           
%constraint forall(courier in C) (sum(j in dimension where ass[j,courier]!=0 ) %(S[ass[j,courier]])==weights[courier]    
%           /\ weights[courier]<=L[courier]); 
                     
%constraint symmetry_breaking_constraint (forall(courier1,courier2 in C where( L[courier1] >=L[courier2] /\ courier1<courier2/\              
%            weights[courier2]<=weights[courier1] ))( 
%           ( lex_greater(ass[dimension,courier1],ass[dimension,courier2]) )
                   
%           ));

%solve minimize total_dist;
%constraint (total_dist div m + 20) > max(distance);
solve minimize total_dist ;
%solve minimize objective_function;

%output [ show_int(0,low)++"\n"++show_int(0,high)];
%solve minimize total_dist;