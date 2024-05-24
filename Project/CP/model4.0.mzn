include "globals.mzn";
int:m;
int:n;

set of int :C = 1..m;   % Couriers
set of int :I = 1..n;   % Items
set of int :I_ext = 1..(n+1); % D matrix positions

array[C] of int: L;      % Couriers max weight
array[I] of int: S;      % Items weight
array[I_ext,I_ext] of int: D;    % Distance Distributions

int: total = sum(i in I_ext, j in I_ext)(D[i,j]);
int: total_max = total * m;
var 0..total_max : total_dist;

array[1..(n+m-1)] of var I_ext: obj_assign;
array[1..(n+m-1)] of var C union {(n+1)}: cour_assign;

constraint forall(i in C)(
            sum(j in 1..(n+m-1) where cour_assign[j]=i)( S[obj_assign[j]] ) <= L[i]);

constraint count_eq(obj_assign,(n+1) , m-1);
constraint count_eq(cour_assign,(n+1) , m-1);

constraint cour_assign[1] == 1 
         /\cour_assign[n+m-1] == m ;

constraint obj_assign[n+m-1] != (n+1)
         /\obj_assign[1] != (n+1) ;

constraint alldifferent_except(obj_assign,(n+1)..(n+1));

constraint forall(i in 1..(n+m-1))(
              (obj_assign[i] == (n+1) <-> cour_assign[i] == (n+1))
            /\nvalue(m+1, cour_assign));

constraint forall(i in 1..(n+m-2))(
              cour_assign[i] != (n+1) -> 
              (cour_assign[i] == cour_assign[i+1] \/ cour_assign[i+1] == (n+1)));


function var int: dist_sum(array[int] of var int: ass,array[I_ext, I_ext] of int: D) =
    let 
    {                                                
        var int : distance = D[(n+1),ass[1]] + D[ass[n+m-1],(n+1)] +
                             sum(i in 1..(n+m-2))(D[ass[i],ass[i+1]])
    }in distance;            

constraint dist_sum(obj_assign,D) == total_dist;     

solve minimize total_dist;