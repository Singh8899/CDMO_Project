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
 
array[I] of var I_ext: init_obj;
array[I] of var C: obj_assign;
array[I] of var I_ext: next_obj;

% max weight and  
constraint forall(i in C)(
            sum(j in I where obj_assign[j]=i)( S[j] ) <= L[i]
          /\count_eq(init_obj, i, 1));
                   
constraint forall(i,j in I where i > j)(
            obj_assign[i] == obj_assign[j] -> next_obj[i] != next_obj[j] );                                            

constraint forall(i in I)(
            (next_obj[i] != 8 -> 
            (obj_assign[next_obj[i]] == obj_assign[i])/\ next_obj[i]!= i )
          /\(init_obj[i]!= 8 -> init_obj[i]==obj_assign[i])
          /\(obj_assign[i] == init_obj[i] \/ member( next_obj,i)));

constraint forall(i,j in I where i > j)(
              obj_assign[i]==obj_assign[j] ->
              XOR(next_obj[i]==j,next_obj[j]==i));
             
constraint count_eq(next_obj, 8, m);
constraint count_eq(init_obj, 8, n-m);

predicate XOR(var bool: A,var bool: B )=
            not A \/ not B;
            
function var int: dist_sum(array[I] of var int: init, array[I] of var int: assign, 
                            array[I] of var int: next,array[I_ext, I_ext] of int: D) =
    let 
    {                                                
        var int : distance = sum(i in I where init[i]!=8 )(D[8,i])+
                             sum(i in I)(D[i,next_obj[i]]);
    }in distance;            
            
constraint dist_sum(init_obj,obj_assign,next_obj,D) == total_dist;     

solve minimize total_dist;      
              
                         
                              
                                   
                                        
                                             
                                                  
                                                       
                                                            
                                                                 
                                                                           
            