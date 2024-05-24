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

array[I_ext,I_ext] of var 0..m: routes;

array[C] of var 0..total: distances; 

array[I] of var C: obj_assign;


constraint forall(i in I, j in I_ext)( 
              (routes[i,j] == obj_assign[i] \/ routes[i,j]==0)
            /\(routes[j,i] == obj_assign[i] \/ routes[j,i]==0) );
%(routes[j,i] != routes[i,j] \/ routes[i,j]==0 \/ i==j  )
                    

constraint forall(i,j in I where i != j)(            
                routes[j,i] != routes[i,j] \/ routes[i,j]==0);                                
                                                                                                   
constraint forall(i in I)(              
              k_occurrences_of_x(routes[I_ext,i],obj_assign[i],2)
            /\k_occurrences_of_x(routes[i,I_ext],obj_assign[i],2)
            /\obj_assign[i]==routes[i,i]);
                         
constraint forall(i in C)(              
              k_occurrences_of_x(routes[I,n+1],i,1)
            /\k_occurrences_of_x(routes[n+1,I],i,1)
            /\sum(j in I where obj_assign[j]=i)( S[j] ) <= L[i]
            /\dist_sum(i,routes,D) == distances[i]);
                            
constraint routes[n+1,n+1]==0;             
                                                  
predicate k_occurrences_of_x(array[int] of var int: arr, var int: X, int: K) =
          count_eq(arr, X, K);
        
        
function var int: dist_sum(int: courier, array[I_ext,I_ext] of var int: assign, 
                                                  array[I_ext, I_ext] of int: D) =
    let 
    {                                                
        var int : distance = sum(i in I_ext, j in I_ext 
                              where courier == assign[i,j])
                                (D[i,j]);
    }in distance;
                              
%constraint forall(i in C)(
%            dist_sum(i,routes,D) == distances[i]);
            
constraint sum(i in distances)(i) == total_dist ;   

%solve minimize total_dist;       
              
              
%constraint forall(i in I)(              
%              obj_assign[i]==routes[i,i]);              
%constraint forall(i in I)(              
%              k_occurrences_of_x(routes[i,I_ext],obj_assign[i],2));              
                            
%constraint forall(i,j in I where i != j)(            
%                routes[j,i] != routes[i,j] \/ routes[i,j]==0);                                                                                                  
                
% forall(i in I, j in I_ext)( 
%               ((j!=n+1 /\ j!=i) -> (( routes[i,j] != routes[j,i]) \/routes[i,j]==0))) ;                            

%constraint forall(i in C)(              
%              k_occurrences_of_x(routes[n+1,I],i,1));    

%constraint forall(i in C)(
%            sum(j in I where obj_assign[j]=i)( S[j] ) <= L[i]);
                      
                      
                      
                        