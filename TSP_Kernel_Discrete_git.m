function TSP_Kernel_Discrete_git
% 
clear all;
close all;
format short g;
global i_time
global graph

global sigma_i
global ID0 
global sigma_J
global sigma_L
global sigma_K
global sigma_LR


i_time       = 0;

graph    = construct_graph_kroA100();


sigma_J = 1.0e-8;
sigma_L = 1.0e-8;




sigma_LR = [1 , 1];
sigma_i = (100.0)^2;  
sigma_L = (20.0)^2;

sigma_K = 1.0e-15;


ID0      = 1;
 


% Initialize 
[lb,ub] = Initialize_design_varaibles(graph);


x0 = initialize_x0(graph);

% SQP is used to search the optimal route 
fun     = @my_route_cost;
options = optimoptions(@fmincon,'Algorithm','sqp','FiniteDifferenceStepSize',1.0e-12,'OptimalityTolerance',1.0e-30,'TolFun',1.0e-30,'TolX',1.0e-30,'MaxIter',1800,'Display','iter','PlotFcn','optimplotfval','OutputFcn',@outfun);
[x,fval] = fmincon(fun,x0,[],[],[],[],lb,ub,[],options);

[J] = my_route_cost(x);


return
end

 
 
 
function [J,C] = my_route_cost(x)
global selected_id
global tour_fitness_sum_Penalty_Log_P
global graph
global ID0


node_list         = graph.ID;
sum_Penalty_Log_P = 0;              % chi-square threshold
sum_y             = 0;
 
 
% the starting node
prior_node = [graph.node(ID0).x , graph.node(ID0).y];
graph_zero = graph;
 
% remove ID0 from the list
node_list  = remove_visited_node(ID0,node_list);


id     = ID0;
K_pre  = zeros(2,2);
pre_k  = 0.0;
% create route

selected_id    = zeros(1,graph.n-1);

theta        = x(end-2:end);
theta(1)     = 10.0.^theta(1);
theta(2:3)   = 10.0.^theta(2:3);


for i_node = 1:graph.n-1
    
        % design variables of the node 
        [mu_x] = design_varaibles_of_node(x,i_node,graph);
        
        
        % probability of each node on the list using hyperparameters theta and positon x
        [id_negLik,K_ij_list] = Likelihood_of_Node(graph,node_list,mu_x,theta,prior_node,id,pre_k);
        
        % select node
        [id,negLik,y_j,mu_j,K_j,pos_k]   = Select_Next_Node(id_negLik,K_ij_list);
        
        selected_id(i_node)           = id;
        
        node  = [graph.node(id).x, graph.node(id).y];
        
        
        
        % update list of candidtae nodes
        [new_node_list] = remove_visited_node(id,node_list);
        

        % update estimate parameters 
        sum_Penalty_Log_P     = sum_Penalty_Log_P  ...
                              + negLik ;                                       % log liklihood of the i-th node
        negLik_list(i_node) = negLik;
        mu_j_list(i_node)   = mu_j; 
        % update node and id
        prior_node  = node;
        node_list   = new_node_list;
        
        pre_k       = pos_k;
end
 
 
 
% fiteness function
whole_route    = [ID0,selected_id,ID0];
fitness = fitnessFunction ( whole_route , graph_zero);
 
% paneled fitness
J         = fitness + sum_Penalty_Log_P ; %

C = [];
tour_fitness_sum_Penalty_Log_P = [whole_route,J,fitness,sum_Penalty_Log_P];
%23373.5988
return
end
%

%
function J  = Cost_of_Node(prior_node,node) 
J = norm([node - prior_node]);   
return
end
%


%%%%-------------------
%%%%-------------------


 
 
% compute likleihood of each candidate with hyperparameters theta and positon x
function [id_negLik,K_ij_list] = Likelihood_of_Node(graph,node_list,x,theta,prior_node,ID_pre,pre_k)
global sigma_i
global sigma_J
global sigma_L
 
n_node = length(node_list);
edges   = graph.edges;         % saved egdes between nodes

% for each indiviudal star in the specifed clsuter, comput its probability:
% inversely propotional to  delta Oemga
for i =1:n_node
    X(i,1) = graph.node(node_list(i)).x;
    X(i,2) = graph.node(node_list(i)).y;
    
    
    X_set(i,:) = X(i,:)  - prior_node;
    
    y_set(i)   = edges(ID_pre,node_list(i));     % data set of egdes between nodes
  
    
    
end 

%
x_i_bar = x;
a     = theta(1);
l     = theta(2:3);



for j=1:n_node
           
           x_j     = X_set(j,:);
           y_j     = y_set(j);                              % read data set of edges
           
           [K_ij] = K_THETA(x_i_bar,x_j,a,l);
           
           
           k_ij = K_ij(1,2);
           k_ji = K_ij(2,1);
           k_ii = K_ij(1,1);
           k_jj = K_ij(2,2);
           
           

           k_ii = k_ii + pre_k + sigma_i;
           
           mu_y = k_ij*(k_jj + sigma_J)^-1 * y_j;
           pos_K = k_ii - k_ij*(k_jj + sigma_J)^-1 * k_ji;

           
       
            L_ji   = (y_j - mu_y)/(pos_K + sigma_L)*(y_j - mu_y)  + log(pos_K + sigma_L);
            L_j    = y_j/(k_jj + sigma_J)*y_j + log(k_jj + sigma_J);
            negLik_list(j) = L_ji + L_j;
             

            y_j_list(j)   = y_j;
            mu_j_list(j)  = mu_y;
            K_ij_list{j}  = K_ij;
            k_ji_list(j)  = k_ji;
            pos_K_list(j) = pos_K;
           
end

% select planet with respect to the probabilty
if(sum(negLik_list) == 0)
        disp('wronn!!!')
 
end
% negLik_list = negLik_list + sum_L_j;
id_negLik   = [node_list',negLik_list',y_j_list',mu_j_list',k_ji_list',pos_K_list'];
 

return
end



function k = Kernel_Vector_of_Data_Set(x_in,X_set,a,l)
[n_r,n_c] = size(X_set);
for i=1:n_r
    x_i  = X_set(i,:);
    
    k(i) = K_THETA_ij(x_in,x_i,a,l);
end
return
end



function K = Kernel_Matrix_of_Data_Set(X_set,a,l)
[n_r,n_c] = size(X_set);
for i=1:n_r
    for j=1:n_r
        x_i    = X_set(i,:);
        x_j    = X_set(j,:);
        
        K(i,j) = K_THETA_ij(x_i,x_j,a,l);
    end
end
return
end
function [K_ij,k_ij,k_ii] = K_THETA(x_i,x_j,a,l)
n    = length(x_i);
n_j    = length(x_j);
    
k_ii = K_THETA_ij(x_i,x_i,a,l);
k_ij = K_THETA_ij(x_i,x_j,a,l);
k_ji = K_THETA_ij(x_j,x_i,a,l);
k_jj = K_THETA_ij(x_j,x_j,a,l);



K_ij = [k_ii  k_ij
        k_ji  k_jj ];
return
end
function k_ij = K_THETA_ij(x_i,x_j,a,l,c)
global sigma_K
global sigma_LR 
L      = diag(l.^2);


inv_L  = diag(l.^-2) ;
k_ij = a^2*exp(-0.5*(x_i - x_j) * inv_L * (x_i - x_j)'); 

return
end




function A_PD = fix_unpositive_matrix(A)
% [V,D] = eig(A);       % Calculate the eigendecomposition of your matrix (A = V*D*V') 
%                         % where "D" is a diagonal matrix holding the eigenvalues of your matrix "A"
[V,D] = qdwheig(A);
                        
                        
d= diag(D);           % Get the eigenvalues in a vector "d" 
d(d <= 1e-7) = 1e-7;  % Set any eigenvalues that are lower than threshold "TH" ("TH" here being 
                        % equal to 1e-7) to a fixed non-zero "small" value (here assumed equal to 1e-7)
D_c = diag(d);        % Built the "corrected" diagonal matrix "D_c"
A_PD = V*D_c*V';      % Recalculate your matrix "A" in its PD variant "A_PD"
return
end



% porpagate the sequence using priori, construct the sequence, and compute
% the paramteres after manuever at each node. Data set of the paramters are
% computed and compared to the expceted paramters? and then update the
% expected values and covaraince paramteres of the expceted trajceory at each node
 
function [id,negLik,y,mu,K,pos_k] = Select_Next_Node(id_negLik,K_ij_list)
% id_negLik = [node_list',negLik_list',y_j_list',mu_j_list',k_ji_list',pos_K_list'];
 
%  
[M,I]   = min(id_negLik(:,2));
id      = id_negLik(I,1);
negLik  = id_negLik(I,2); 
y       = id_negLik(I,3); 
mu      = id_negLik(I,4); 
K       = K_ij_list{I}; 
pos_k   = id_negLik(I,end);
 
%  
return
end
%
%
%
function [mu_x] = design_varaibles_of_node(x,i_node,graph)
 
    
n_var        = 2;

%
m_target     = x((i_node - 1)*n_var + 1  : (i_node - 1)*n_var + 2);         % dx,dy


% from [0  1] to [min_dr max_dr]

min_d_pos     = [graph.min_dx        graph.min_dy];
max_d_pos     = [graph.max_dx        graph.max_dy];
%
mu_x     = min_d_pos + (max_d_pos - min_d_pos).*m_target;   
 



return
end
 
function [new_node_list] = remove_visited_node(ID,node_list)
 
new_node_list = node_list;
 
% remove the node that has been visted
IID = find(node_list == ID);
new_node_list(IID) = [];
 

 
return
end
 
 
 
 
function [lb,ub] = Initialize_design_varaibles(graph)
% the number of edges is n-1 
lb = [];
ub = [];
 
 
    
for i_node = 1:graph.n-1
    
    [lb_node,ub_node] = Initialize_design_varaibles_node();
    
    lb = [lb,lb_node];
    ub = [ub,ub_node];
end

%


LowerA     = ones(1,1).* -3.0;  % 
UpperA     = ones(1,1).* 7.0;

LowerTheta = ones(1,2).*-3.0;  % 
UpperTheta = ones(1,2).*7.0;   % 9919


lb = [lb, LowerA, LowerTheta];
ub = [ub, UpperA, UpperTheta];
return
end
 
 
function [lb,ub] = Initialize_design_varaibles_node()

%
  
min_dx = 0.0;
min_dy = 0.0;
 
max_dx = 1.0;
max_dy = 1.0;
 
 

% expected dx and dy 
lb_dx =  min_dx;
ub_dx =  max_dx;
 
lb_dy =  min_dy;
ub_dy =  max_dy;
 
 
 
%

lb(1)    = lb_dx;
lb(2)    = lb_dy;


%
ub(1)    = ub_dx;
ub(2)    = ub_dy;




return
end
%---------------------------------------------
 
 
 
function [ fitness ] = fitnessFunction ( tour , graph)
 
 
fitness = 0;
 
for i = 1 : length(tour) -1
    
    currentNode = tour(i);
    nextNode = tour(i+1);
    
    fitness = fitness + graph.edges( currentNode ,  nextNode );
    
end
return
end
 
 
 
 
 





 
function [ ] = drawBestTour(currentSolution , graph, fitness)
for i = 1 : length(currentSolution) - 1
    
    currentNode = currentSolution(i);
    nextNode =  currentSolution(i+1);
    
    x1 = graph.node(currentNode).x;
    y1 = graph.node(currentNode).y;
    
    x2 = graph.node(nextNode).x;
    y2 = graph.node(nextNode).y;
    
    X = [x1 , x2];
    Y = [y1, y2];
    plot (X, Y, '-r');
    text(x1+0.2, y1,num2str(currentSolution(i)));
    hold on;

end




title(['Best tour','Total length: ',num2str(fitness)]);
box('on');
hold off;

return
end
 
 



 
 


function stop = outfun(x,optimValues,state)
global tour_fitness_sum_Penalty_Log_P
global fitness
global graph

stop = false;

switch state
    case 'init'
        hold on
        
    case 'iter'
        % Concatenate current point and objective function
        % value with history. x must be a row vector.
        hold off
 
        data_tour_fitness_sum_Penalty_Log_P = tour_fitness_sum_Penalty_Log_P;
        currentSolution = tour_fitness_sum_Penalty_Log_P(1:end-3);
        fitness         = tour_fitness_sum_Penalty_Log_P(end-1);
        drawBestTour(currentSolution , graph, fitness);
 
        outmsg = [ ' Shortest length = ' , num2str(fitness) ];
        disp(outmsg)
    case 'done'
        hold off
        i_time = 0;
    otherwise
end

return
end

 
 
 
function x0 = initialize_x0(graph)
init_step = [ graph.min_abs_dst_x /(graph.max_dx - graph.min_dx) , ...
              graph.min_abs_dst_y /(graph.max_dy - graph.min_dy) ] ;
          
          
mu_x0     = 0.5 - 5.0*init_step;


% 
x0 = [];
for i=1:graph.n-1
    x0 = [x0 , [mu_x0]];          %kroa100,
end

theta0 =  [3.5,  3.8,  3.5];

x0 = [x0, theta0];


return
end





