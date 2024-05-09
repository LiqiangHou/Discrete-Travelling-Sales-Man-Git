



 
function graph = construct_graph_kroA100()
% 100 nodes
nodes = [1380,2848,3510,457,3888,984,2721,1286,2716,738,1251,2728,3815,3683,1247,123,1234,252,611,2576,928,53,1807,274,2574,178,2678,1795,3384,3520,1256,1424,3913,3085,2573,463,3875,298,3479,2542,3955,1323,3447,2936,1621,3373,1393,3874,938,3022,2482,3854,376,2519,2945,953,2628,2097,890,2139,2421,2290,1115,2588,327,241,1917,2991,2573,19,3911,872,2863,929,839,3893,2178,3822,378,1178,2599,3416,2961,611,3113,2597,2586,161,1429,742,1625,1187,1787,22,3640,3756,776,1724,198,3950;939,96,1671,334,666,965,1482,525,1432,1325,1832,1698,169,1533,1945,862,1946,1240,673,1676,1700,857,1711,1420,946,24,1825,962,1498,1079,61,1728,192,1528,1969,1670,598,1513,821,236,1743,280,1830,337,1830,1646,1368,1318,955,474,1183,923,825,135,1622,268,1479,981,1846,1806,1007,1810,1052,302,265,341,687,792,599,674,1673,1559,558,1766,620,102,1619,899,1048,100,901,143,1605,1384,885,1830,1286,906,134,1025,1651,706,1009,987,43,882,392,1642,1810,1558];
x = nodes(1,:);
y = nodes(2,:);


%
graph.n = length(x);


% pos of vertex
graph.ID = [1:graph.n];
 
 
for i = 1 : graph.n
    graph.node(i).x = x(i);
    graph.node(i).y = y(i);
end
 
graph.edges  = zeros(  graph.n , graph.n );
graph.dist_x = zeros(  graph.n , graph.n );
graph.dist_y = zeros(  graph.n , graph.n );
 


for i = 1 : graph.n
    for j = 1: graph.n
        x1 = graph.node(i).x ;
        x2 = graph.node(j).x;
        y1 = graph.node(i).y;
        y2 = graph.node(j).y;
        
        graph.edges(i,j) = sqrt(  (x1 - x2) ^2 + (y1 - y2)^2  );
        graph.dist_x(i,j)  = [(x1 - x2)];
        graph.dist_y(i,j)  = [(y1 - y2)];

        
        
    end


    graph.mean_dist_x(i) =  sum(graph.dist_x(i,:))/(graph.n - 1);
    graph.mean_dist_y(i) =  sum(graph.dist_y(i,:))/(graph.n - 1);
end
%
edges  = graph.edges;
dist_x = graph.dist_x;
dist_y = graph.dist_y;


graph.min_dx   = min(min(dist_x)) ;
graph.max_dx   = max(max(dist_x)) ;

graph.min_dy   = min(min(dist_y)) ;
graph.max_dy   = max(max(dist_y)) ;


abs_dist_x = abs(dist_x);
abs_dist_y = abs(dist_y);

graph.min_abs_dst_x = min(abs_dist_x(abs_dist_x > 0));
graph.min_abs_dst_y = min(abs_dist_y(abs_dist_y > 0));

graph.max_abs_dst_x = max(abs_dist_x(abs_dist_x > 0));
graph.max_abs_dst_y = max(abs_dist_y(abs_dist_y > 0));




graph.min_edge = min(edges(edges >0)) ;
graph.max_edge = max(edges(edges >0)) ;
graph.mean_edge = mean(edges(edges >0)) ;



graph.min_std_dist_x = min(std(dist_x)); 
graph.max_std_dist_x = max(std(dist_x)); 

graph.min_std_dist_y = min(std(dist_y)); 
graph.max_std_dist_y = max(std(dist_y)); 

graph.min_std_edges =  min(std(edges));
graph.max_std_edges =  max(std(edges));
return
end
