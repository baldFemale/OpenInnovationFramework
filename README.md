# OpenInnovationFramework
* generalSpecialist 
  * search depth & width 
* generalSpecialist2 
  * multiple state 


# Same key and unique feature that cause version divisions

## Reproduction
* Fixed element for the unknown domain. 
* Only search half of the depth for Generalist
* This is similar to the sub-task, or the decomposition of task

## Reproduction_cognitive
* follow the cognitive local search
* Generalist domain have a simplified representation of the state (i.e., "A" and "B")
* Thus, the valid state bit includes [0,1,2,3, "A","B","\*"] where "\*" refers to the unknown domain, while 0-3 refers to the specialist perception/decision.
* average and max pooling as the cognitive algorithm

## Reproduction_cognitive_8
* expand the 4 states framework into 8 states, where the decision representation/abstractiion is of 3 levels.
* Generalist domain have a simplified representation of the state (i.e., "A" and "B" for the top level; "a", "b", "c", "d" for the middle level as below)
 *     A      |     B
 *  a      b  |  c      d
 * 0 1    2 3 | 4 5    6 7



## Reproduction_diff_GS_roles
* Generalist and Specialist *domain* will have unique cognitive search algorithm
* Provide a new indicator *potential fitness* to highlight the value of Generalist



