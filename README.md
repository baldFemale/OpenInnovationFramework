# OpenInnovationFramework
* generalSpecialist 
  * search depth & width 
* generalSpecialist2 
  * multiple state 


# Same key and unique features that cause version divisions

## Reproduction
* Fixed element for the unknown domain. 
* Only search half of the depth for Generalist
* This is similar to the sub-task, or the decomposition of task

## Reproduction_cognitive
* follow the cognitive local search
* Generalist domain have a simplified representation of the state (i.e., "A" and "B")
* Thus, the valid state bit includes [0,1,2,3, "A","B","\*"] where "\*" refers to the unknown domain, while 0-3 refers to the specialist perception/decision.
* Average and max pooling as the cognitive algorithm

## Reproduction_cognitive_8
* Expand the 4 states framework into 8 states, where the decision representation/abstractiion is of 3 levels.
* Generalist domain have a simplified representation of the state (i.e., "A" and "B" for the top level; "a", "b", "c", "d" for the middle level)
* The representation of the 3-level knowledge abstraction

---A---|---B-----

--a---b-|-c---d---

-0-1-2-3|4-5-6-7--


## Reproduction_diff_GS_roles
* Generalist and Specialist **domain** will have unique cognitive search algorithm
  * Generalist will use average_cognitive_search
  * Specialist will use default_cognitive_search
  * T shape will use either average or default pattern, depending on which bit T shape is going to change
* Provide a new indicator *potential_fitness* to highlight the value of Generalist
* Agent class will have two state representation systems
  * Original/Distinct state representation (e.g., "0", "1", "2", "3" when state is equal to 4)
  * Cognitive state representation will include unknown depth symbols or abstraction symbols (e.g., "A" for \["0", "1"]) and unknown domain symbol (i.e., "*")
  * The double-representation system will enable the communication between Agents using the distinct state as bridge

## Socialized_innovation_crowdsourcing
* **Three** model specifications make up the key contribution to NK landscape literature
  * First, GS and corresponding cognitive search pattern build the participant set-ups.
  * Second, the proportion of GS and their communication build the platform or community set-ups.
  * Third, the platform resilienceshred a light on a new dimension of innovation performance or collective intelligence.
* We highlight the idea polishment story such that agents should initialize their ideas based on the observation and polish the idea forward. The polishment degree will play an important role in the platform resiliance. Such a polishment logic is based what the previous literature hold, that is, idea generation and thus the tradoff between knowledge diversity and knowledge depth, but is more than the temporary consideration and generate a more general framework for the open innovation communities.
  * Seed Agents provide the initialization points for the new-comers
  * New-comer agents are exposed to particular ideas, and continue their own search
  * Finally, the whole community will achieve some performance level measured by the aggregated performance of its members.
* Platform design
  * Randon exposure plan as the performance baseline
  * Rank-directed exposure will evaluate the idea firstly based on the crowd. Top ideas might get a higher exposure rate to the crowd.
  * Interest-directed exposure means each agent will gather around what is most attractive to them, and initialize their own search there.
  * Simulated intervention
