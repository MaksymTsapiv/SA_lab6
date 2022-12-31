
## System Analysis lab6: Reinforcement learning
### Author: Maksym Tsapiv
### Variant: 0

> The Task

Create a maze (more like room with obstacles) with one `entry` and one `exit`. Put an `agent` on the `entry` and make it learn how to find the `exit` using reinforcement learning.


> So what is reinforcement learning?

Reinforcement Learning is an aspect of Machine learning where an agent learns to behave in an environment, by performing certain actions and observing the rewards/results which it get from those actions.

> Environment

I made a randomly generated maze with different types of obstacles:
* Walls ( `#` ) -- simple walls, `agent` can not go through them neither can it move them. 
* Firepits ( `$` ) -- instantly kills the `agent` stopping the round.
* Teleports ( `@` ) -- basicaly a teleport, always in pairs, if `agent` steps into teleport it instantly comes out of the other teleport somewhere on the maze. Works both ways, linked teleports are liked permanently.
* Emptiness ( ` ` ) -- `agent` can freely go through.
* Entry ( `>` ) -- works just like `Emptiness` except it also is the point where the `agent` starts its round.
* Exit ( `<` ) -- if `agent` finds `Exit` round stops, `agent` wins.
* Path ( `=` ) -- serves visualization purposes only, shows the path `agent` made from the `Entry` to where it stoped.

Mazes can be different sizes but they always will be surrounded with `Walls` so that the `agent` stays inside.

> ! Important: It is NOT guaranteed that there exists such path that connects `Entry` and `Exit`, but with default parameters it is highly probable. You can always change the random seed to regenerate the maze!

For more detailed inforamtion about environment generation you can find in `environment.py` inside the class `Maze`.

```python
class Maze:
    def __init__(self, conf: Config) -> None:
        self.conf = conf
        self.maze: List[List[Patch]] = [[None for _ in range(conf.grid_size)] for _ in range(conf.grid_size)]
        self.agent_state: AgentState = AgentState() 
        self.entry: Optional[Patch] = None
        self.exit: Optional[Patch] = None
```
Also an example of the (solved) maze 27x27 (25x25 if we don't count the walls) 

```
[# # # # # # # # # # # # # # # # # # # # # # # # # # #]
[#   #     # @ @       # @                     #     #]
[#   #     @ #                             #     #   #]
[#         @     @   $                         #     #]
[# #                                                 #]
[#   $   $                       #   @           @   #]
[#           @                 $   #       @         #]
[# #           #     #     # @             #   $     #]
[# #     $       # #           @     @               #]
[#                 $       #     #     #       #     #]
[#           # #     @         # #               @   #]
[#   # @         @           @ #   #   $   #         #]
[#     =         $       $                   $ #     #]
[#     =               #     #             #         #]
[#     =         @           #           #   #     @ #]
[#     =         #     @           #         $     @ #]
[# > = =                     #                 $     #]
[#           #   @     $   $     #           #     # #]
[#           # #                                     #]
[#             #       #   #         @       @ # $   #]
[# @     #           #           #           #       #]
[#     #   #             $ @ @   #                   #]
[#         @ #           @           $     $         #]
[# $   #         #                       #   # #     #]
[#                                   #   = = <       #]
[#     $       $           # @         $ @           #]
[# # # # # # # # # # # # # # # # # # # # # # # # # # #]
```

> States and Actions

The `agent` has only 3 actions:
* Turn left 90 degrees
* Turn right 90 degrees
* Step forward

Therefore the number of states is [`grid_size ^ 2 * 4`] because for every position in the maze (grid) our `agent` can stay heading in 4 direction thus making 4 unique states per 1 position.

> Learning

I use tabular Q-Learning for our agent. Q-Table is a simple lookup table where we calculate the maximum expected future rewards for action at each state. Basically, this table will guide us to the best action at each state. 

Each Q-table score will be the maximum expected future reward that the `agent` will get if it takes that action at that state. This is an iterative process, as we need to improve the Q-Table at each iteration.

That means that we need our `agent` to explore the enviroment and fill up the Q-table before we can rely on it.

> Impementation

Starting with config:
```yaml
fire_prob: 0.04 # probility of fire appearing on the maze, it essentialy is configuration of how many firepits will be in the maze
wall_prob: 0.09 # same but for walls
teleport_prob: 0.03 # same but for teleports
move_prob: 0.9 # probability that the move forward will be successful (agent will stay still if move is not successful)
turn_prob: 0.9 # probability that the turn will be successful (agent will turn the other way if move is not successful)
grid_size: 27 # size of the maze
learning_rate: 0.5
discount_factor: 0.9
exploration_rate: 0.7
exploration_decay_rate: 0.99982
max_timesteps_per_episode: 100 # number of steps agent can make before force-end of the round
max_episodes_to_run: 10000 # number of runs (epoches)
```
These are configurations for the `enviroment` and the `agent` and they are VERY important.

> About 
`exploration_rate` and `exploration_decay_rate`

These are configurations of `learning process`.
Let's see the code to understand what they are for. 

```python
enable_exploration = (1 - self.exploration_rate) <= np.random.uniform(0, 1)
if enable_exploration:
    next_action = np.random.randint(0, self._num_actions)
else:
    next_action = np.argmax(self.q[next_state])
```

```python
self.exploration_rate *= self.exploration_decay_rate
```

If we choose exploration `(enable_exploration == True)`, we perform a random action. If we choose exploitation, we perform the best possible action for this state.

So, `exploration rate` is used to force our agent not to use `Q-table` early on but explore the enviroment instead. And `exploration decay` is to slowly make our `agent` use `Q-table` more and stop exploring. The `exploration decay` value is very important because we want to find such value that our `agent` would learn enogh to use `Q-table` and that's why,  for example, in my case if I put `0.99` instead of `0.99982` the agent would not keep up with decaing pace and stoped exploring too early and then the `agent` would use unfinished `Q-table` and be as good as random. That's how important these configurations are. 

> About `learning_rate` and `discount_factor`

We have received a reward from our previous step, and we know our future state and what action to perform next. Now, recalculate Q[state, action] in the Q-Table using the update formula.

```python
self.q[self.state, self.action] = (1 - self.learning_rate) * self.q[
    self.state, self.action] + self.learning_rate * (reward + self.discount_factor * np.max(self.q[next_state]))
```

So, `learning rate` and `discount factor` are basically to update our `Q-table` and change it so `agent's` previous experience would not matter that much. Meaning that the agent would learn and not simply follow random past.

> Rewards

Setting the rewards is also very important part to make our `agent` not only `logical` bur `efficient`.

Rewards are given in response to the `agent's` action accordeing to the state the `agent` is in after the action. Rewards can be punishments for bad actions.

> About the reward system for my `agent`

First of all, I set reward for any kind of `turn` (turn right, left...) to be `-1` so that `agent` wolud not stay spinning around and also try to minimize uneeded turns.

Then, for the step into `empty` patch `agent` also recieves `-1` so that `agent` would look for the shortest path possible.

If `agent` tries to go through the `wall` it recieve `-10` because it is meaningless move and `agent` should realize it soon.

If `agent` steps on `fire` it recieve `-1000` becuse death is bad and we want to make absolutely sure that our agent knows that.

If `agent` steps on `exit` it recieve `1000` so that agent would always go the `exit`. Important note is that making reward for the win too big might be hurmful for the agent because once it achieves the reward velue will be so big that the agent would stop exploring better ways to achieve it or basicaly stop being efficient because those `-1`'s for extra steps would not matter much if the reward is infinite.

And finally if `agent` steps on `teleport` it also recieves `-1`. I made it that way because I do not want to add for example relative value depending on how far this will get the `agent` or any other `heuristics` because it is up the `agent` to decide should or should not it use `teleport` and it should make the decision the same way as with the `fire` -- the hard way. 

> About probabilities to do something wrong

Those probabilities do not change the process or basically anything about the agent, but if you make it too probable to make something wrong, you shoud not expect the `agent` to learn something because he do not understand what he does wrong. But overall even `0.9` for the successfull move is enough to make the learning process much harder for the `agent`.

### I think I covered preetty much everything, so last is usage:

> before run:

go check `conf.yaml` and change something if needed.

> to run:
```shell
$ python3 main.py
```
> output:

```
Confing infornation

Config(fire_prob=0.04, wall_prob=0.09, teleport_prob=0.03, move_prob=0.9, turn_prob=0.9, grid_size=27, learning_rate=0.5, discount_factor=0.9, exploration_rate=0.7, exploration_decay_rate=0.99982, max_timesteps_per_episode=100, max_episodes_to_run=10000)
[(<PatchType.BLANK: ' '>, 0.84), (<PatchType.FIRE: '$'>, 0.04), (<PatchType.WALL: '#'>, 0.09), (<PatchType.TELEPORT: '@'>, 0.03)]

This is the maze you generated (I like this one) 

[# # # # # # # # # # # # # # # # # # # # # # # # # # #]
[#   #     # @ @       # @                     #     #]
[#   #     @ #                             #     #   #]
[#         @     @   $                         #     #]
[# #                                                 #]
[#   $   $                       #   @           @   #]
[#           @                 $   #       @         #]
[# #           #     #     # @             #   $     #]
[# #     $       # #           @     @               #]
[#                 $       #     #     #       #     #]
[#           # #     @         # #               @   #]
[#   # @         @           @ #   #   $   #         #]
[#               $       $                   $ #     #]
[#                     #     #             #         #]
[#               @           #           #   #     @ #]
[#               #     @           #         $     @ #]
[# >                         #                 $     #]
[#           #   @     $   $     #           #     # #]
[#           # #                                     #]
[#             #       #   #         @       @ # $   #]
[# @     #           #           #           #       #]
[#     #   #             $ @ @   #                   #]
[#         @ #           @           $     $         #]
[# $   #         #                       #   # #     #]
[#                                   #       <       #]
[#     $       $           # @         $ @           #]
[# # # # # # # # # # # # # # # # # # # # # # # # # # #]

The agent is learning...
This is how the learning went

[ 87 245 269 285 327 331 350 348 354 352 342 351 359 355 352 356 348 362
 359 354 353 354 368 351 362]

This is how the agent solves the maze after learning

[# # # # # # # # # # # # # # # # # # # # # # # # # # #]
[#   #     # @ @       # @                     #     #]
[#   #     @ #                             #     #   #]
[#         @     @   $                         #     #]
[# #                                                 #]
[#   $   $                       #   @           @   #]
[#           @                 $   #       @         #]
[# #           #     #     # @             #   $     #]
[# #     $       # #           @     @               #]
[#                 $       #     #     #       #     #]
[#           # #     @         # #               @   #]
[#   # @         @           @ #   #   $   #         #]
[#     =         $       $                   $ #     #]
[# = = =               #     #             #         #]
[# =             @           #           #   #     @ #]
[# =             #     @           #         $     @ #]
[# >                         #                 $     #]
[#           #   @     $   $     #           #     # #]
[#           # #                                     #]
[#             #       #   #         @       @ # $   #]
[# @     #           #           #           #       #]
[#     #   #             $ @ @   #                   #]
[#         @ #           @           $     $         #]
[# $   #         #                       #   # #     #]
[#                                   #   = = <       #]
[#     $       $           # @         $ @           #]
[# # # # # # # # # # # # # # # # # # # # # # # # # # #]
```
```
[ 87 245 269 285 327 331 350 348 354 352 342 351 359 355 352 356 348 362
 359 354 353 354 368 351 362]
 ```
 These numbers represent how the `agent` was learning. Every number is the amount of times the agent found `exit` out of 400 runs, for example here we see that last 400 runs `agent` found `exit` 362 times, that means that it either died or walked around for more then 100 steps 38 runs.

 So, basically these numbers should show how the `agent` is learning and improving.

 If we run the `agent` without chance to move or turn wrong, these numbers wolud look like this:
 ```
 [129 293 340 347 362 376 383 385 392 389 393 393 395 394 393 399 399 398
 397 400 399 399 400 398 399]
 ```

 That means that `agent` knows how to solve the maze. Usually he does)