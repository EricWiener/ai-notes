Tags: February 8, 2021 7:45 AM

## **Motivating example**: dolphins

a dolphin pushed a button 15/16 times. The researchers wanted to know whether he got lucky or knew what he was doing.

- Probability of pushing correctly by luck $p = 0.50$
- Probability of knowing what he was doing $p > 0.50$

![[Simulation/Screen_Shot.png]]

You can simulate the result using a coin flip. You can then see if his results were more likely than the simulated random results. By comparing the dolphin's results with the simulation, we can get a sense of how lucky he would have to be if he were just guessing.

Flipping a coin takes too long, so we use computer simulations.

- **The law of large numbers** says that as more observations are collected, the proportion $\hat{p}$ of occurrences with a particular outcome converges to the probability $p$ of that outcome.

### R Code:

![[Simulation/Screen_Shot 1.png]]

- `count` is an array of 10,000 numbers. Each index holds the number of successes one of those 10,000 trials.
- Each trial generates 16 random numbers and then counts the number of successes (a value ≥ 0.5).
- You then plot the histogram of counts

### Distribution

- 7,8, 9 happen quite a lot
- There are only 3 occurrences of 15 heads out of 10,000 tries
- This was very rare, so the evidence is strong that the dolphin wasn't guessing because it was outside of the typical range.

## Motivating example: rock-paper-scissors

- An article found that players, particularly novices, tend to throw scissors less than 1/3 of the time.
- You want to play against your novice friend. You play 20 games and each game has two outcomes (scissors or not scissors).
- You friend will either:
    - Throw scissors 1/3 of the time
    - Throw scissors less than 1/3 of the time
    

### R Code:

![[Simulation/Screen_Shot 2.png]]

![[Simulation/Screen_Shot 3.png]]

![[Simulation/Screen_Shot 4.png]]

- This was a pretty shitty example, but it comes down to we simulated how often someone will play scissors under the hypothesis that players do it 1/3 of the time.
- If our friend played only 4/20 scissors, there is a ~15% chance of this occurring based on the simulated data according to the hypothesis that players play scissors 1/3 of the time.
- 15% is pretty big, so we can't reject the hypothesis that players play scissors 1/3 of the time. Our friends results are reasonable under the hypothesis that players do it 1/3 of the time.