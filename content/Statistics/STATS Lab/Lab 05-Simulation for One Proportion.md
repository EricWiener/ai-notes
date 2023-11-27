Files: lab05-slides-async.pdf

## Vectors

- A way to store a collection of objects in R
- Create it with the `c()` function. This stands for **c**ombine.
- Everything needs to be the same type.

```r
x <- c("hi", "test") # This is a character vector
x <- c(3, 5.6, -7.2) # This is a number vector
x <- c("mixed", 5) # 5 will get converted to string "5"
```

## Defining a Hypothesis

![[Lab 05 Sim/Screen_Shot.png]]

# Worked Example

![[Lab 05 Sim/Screen_Shot 1.png]]

- Here $\hat{p}$, the observed (sample) proportion of correct guesses, is $\frac{24}{26} = 0.923$

![[Lab 05 Sim/Screen_Shot 2.png]]

- We simulate assuming that the null hypothesis is true.
- Under $H_0$, the probability of a correct guess is 50%
    - We can use 100 poker chips (50 blue and 50 yellow)
- One reptition is drawing 26 chips (one for each student)
    - We should sample with replacement because otherwise the probability of drawing a blue poker chip would change over time.

## `rep`

You can use `rep()` to create a vector pre-filled with values. The first argument is the value to use and the second argument is how many values.

![[Lab 05 Sim/Screen_Shot 3.png]]

You can also use `c()` to combine multiple `rep()` into one vector.

![[Lab 05 Sim/Screen_Shot 4.png]]

- You use the number of values in order to get the number of decimals in your proportion. Example, if your proportion is 0.855, you would generate 1000 chips so you can represent this proportion was values. This is so you can assign 855 blue chips and 1000 - 855 yellow chips.

## `seed`

You can seed the random generation in R with `set.seed(8362)` where `8362` is an example number to use.

## `sample`

You can use `sample` to generte a random sample from a vector:

![[Lab 05 Sim/Screen_Shot 5.png]]

![[Lab 05 Sim/Screen_Shot 6.png]]

![[Lab 05 Sim/Screen_Shot 7.png]]

- `sample` will produce vector
- You can count how many times each element occurs by creating a frequency table with `table()` or using `sum()` with the logical equivalence (`==`) to count how many occurences of a spefic value occur.
- We can get our first $\hat{p}$ of the simulation by just doing `numBlues / 26`. This tells us the first simulated proportion of blues
    - Note that both the simulated proportion and the proportion received through the original sampling are both referred to as $\hat{p}$
    

## `replicate`

We can use `replicate` to re-run our sampling.

![[Lab 05 Sim/Screen_Shot 8.png]]

- Because we don't assign the `sum() / 26` operation to a variable, it returns its value
- `replicate` collects all these returned values and produces a vector with all the results.
- Note that `rep()` is a different function (basically fancy copy/paste)
- The first argument is the number of times to run the code and the second bit is the code to run.

## Displaying results

![[Lab 05 Sim/Screen_Shot 9.png]]

- We can display the results of the `replicate` using a histogram with `hist`
- In order to draw the proportion of students who correctly guessed which dog was the teachers, you need to do:
    - `xlim = c(0, 1)`: increase the x-limit. This takes a vector with the range of values to cover.
    - `abline()`: will draw a line on the graph. `v` stands for a vertical line.
        
        

## Computing a p-value

![[Lab 05 Sim/Screen_Shot 10.png]]

- We divide by `100`, which is the number of trials.
- Here, none of the simulated results had a proportion greater than 24/26.

**Making a conclusion:**

![[Lab 05 Sim/Screen_Shot 11.png]]