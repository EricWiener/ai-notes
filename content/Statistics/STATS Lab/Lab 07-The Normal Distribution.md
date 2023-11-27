Files: lab07-slides-async.pdf

- Distributions describe values a random variable can take as well as the probability of taking those values.
- Distributions describe populations (not samples).
- The normal model is just a model
    - A lot of things apprxoximately follow a normal distribution
    - Nothing actually follows a normal distribution
- Not all distributions are normal

## Example: Whales Babys

![[Lab 07 The/Screen_Shot.png]]

![[Lab 07 The/Screen_Shot 1.png]]

- `shadeValues` is where you want to share (we shade from `3` to the left)
- `direction` tells you which direction to shade. Can be `less`, `greater`, `between`, `beyond`.

## pnorm

![[Lab 07 The/Screen_Shot 2.png]]

- This tells you the area under the curve (to the left)

![[Lab 07 The/Screen_Shot 3.png]]

- `col.shade` is the color of the shading
    
    

## pnorm (upper tail)

![[Lab 07 The/Screen_Shot 4.png]]

- You can either do `1 - pnorm(...)`
- Or you can set `lower.tail = FALSE` (it defaults to `TRUE`).
    
    

![[Lab 07 The/Screen_Shot 5.png]]

- `shadeValues` is now a vector
- `direction` is now `"between"`

## pnorm with range

![[Lab 07 The/Screen_Shot 6.png]]

![[Lab 07 The/Screen_Shot 7.png]]

## Two tails

![[Lab 07 The/Screen_Shot 8.png]]

# Normal Quantiles

![[Lab 07 The/Screen_Shot 9.png]]

- Percentiles (100 parts) and quartiles (4 parts) are both examples of quantiles
- Any value on x-axis of distribution is basically a quantile.
- `qnorm` tells us value on x-axis such that we will have `p` area to the left or right of that number in distribution

![[Lab 07 The/Screen_Shot 10.png]]

- Previously found area to the left of `3` was about `0.239`.
- We can now plug in `0.239` to `qnorm()` to get `3`

![[Lab 07 The/Screen_Shot 11.png]]

- You can't do `1 - qnorm(p)` because the spot on the normal curve range from - infinity to + infinity (there is no nice bounding where things must add up to 1).