# Lab 1

`**<-**`

- Assigns a value to a name
- `name <- value`

`read.csv(file, stringsAsFactors = TRUE)`

- `file` is the name of the file from which data are to be read
- `stringsAsFactors` should generally be set to `TRUE` (all caps): this determines how text-like data in a CSV file is interpreted by R.

`**head(x, n)**`

- `x` is a data.frame
- `n` is the number of rows to print

`**str(object)**`

- `object` is the R object about which you want to have some information (i.e., see the `str`ucture of).

# Lab 2

### `$`

- what is a dollar sign really
- `data_set_name$data_set_variable`

### `table(data_set_name$data_set_variable)`

- creates a table of the frequencies of one categorical variable

### `table(data_set_name$data_set_variable, data_set_name$data_set_variable)`

- creates a two way table of the frequencies of two categorical variables

### `barplot(table(data_set_name$data_set_variable))`

- creates a barplot of a categorical variable

### `summary(data_set_name$data_set_variable)`

- creates the five number summary of a quantitative variable

### `hist(data_set_name$data_set_variable)`

- creates a histogram of a quantitative variable

### `boxplot(data_set_name$data_set_variable)`

- creates a boxplot of a quantitative variable

## Important plotting arguments

### `main = "Title of Your Graph in Double Quotes"`

- graph title that must be inside a set of double quotes

### `xlab = "x-axis Label of Your Graph in Double Quotes"`

- the x- (horizontal) axis label that must be inside a set of double quotes

### `ylab = "y-axis Label of Your Graph in Double Quotes"`

- the y- (vertical) axis label that must be inside a set of double quotes

# Lab 3

### `read.csv("filename", stringsAsFactors = TRUE)`

- `filename` is the name of the file from which data are to be read (IN QUOTES)
- `stringsAsFactors` should generally be set to `TRUE` (all caps): this determines how text-like data in a CSV file is interpreted by R.

### `subset(data, subset)`

- `data` is the name of the data.frame you want to make a subset of
- `subset` is a logical expression indicating rows to keep. Remember to use "logical equals", `==` to test for equality; you can use regular comparison operators as well, like `>` for greater than, `<=` for less than or equal to, etc.

### `boxplot(x, ..., xlab, ylab, main, col)`

- `x` is the data you want to create a boxplot of
- `...` is additional data to include in the boxplot
- `xlab` is the x-axis label
- `ylab` is the y-axis label
- `main` is the main title
- `col` is a *vector* of color names enclosed in the `c()` function; e.g., `col = c("red", "blue", "green")`.

### `hist(x, breaks, xlim, ylim, main, col)`

- `x` is the variable you want to make a histogram of
- `breaks` is the approximate number of breaks you want in the histogram (R will sometimes ignore this to make the plot prettier)
- `xlab` is the x-axis label
- `ylab` is the y-axis label
- `main` is the main title
- `col` is a *vector* of color names enclosed in the `c()` function; e.g., `col = c("red", "blue", "green")`.

### `plot(x, y, xlab, ylab, main, col)`

- `x` is the variable you want to plot on the x axis
- `y` is the variable you want to plot on the y axis
- `xlab` is the x-axis label
- `ylab` is the y-axis label
- `main` is the main title
- `col` is a *vector* of color names enclosed in the `c()` function; e.g., `col = c("red", "blue", "green")`
- `pch` is a *vector* of **p**lotting **ch**aracters, indicated by numbers 0-25.
    
    [[]]
    

### `legend(position, legend, col, pch, ...)`

- `position` is a position for the legend, in quotes. One of `"bottomright"`, `"bottom"`, `"bottomleft"`, `"left"`, `"topleft"`, `"top"`, `"topright"`, `"right"` or `"center"`.
- `legend` is a *vector* of labels to appear in the legend, enclosed inside `c()`
- `col` is a *vector* of color names enclosed in the `c()` function; e.g., `col = c("red", "blue", "green")`
- `pch` is a *vector* of **p**lotting **ch**aracters, indicated by numbers 0-25.

# Lab 4

### `~`

- Read as "by"
- Used in side-by-side boxplots, scatterplots, and finding linear models
- `y ~ x` is the correct syntax for linear; `numeric_variable ~` categorical_variable` is the correct syntax for side-by-side boxplots

### `cor(x, y)`

- Finds the correlation coefficient between the numeric variable *x* and the numeric variable *y*

### `cor(numeric_data_frame)`

- Prints a correlation matrix for a data frame with all numeric variables

### `lm(y ~ x, data = data_name)`

- Finds the linear model between *x* and *y* from `data_name`
- You'll want to assign this a name in order to use it later

### `summary(linear_model_name)`

- Prints relevant values of a linear model

### `abline(linear_model_name)`

- Will plot the line found in `linear_model_name`

# Lab 5

### `c(...)`

- `...` is a comma-separated list of objects to be combined into a vector. These can be numbers, text/"strings", etc., but they all need to be the same kind of thing.

### `rep(x, times)`

Makes a vector consisting of `x` copy/pasted `times` times

- `x` is the thing you want `rep`eated
- `times` is the number of times you want to `rep`eat `x`

### `set.seed(seed)`

Sets the "seed" of R's random number generator. After setting the seed, the sequence of random numbers R will produce is entirely determined/predictable. This is useful for ensuring you get the same results whenever you knit your code.

- `seed` is an integer. The seed you want to set.

### `sample(x, size, replace = FALSE)`

Takes a sample of the specified size from the elements of `x` using either with or without replacement.

- `x` is a vector of elements to sample from
- `size` is a non-negative integer giving the number of items to choose from `x`
- `replace` should be `TRUE` if you want to sample with replacement; defaults to `FALSE` to sample without replacement

### `replicate(n, expr)`

Runs the code in `expr` `n` times and reports the results. *NOT* the same as `rep()` - `replicate()` re-runs code; `rep()` copies and pastes the result of one run of the code.

- `n` is the number of times to run the code in `expr`
- `expr` is an "expression" -- basically a block of code contained in curly braces {} that you want to run `n` times.

# Lab 6

- This lab didn't have any new functions

# Lab 7

### `pnorm(q, mean = 0, sd = 1, lower.tail = TRUE)`

- **`q`** refers to the value you want to find the area above or below
    - `pnorm(q, 0, 1)` gives $P(Z < q)$ where $Z$ is $N(0,1)$
- **`mean`** refers to $\mu$, defaults to 0
- **`sd`** refers to $\sigma$, defaults to 1
- **`lower.tail`** controls which direction to "shade": `lower.tail = TRUE` goes less than `q`, `lower.tail = FALSE` goes greater than `q`; defaults to `TRUE`

### `qnorm(p, mean = 0, sd = 1, lower.tail = TRUE)`

- **`p`** refers to the area under the curve
    - `qnorm(p, 0, 1)` is the number such that the area to the left of it is `p`
- **`mean`** refers to $\mu$, defaults to 0
- **`sd`** refers to $\sigma$, defaults to 1
- **`lower.tail`** controls which direction to "shade": `lower.tail = TRUE` goes less than `q`, `lower.tail = FALSE` goes greater than `q`; defaults to `TRUE`

### `plot_norm(mean = 0, sd = 1, shadeValues, direction, col.shade, ...)`

- **`mean`** refers to $\mu$, defaults to 0
- **`sd`** refers to $\sigma$, defaults to 1
- **`shadeValues`** is a vector of up to 2 numbers that define the region you want to shade
- **`direction`** can be one of `less`, `greater`, `between`, or `beyond`, and controls the direction of shading between `shadeValues`. Must be `less` or `greater` if `shadeValues` has only one element; `between` or `beyond` if two
- **`col.shade`** controls the color of the shaded region, defaults to `"cornflowerblue"`
- **`...`** lets you specify other graphical parameters to control the appearance of the normal curve (e.g., `lwd`, `lty`, `col`, etc.)

# Lab 8

### `prop_test(x, n, conf.level, alternative)` (one proportion CI)

- `x` is the number of successes (observed counts)
- `n` is the sample size
- `conf.level` is the confidence level desired
- `alternative` is set to `two.sided` as a default to provide a CI

### `prop_test(x, n, p, alternative)` (one proportion HT)

- `x` is the number of successes (observed counts)
- `n` is the sample size
- `p` is the value of the null hypothesis
- `alternative` is the direction of the alternative hypothesis (`greater`, `less`, `two.sided`), where `two.sided` is the default

### `prop_test(x = c(a,b), n = c(f,g), conf.level, alternative)` (two proportion CI)

- `x` is a vector of the successes (observed counts) where a is for the first group and b is for the second group
- `n` is a vector of the sample sizes where f is for the first group and g is for the second group
- `conf.level` is the confidence level desired
- `alternative` is set to `two.sided` as a default to provide a CI

### `prop_test(x = c(a,b), n = c(f,g), alternative)` (two proportion CI)

- `x` is a vector of the successes (observed counts) where a is for the first group and b is for the second group
- `n` is a vector of the sample sizes where f is for the first group and g is for the second group
- `alternative` is set to `two.sided` as a default to provide a CI
- Note that we do NOT need to send the value of the null hypothesis, since the value is always 0, and R knows this!

# Lab 9

- **`pt(q, df, lower.tail = TRUE)`**
    - `q` is the x-axis value you want to find an area related to
    - `df` is the degrees of freedom of the $t$ distribution
    - `lower.tail` determines whether `pt()` finds the area to the left or right of `q`. If `lower.tail = TRUE` (the default), it shades to the left. If `lower.tail = FALSE`, it shades to the right.
- **`qt(q, df, lower.tail = TRUE)`**
    - `p` is the probability or area under the curve you want to find an x-axis value for
    - `df` is the degrees of freedom of the $t$ distribution
    - `lower.tail` determines whether `pt()` finds the area to the left or right of `q`. If `lower.tail = TRUE` (the default), it shades to the left. If `lower.tail = FALSE`, it shades to the right.
- **`plot_t()`**
    - `df` refers to the degrees of freedom of the distribution to plot. You must provide this value.
    - `shadeValues` is a vector of up to 2 numbers that define the region you want to shade
    - `direction` can be one of `less`, `greater`, `beyond`, or `between`, and controls the direction of shading between `shadeValues`. Must be `less` or `greater` if `shadeValues` has only one element; `beyond` or `between` if two
    - `col.shade` controls the color of the shaded region, defaults to `"cornflowerblue"`
    - `...` lets you specify other graphical parameters to control the appearance of the normal curve (e.g., `lwd`, `lty`, `col`, etc.)
- **`qqnorm(y, ...)`**
    - `y` refers to the variable for which you want to create a Q-Q plot
    - `...` lets you control graphical elements of the plot like `pch`, `col`, etc.
- **`qqline(y, ...)`**
    - `y` refers to the variable for which you created a Q-Q plot
    - `...` lets you control graphical elements of the plot like `pch`, `col`, etc.
    - Function can only be used *after* using `qqnorm()`
- **`t.test(x, alternative = c("two.sided", "less", "greater"), mu = 0, conf.level = 0.95)`**
    - `x` is a vector of data values
    - `alternative` specifies the direction of the alternative hypothesis; must be one of "two.sided", "less", or "greater"
    - `mu` indicates the true value of the mean (under the null hypothesis); defaults to 0
    - `conf.level` is the confidence level to be used in constructing a confidence interval; must be between 0 and 1, defaults to 0.95
- `sd` gives you the sample standard deviation.
    - Ex: `sd(penguins$flipper_length_mm)`