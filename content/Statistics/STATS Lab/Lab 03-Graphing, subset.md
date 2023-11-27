### Reading in data in R

![[Lab 03 Gra/Screen_Shot.png]]

- `stringsAsFactors = TRUE` just means treat the strings in the CSV as levels of categorical variables and not as text data (ex. a product review).
- Categorical variables are called factors in R.

### Creating a subset of data

![[Lab 03 Gra/Screen_Shot 1.png]]

- `species == "Chinstrap"` is a logical expression.
- You could also have done `penguins$species == "Chinstrap"`
- We are saying take the values from penguins where the species value is equal to "Chinstrap"
- You can use `head`, `str`, or `levels` to find out what values a variable takes

You can also create a subset using square brackets:

```cpp
chinstrap <- penguins[penguins$species == "Chinstrap", ]
```

- Use the square brackets to select the rows where species is "Chinstrap"
- Notice that there is a `,` after `penguins$species == "Chinstrap"`. This selects all the columns (we just leave the column selector blank).
- You will get an error if you don't include the `,`m

You can tabulate this data to make sure it worked by doing

```r
table(chinstrap$species)
```

### Subsetting variables

![[Lab 03 Gra/Screen_Shot 2.png]]

- Here, we are just getting all the body masses of male penguins.
- Since we are subsetting a single variable, we don't need the comma to select columns (there only is one column we are working with)

# Box plots

![[Lab 03 Gra/Screen_Shot 3.png]]

You can also create side-by-side box plots by passing in multiple lists:

![[Lab 04 Cor/Screen_Shot 2.png]]

- The `names` argument receives a list. `c()` stands for combine and is a way to make a list.

**Adding color to a box plot:**

![[Lab 03 Gra/Screen_Shot 4.png]]

- You can pass a list of colors

# Histograms

![[Lab 03 Gra/Screen_Shot 5.png]]

You can change the number of `breaks` to change the number of bins.

![[Lab 03 Gra/Screen_Shot 6.png]]

- Note that `breaks` is only a suggestion and R sometimes chooses not to use it in order to keep the data looking nice.
- You want the histograms to be smooth. You also want to be able to see what the bin width is.

# Scatterplots

![[Lab 03 Gra/Screen_Shot 7.png]]

- You can visualize the relationship between two numeric variables
- One variable is the explanatory (x) variable, one is the response (y) variable
- If you say "make a scatterplot of room and board costs vs. in-state tuition," we say "y vs. x", so room and board should be on the y axis.
- You plot as an ordered pair (x, y). You use the `plot` function.

**Adding color to a scatterplot:**

![[Lab 03 Gra/Screen_Shot 8.png]]

- You can use the `col` argument to specify what colors to use.
- It will assign color depending on the type of `penguins$species`. Note you use the square brackets
- The order the color is applied in is assigned by the variables alphabetically (A → Z)
    
    ![[Lab 03 Gra/Screen_Shot 9.png]]
    

**Adding a legend**

![[Lab 03 Gra/Screen_Shot 10.png]]

- You can add a legend using the `legend` function.
- The first argument is where to position it
- `legend` argument is the text you want to show up
- `col` specifies what color to use for the text in the legend
- `pch`: this controls what plotting character to use

**Modifying pch**

![[Lab 03 Gra/Screen_Shot 11.png]]

- You can use `pch` to modify the shape

### How to use color

- Color should be used to convey information. Don't just make colorful plots for no reason.
- Use different colors only when they correspond to different meanings in the data
- Something like a histogram should be only one color
- Avoid using a combination of red and green in the same display to help out color blind people. You can fix this issue by adding different shapes.