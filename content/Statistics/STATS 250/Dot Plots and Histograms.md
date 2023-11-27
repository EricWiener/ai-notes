Files: Histogram.pdf
Tags: February 1, 2021 9:34 AM

# Dot Plot

![[Dot Plots /Screen_Shot.png]]

![[Dot Plots /Screen_Shot 1.png]]

- Each observation in a data set is represented by a single dot plotted along the x-axis
- Dot plots can have issues if they have a ton of samples. This will require a lot of dots.
- Dot plots also don't have additional information associated with them to tell you more about each dot (ex. what year a sample was from).

# Histogram

- One of the most effective types of plots

### **Construction**

- Find the smallest and largest value.
- Take this overall range and break it up into bins (of equal width).
- **Intervals include the right endpoint, but not the left.**

![[Dot Plots /Screen_Shot 2.png]]

![[Dot Plots /Screen_Shot 3.png]]

- Each bar (or "bin") represents a class, where the base of the bar covers the class and the height indicates the number of cases for the class.
- The above table and histogram show the **distribution** of the quantitative variable Living Area.
- You can see the overall pattern of how often possible values occur.

```cpp
// R function
hist()
```

### Describing a histogram: shape

- Shape:
    - **Mode**: The mode of a set of data is the most frequently observed value. The mode for a distribution/histogram is the value/class that corresponds to a peak. A distribution is **unimodal** if it has one mode, **bimodal** if it has two modes, and **multimodal** if it has three or more modes.
    - **Symmetry**:
        
        ![[Dot Plots /Screen_Shot 4.png]]
        
- Center: eg. mean or median
- Variability: standard deviation or IQR.
- Outliers: a data point that doesn't seem to be consistent with the bulk of the data. They should not be discarded without justification. You should investigate why there are outliers.

### Outliers

- The IQR and median are not affected much by outliers.
- The mean, range, and standard deviation are sensitive to outliers
- The mean follows the skew of a histogram. For a right-skewed distribution, the mean is lower. For a left-skewed distribution, the mean is higher.

### When to use mean+standard deviation vs median and IQR

![[Dot Plots /Screen_Shot 5.png]]

![[Dot Plots /Screen_Shot 6.png]]

![[Dot Plots /Screen_Shot 7.png]]

![[Dot Plots /Screen_Shot 8.png]]