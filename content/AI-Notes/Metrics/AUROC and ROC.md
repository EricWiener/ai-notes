# AUROC and ROC

Source: [https://www.youtube.com/watch?v=4jRBRDbJemM](https://www.youtube.com/watch?v=4jRBRDbJemM)

If you were trying to calculate logistic regression for whether a mice is obese or not, you would end up with a prediction between 0 and 1. You could choose a threshold of 0.5 to decide obese or not, but you could move this threshold up or down if you cared about being extra careful or not (imagine this was ebola and you wanted to be extra sure that no one had it).

![https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581280615998_Screen+Shot+2020-02-09+at+3.36.53+PM.png](https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581280615998_Screen+Shot+2020-02-09+at+3.36.53+PM.png)

![https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581280653614_Screen+Shot+2020-02-09+at+3.37.28+PM.png](https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581280653614_Screen+Shot+2020-02-09+at+3.37.28+PM.png)

![https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581280700547_Screen+Shot+2020-02-09+at+3.38.18+PM.png](https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581280700547_Screen+Shot+2020-02-09+at+3.38.18+PM.png)

Every-time you move the threshold, you get a new confusion matrix. However, the confusion matrix sometimes doesn’t change and it will only change when you move above/below a point. If you change the threshold, but the points remained classified the same, it won’t change your confusion matrix. This is why the ROC appears stepped.

![https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581280955378_Screen+Shot+2020-02-09+at+3.42.31+PM.png](https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581280955378_Screen+Shot+2020-02-09+at+3.42.31+PM.png)

ROC curves provide a display between sensitivity and 1 - specificity. Sensitivity is how many positive you predict out of total positive (**what proportion of obese samples were correctly classified**).

Specificity is correctly predicted negative out of total negative. 1 - Specificity is the number of incorrectly predicted negative out of total negative (**the proportion of not obese samples that were incorrectly classified).**  

![https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581281239232_Screen+Shot+2020-02-09+at+3.47.15+PM.png](https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581281239232_Screen+Shot+2020-02-09+at+3.47.15+PM.png)

If you set your threshold very low, you will correctly predict all the positive values as positive.

If you set it super low, you will incorrectly predict all the positive values as negative. You will have 0 sensitivity, but you will correctly classify all negatives as negatives. You will have a lower false positive rate. This would correspond to the point (0,0).

![https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581281949609_Screen+Shot+2020-02-09+at+3.59.04+PM.png](https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581281949609_Screen+Shot+2020-02-09+at+3.59.04+PM.png)

You want a lower false positive rate and a higher true positive rate.

![https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581282119883_Screen+Shot+2020-02-09+at+4.01.56+PM.png](https://paper-attachments.dropbox.com/s_0A2774EF83209DC129C46B6AD173B78B76341D5EFD94012ED6D9E926A2675D24_1581282119883_Screen+Shot+2020-02-09+at+4.01.56+PM.png)

AUROC is the area under the curve. It allows you to compare two ROC curves. You can decide what classifier you want depending on which AUROC is higher.