Created: July 17, 2020 8:25 AM
Type: Notes

Parameterized by:

![https://paper-attachments.dropbox.com/s_416E449F70987E51AD7B2B8E34D5B0709B59F6F8A3F3B035C7704F247FA067F1_1586179034345_Screen+Shot+2020-04-06+at+9.17.08+AM.png](https://paper-attachments.dropbox.com/s_416E449F70987E51AD7B2B8E34D5B0709B59F6F8A3F3B035C7704F247FA067F1_1586179034345_Screen+Shot+2020-04-06+at+9.17.08+AM.png)

![https://paper-attachments.dropbox.com/s_1B0611DAAD9036AFC43296683495D7848754213B8F22EC19969E8C17BBE7EC02_1582574774978_Screen+Shot+2020-02-24+at+3.06.12+PM.png](https://paper-attachments.dropbox.com/s_1B0611DAAD9036AFC43296683495D7848754213B8F22EC19969E8C17BBE7EC02_1582574774978_Screen+Shot+2020-02-24+at+3.06.12+PM.png)

- D: number dimensions
- x: vector of probabilities we want
- p(x): probability x lies within Gaussian distribution
- $\bar{\mu} = E[\bar{x}]$
- $\Sigma = E[(\bar{x}-\bar{\mu})(\bar{x}-\bar{u})^T]$

We use the Multivariate Gaussian model when our data isn’t just in the reals, but has multiple dimensions. Now, our $\bar{x}, \bar{\mu} \in \mathbb{R}^d$. We need to use a covariance matrix instead of a scalar.

**Covariance matrix:**

![https://paper-attachments.dropbox.com/s_416E449F70987E51AD7B2B8E34D5B0709B59F6F8A3F3B035C7704F247FA067F1_1586180003986_IMG_73AB595F9A3B-1.jpeg](https://paper-attachments.dropbox.com/s_416E449F70987E51AD7B2B8E34D5B0709B59F6F8A3F3B035C7704F247FA067F1_1586180003986_IMG_73AB595F9A3B-1.jpeg)

![https://paper-attachments.dropbox.com/s_1B0611DAAD9036AFC43296683495D7848754213B8F22EC19969E8C17BBE7EC02_1582575292989_image.png](https://paper-attachments.dropbox.com/s_1B0611DAAD9036AFC43296683495D7848754213B8F22EC19969E8C17BBE7EC02_1582575292989_image.png)

- $\mathcal{E}$ is the covariance matrix
- Diagonal terms tell you the variance of an element with itself
- The off-diagonal terms tell you correlation - covariance between $x_i$ and $x_j$
- $|\Sigma|$ is the determinant of the covariance matrix
- $\Sigma$ is symmetric and positive definite (all eigenvalues positive)
- $\Sigma$ can be decomposed in the form $UDU^T$ (D is diagonal)
- The mean of the Gaussian is the center point