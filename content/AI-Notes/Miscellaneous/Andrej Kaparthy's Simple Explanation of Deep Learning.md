---
tags: [flashcards]
source: https://youtu.be/PlLmdTcsAWU
summary: how Andrej Kaparthy explains deep learning
---

TLDR: I will use a concrete example of recognizing objects in images like cats and dogs. Traditionally, programmers would have to write a program with a series of instructions to do this, but it's too complicated because there are so many different ways that a cat or dog can look. Instead, deep learning uses a large dataset of images and their labels to teach a computer to recognize things. This is done by using a neural network, which is like a mathematical equation with a bunch of parameters that can be adjusted. The network takes in a mathematical representation of the image and then based on the current parameter settings, it will make some predictions. You compare these predictions to the labels and then update the parameters to try to improve the predictions. You keep repeating this process until you find a set of good parameters. It's a different way of programming that allows us to specify what we want and let the computer figure out how to do it.

# Original Transcript
how would you describe deep learning to i don't know
your parents or you know uncle or something who doesn't work in in the space let's use a specific
example because i think it's useful so let's let's talk about image recognition right so we have images and they are
just um images are made up to computer of a large number of pixels and each pixel just tells you the amount
of brightness in the red green and blue channel at that point and so you have a large array of numbers and you have to
go from that to hey it's a cat or a dog and typical conventional software is
written by a person programmer writing a series of instructions to go from the
input to the output so in this case you want someone to write a program for how do you combine these millions of pixel
values into like is it a cat or a dog turns out no one can write this program it's a very
complicated program because there's a huge amount of variability in what a cat or doc can look like in different
brightness conditions arrangements poses occlusions basically no one can write this program
so deep learning is a different class of programming in my mind
where no one is explicitly writing the algorithm for this recognition problem instead we are
structuring the entire process slightly differently so in particular we arrange a large data set of uh possible images
and um the desired labels that should come out from the algorithm so hey when you get
this input this is a cat when you get this output this should be a dog and so on so we're kind of stipulating what is
the desired behavior on a high level we're not talking about what is the algorithm we're measuring the performance of some algorithm
and then you need some and then roughly what we do is we lay out a neural network which is these um
it's it's a bunch of neurons connected to each other with some strengths and you you feed them images and they
predict what's in them and the problem now is reduced because
um you're just trying to find the setting of these synaptic strengths between the neurons so that the outcomes
are what you want and so as an example the 2012 imagenet model which was roughly 60 million
parameters so the weights of the neural network were really 60 million knobs and
those knobs can be arbitrary values and how do you set the 60 million weights so that the network gives you
the correct predictions and so deep learning is is a class of is a way of of
training this neural network and finding a good setting of these 60 million numbers um
and so roughly uh the neural network sort of looks at the image gives you a prediction and then you measure the
error it's like okay you said this is a cat but actually this is a dog and then and there's a mathematical procedure for
tuning uh the strengths so that the neural network adapts itself to agree with you
and so deep learning is is basically a different software programming paradigm where we specify what we want
and then we use sort of mathematics and algorithms to tune the system to give you what you want
and there's some design that goes into the neural network architecture and how do you wire everything up but then there's
also a huge amount of design and effort spent on the data sets themselves and curating them and
you know because those data sets are now your constraints on the behavior that you are asking from the system
so it's a very different way of approaching problems that was not there before everything used to be written by
person now we just write the specification and we write a rough layout of the algorithm but it's a it's
what i refer to as fill in the blanks programming because we sort of lay out an architecture and a rough layout of the
net but there's a huge amount of blanks which are the weights and the knobs and those are set now during the training of