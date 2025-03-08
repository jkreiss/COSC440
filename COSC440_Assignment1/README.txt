# Answer the conceptual questions here
Q1: Is there anything we need to know to get your code to work? If you did not get your code working or handed in an incomplete solution please let us know what you did complete (0-4 sentences)
    - Should work no problem
Q2: Why do we normalize our pixel values between 0-1? (1-3 sentences)
    - Rather than having large and often times harder to work with range of data, a standardized set of data (usually between 0-1) helps improve interoperability, and reduce complexity.
Q3: Why do we use a bias vector in our forward pass? (1-3 sentences)
    - Bias is added to help improve performance by more accurately fitting data, reducing the chance of getting stuck at a local extrema, and learn the correct shift needed to predict a wider range of data.
Q4: Why do we separate the functions for the gradient descent update from the calculation of the gradient in back propagation? (2-4 sentences)
    - The separation of the calculation and updating of values helps increase modularity. This allows for flexibility in the changing the functions.
Q5: What are some qualities of MNIST that make it a “good” dataset for a classification problem? (2-3 sentences)
    - The MNIST data set is relatively small, very uniform, and well labeled. It is also confined to 0-9 making networks deciphering it less complex.
Q6: Suppose you are an administrator of the NZ Health Service (CDHB or similar). What positive and/or negative effects would result from deploying an MNIST-trained neural network to recognize numerical codes on forms that are completed by hand by a patient when arriving for a health service appointment? (2-4 sentences)
    - Positive, it would reduce the amount of work needed to process forms. Negative, it is on the smaller side for a dataset to train a neural net so it would not be uncommon for it to make mistakes. Another downside would be the preprocessing work for the model to be effective
