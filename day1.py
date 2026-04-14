import torch


'''
Question 1: Basic shapes.

You have:
    x: shape (B,D)

If:
    B = 4
    D = 3

and x is a batch of 4 vectors of size 3,
what is the shape of x?

What does the first dimension mean?
What does the second dimension mean?
'''

### Question 1: Answer and reasoning:

# I'll impliment in torch for practice:

# dims
b=4
d=3

# build tensor x with dims
x = torch.randn(b,d)

# x: shape = (4, 3)
# dim 0: is the batch dimention, 4 vectors. 
# dim 1: is the feature dimention, each vector has a length of 3.








'''
Question 2: Add a linear layer.

You have:
    x: shape (B,D)
    W: shape (D,H)

You compute:
    y = x @ W

If:
    x has shape (4,3)
    W has shape (3,5)

what shape is y?

Explain which dimensions must match and why.
'''

### Question 2: Answer and reasoning:

b = 4 # batch
d = 3 # feature
h = 5 # hidden

x = torch.randn(b,d)
W = torch.randn(d,h)

# We compute a linear layer:
y = x @ W

# y: shape(4,5) or shape(b, h)
# The matching dimentions are the inner dims, d. 
# They must match because each row of x is a length-3 vector, and W expects an input of size 
    # 3 in order to transform it into a length-5 vecctor. 
# The output keeps b from x and h from W. 








'''
Question 3: Small RL shape question.

You have:
    logits: shape (B,A)

This means:
    B = batch size
    A = number of actions

If B = 6 and A = 2,
what does logits[0] represent?

What shape is logits[0]?
'''

### Question 3: Answer and reasoning:

logits = torch.randn(6,2)
print('logits', logits)
print('logits[0]', logits[0])
print('shape: ', logits[0].shape)
print('dims: ', logits[0].dim)

# logits has 6 batch elements
# In each batch element the model outputs 2 numbers, one per action

# logits[0] shape would be (2,) 
# indexing with 0 looks into the values of the first row removing the outer dimention. 








'''
Question 4: Basic broadcasting.

You have:
    x: shape (B,D)
    mask: shape (B,1)

You compute:
    x * mask

If x has shape (4,3) and mask has shape (4,1),
will broadcasting work?

If yes, explain how the (4,1) mask expands against (4,3).
'''

### Question 4: Answer and reasoning:

# In broadcasting the shapes are aligned from right to left.
# The second dimention of the mask is 1. This will be streched or copied accoess that axis to match the 3 columns of x.
# Then moving to the first dimention, they already match, no changes required. 
# Therefore mask shape: (4,1)->(4,3)
# Each row mask value, (1,) is applied accross all 3 features of x.






'''
Question 5: Very small debugging question.

You have:
    logits = torch.randn(8, 4)
    target = torch.randn(8)
    loss = torch.nn.functional.cross_entropy(logits, target)

What is wrong with target here?

Do not give a full fix yet.
Just explain what kind of tensor cross_entropy expects for target.
'''

### Question 5: Answer and reasoning:
# logits have 8 samples, and 4 classes.
# The shape requirements for cross entropy are correct. 
# Where the target is the correct shape of (8,). Where each element represents the correct class for each sample. 
# However, this issue is the datatypes. The target needs to be an integer as it represents a class index.


'''
Question 6: Basic indexing with sequences.

You have:
    x: shape (B,T,D)

If:
    B = 2
    T = 4
    D = 3

what does x[0] mean?

What shape is x[0]?
'''

### Question 6: Answer and reasoning:
# In x there are 2 batch elements. Therefore x[0] referes to the 0th batch, with size x[0]: shape(T,D)


'''
Question 7: Basic slicing with sequences.

You have:
    x: shape (B,T,D)

If x has shape (2,4,3),
what shape is x[:, 1]?

Explain what slice was taken.
'''

### Question 7: Answer and reasoning:


'''
Question 8: Flattening a 2D tensor.

You have:
    rewards: shape (B,T)

If rewards has shape (2,4),
what shape is rewards.reshape(-1)?

Explain what reshape(-1) is doing here.
'''

### Question 8: Answer and reasoning:


'''
Question 9: Reading action logits.

You have:
    logits: shape (B,T,A)

If logits has shape (2,4,5),
and the last dimension is actions,
what does the 5 mean?
'''

### Question 9: Answer and reasoning:


'''
Question 10: Broadcasting check.

You have:
    x: shape (B,T,D)
    mask: shape (B,T)

If x has shape (2,4,3) and mask has shape (2,4),
will x * mask work directly?

If not, explain why not.
'''

### Question 10: Answer and reasoning:
