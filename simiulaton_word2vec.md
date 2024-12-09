# what’s happening during the training process of Word2Vec.

---

### **1. Key Terms and Concepts**
---

#### **Center Word and Context Word**
- **Center Word**: The target word whose relationship with its neighbors we want to learn. For example, in the sentence "The cat sat on the mat," if "cat" is the center word, its context words could be ["The," "sat"].
- **Context Words**: Words surrounding the center word within a certain "window size" (e.g., 2 words to the left and right).

---

#### **Forward Pass**
1. **Similarity Between Embeddings**:
   - **Word Embedding**: A vector that represents a word in a continuous vector space (e.g., `[1.2, -0.5, 0.3]` for "cat").
   - During the forward pass, the model computes how similar the center word’s embedding is to all other word embeddings in the vocabulary.

2. **Softmax Function**:
   - Converts raw similarity scores (which could be any numbers) into probabilities that sum to 1.
   - This helps interpret the similarity as a probability of a word being the correct context word.

   Example:
   - Raw Scores: `[2.1, 1.3, 0.9]` → Softmax Probabilities: `[0.6, 0.3, 0.1]`.

---

#### **Loss Calculation**
- **One-Hot Encoding**: Represents the true context word as a vector where one position is `1` (indicating the correct word) and all others are `0`.

   Example:
   - Vocabulary = ["cat", "dog", "mat"].
   - True Context Word = "dog".
   - One-Hot Vector = `[0, 1, 0]`.

- **Cross-Entropy Loss**:
   - Compares the predicted probabilities from softmax to the true one-hot vector.
   - Measures how well the model predicted the correct context word.
   - Lower loss = better performance.

   Example:
   - Predicted Probabilities: `[0.6, 0.3, 0.1]`.
   - True Context Word = "dog" → One-Hot Vector = `[0, 1, 0]`.
   - Loss penalizes the model because it gave a higher probability to "cat" instead of "dog."

---

#### **Backpropagation**
1. **What is it?**
   - A process where the model learns from its mistakes by adjusting its parameters (embeddings) to minimize the loss.

2. **How?**
   - Gradients (small changes) are calculated for the embeddings of both the center word and the context words, indicating how they should be updated.

---

#### **Parameter Updates**
- **Optimization Algorithm (e.g., SGD)**:
   - Stochastic Gradient Descent (SGD) adjusts the embeddings slightly in the direction that reduces the loss.

   Example:
   - Current Embedding for "cat": `[1.2, -0.5, 0.3]`.
   - After Update: `[1.15, -0.45, 0.28]` (moved slightly to better fit the data).

---

### **2. Summary of Steps**
1. Take a **center word** (e.g., "cat") and a **context word** (e.g., "sat").
2. Compute the similarity between "cat" and all words in the vocabulary.
3. Use **softmax** to convert these similarities into probabilities.
4. Compare the predicted probabilities to the true one-hot vector for "sat" using **cross-entropy loss**.
5. Use **backpropagation** to adjust the embeddings for "cat" and "sat" to reduce the loss.
6. Repeat for all center-context pairs in the dataset until embeddings represent meaningful word relationships.

## loss function, random weights, ughhhhhh????

at the beginning, the weights would be random, so when you are calculating for the first time, the similarityies, what's the loss funciotn what's the label, to compare, what do yo check




At the beginning of training, the word embeddings (weights) are **randomly initialized**, which means they don't yet encode any meaningful relationships. Let's walk through the steps of the first iteration and clarify what happens during **loss calculation** and **comparison**:

---

### **1. What are the initial similarities?**
- Since the embeddings are random, the computed similarities (dot products between word embeddings) will also be random.
- Example:
  - Random embedding for "cat" (center word): `[0.1, -0.3, 0.5]`.
  - Random embedding for "sat" (context word): `[-0.4, 0.2, 0.6]`.
  - ![image](https://github.com/user-attachments/assets/044f251e-f139-4c08-adec-16e85e5e7369)

---

### **2. What’s the label (true output)?**
- For Word2Vec (Skip-gram):
  - **Center word**: "cat".
  - **Context word**: "sat".
  - Label = the true context word is represented as a **one-hot vector**.

  Example:
  - Vocabulary: ["cat", "dog", "sat", "mat"].
  - True context word: "sat".
  - One-hot vector for "sat" = `[0, 0, 1, 0]`.

---

### **3. How do you calculate loss?**
- **Predicted Output**:
  - Compute the similarity (dot product) between the center word's embedding ("cat") and every word's embedding in the vocabulary.
  - Apply the **softmax function** to get probabilities.

  Example:
  ![image](https://github.com/user-attachments/assets/faea50f2-08d0-4875-b97a-388228e8232f)


- **Loss Function**:
  - Use **cross-entropy loss** to compare the predicted probabilities to the true one-hot vector.
  - Formula:
    ![image](https://github.com/user-attachments/assets/ecfb4485-c96b-4407-8553-7ae73dbad9b9)


---

### **4. What are you checking?**
- The loss tells you how far the predicted probabilities are from the true labels. In the beginning, the loss is **high** because the embeddings are random.

---

### **5. Backpropagation**
- Gradients are computed to adjust the embeddings of the **center word** ("cat") and the **context word** ("sat") to reduce the loss.
- For the first iteration:
  - The embeddings move slightly in the direction that makes the predicted probabilities better match the true one-hot label.
  - Over time, as embeddings update, the loss decreases, and the embeddings begin to encode meaningful relationships.

---

### **Summary**
1. **Similarity Calculation**: Use random embeddings initially to compute similarities.
2. **Softmax**: Converts similarities into probabilities.
3. **Labels**: The true context word is represented as a one-hot vector.
4. **Loss Function**: Cross-entropy loss compares predicted probabilities to the one-hot vector.
5. **Goal**: Adjust embeddings to minimize the loss over many iterations.

## concrete example please!!!


again, a little off at the first step, give me a concrete example, you took a sentance and now you have the randomised embeding,s waht do you compare


Let’s take a **concrete example** from the start, showing how embeddings are initialized, how they’re used to calculate probabilities, and what we compare during training. 

---

### **Example Sentence**  
Sentence: `"The cat sat on the mat"`

- Suppose the vocabulary size is **6 words**: `["The", "cat", "sat", "on", "the", "mat"]`.
- The task is to predict **context words** for a given **center word** using a window size of 1 (one word to the left and right).

---

### **1. Center and Context Words**
For training:
- Center word = `"cat"`.
- Context words = `["The", "sat"]`.

---

### **2. Randomized Embeddings**
Each word in the vocabulary is initialized with a **random embedding vector** of a fixed dimension (e.g., 3-dimensional vectors for simplicity).

Example:  
- `"The"` → `[0.2, -0.1, 0.5]`  
- `"cat"` → `[0.6, 0.8, -0.3]`  
- `"sat"` → `[-0.4, 0.2, 0.9]`  
- `"on"` → `[0.1, -0.5, 0.7]`  
- `"the"` → `[0.3, 0.6, -0.2]`  
- `"mat"` → `[0.8, -0.3, -0.1]`

---

### **3. Forward Pass**  
#### Step 1: Calculate Similarity Scores  
To predict context words for `"cat"`, compute the **dot product** between `"cat"`’s embedding and the embeddings of all words in the vocabulary.

![image](https://github.com/user-attachments/assets/1c50e6f9-808d-4fb1-8a65-2ea8d264f2f9)
 
These scores represent how similar each word is to `"cat"` in this random initialization.

---

#### Step 2: Apply Softmax  
Convert these similarity scores into probabilities using the **softmax function**:

![image](https://github.com/user-attachments/assets/fa08b1ad-d59a-4cd4-8a7f-e53229160bc1)


---

### **4. Labels and Comparison**  
#### True Labels (One-Hot Encoding)  
For the first context word `"The"`:
- Vocabulary index: 0 (based on the vocabulary order).
- True label (one-hot vector):  

  [1, 0, 0, 0, 0, 0]
  

---

#### Loss Calculation (Cross-Entropy Loss)  
The loss measures the difference between the **predicted probabilities** and the **true labels**:

![image](https://github.com/user-attachments/assets/da6e2dea-8d4a-4723-9464-59bfec505d64)


---

### **5. Backpropagation and Updates**  
- Gradients are computed for the embeddings of:
  - The **center word** ("cat").
  - The **context word** ("The").
- These gradients tell how to adjust the embeddings to make the predicted probabilities closer to the true labels in the next iteration.

For example:
- Embedding for `"cat"` changes from `[0.6, 0.8, -0.3]` to `[0.59, 0.78, -0.29]`.
- Embedding for `"The"` changes slightly as well.

---

### **Repeat for All Center-Context Pairs**
The process is repeated for all center-context pairs in the corpus, gradually improving the embeddings so that words with similar contexts have closer embeddings.

##cross-emtropy loss, backpropagation and optmization??

### **1. Cross-Entropy Loss**

#### **Purpose**  
![image](https://github.com/user-attachments/assets/0e9f815d-4cd2-4c21-ab19-26e8faa1eb0f)

---

#### **Example**  

![image](https://github.com/user-attachments/assets/26978725-5a77-48d9-a97d-40ed3bd4a91e)


For multiple center-context pairs, average the losses to compute the overall loss for a batch or epoch.

---

### **2. Backpropagation**  

#### **Objective**
Adjust embeddings to minimize the cross-entropy loss. Backpropagation computes **gradients** for each parameter (embedding values) so they can be updated.

---

#### **Steps**  

1. **Compute Gradient of Loss w.r.t. Predicted Probabilities**:  
![image](https://github.com/user-attachments/assets/f7cfb58b-a140-418e-9ffd-8b48ad485260)

---

2. **Propagate Gradients to Center and Context Word Embeddings**:  
- **Center word** (`"cat"`): Embedding gets updated based on its contribution to the incorrect prediction.  
- **Context word** (`"The"`): Embedding gets updated to make it more likely as a context word for `"cat"`.

The gradient for embeddings is derived from the gradient of the softmax and the dot product used in similarity.

---

3. **Compute Gradients for All Parameters**:  
Gradients are calculated for:  
- Embeddings of the center word (`"cat"`)  
- Embeddings of the predicted words (entire vocabulary, but mostly `"The"`)  

---

### **3. Optimization**  

#### **Purpose**  
Use the computed gradients to **update embeddings**, reducing the loss over time. The most common optimization algorithm used is **Stochastic Gradient Descent (SGD)**.  

#### **SGD Update Rule**  
For each parameter (embedding value):  
New Value = Old Value - eta .Gradient
 

Where:  
- \(\eta\) is the learning rate (controls step size).  
- Gradient is from backpropagation.

---

#### **Example**  
- Embedding for `"cat"` (initial): `[0.6, 0.8, -0.3]`  
- Gradient for `"cat"` embedding: [-0.02, 0.03, -0.01]
- Learning rate: eta = 0.1

**Update**:  
![image](https://github.com/user-attachments/assets/22bd1338-9ca4-46c6-989e-61db50ce3cf7)


---

### **Summary of the Process**
1. Forward pass computes the predicted probabilities (\(\hat{y}\)).
2. Cross-entropy loss is calculated by comparing \(\hat{y}\) with the true labels (\(y\)).  
3. Backpropagation computes gradients for embeddings.  
4. Optimization (SGD) updates embeddings to reduce the loss.  

