# COMP5046 Assignment 1 (20 marks)
<b>Assingment 1 is an individual assessment</b>

<h2>Submission Due: April 17th, 2022 (Sunday, 11:59PM)</h2>
<i>We originally planned to have the due date as 14th of April, but changed to the 17th as it would be nicer for students who work on weekdays.</i>
<br><br>


<b>[XXX] = Lecture/Lab Reference</b><br/>
<b>(Justify your decision) =</b> You are required to justify your final decision/selection with the empirical evidence by testing the techniques in the section 4. Evaluation.<br/>
<b>(Explain the performance) =</b> Please explain the trend of performance, and the reason (or your opinion) why the trends show as they do.
<br>
<br>
<br>




<h2>Topic: Personality Type Classification using Recurrent Neural Networks</h2>
<p><b>In this assignment 1, we will focus on developing a personality type classification model using Word Embedding models and Recurrent Neural Networks (RNN). </b><br/>
We use the personality type classification <b>[Lecture5]</b>, Myers Briggs Type Indicator (or MBTI for short), which divides people into 16 distinct personality types across 4 axes: 1) Introversion (I) – Extroversion (E), 2) Intuition (N) – Sensing (S), 3) Thinking (T) – Feeling (F), and 4) Judging (J) – Perceiving (P).<br/>

In this assignment, we will focus on classifying only a <b>Thinking (T) - Feeling (F)</b> aspect from the 4 axes. <br/><br/>
<i>For your information, the detailed information for each implementation step is specified in the following sections. Note that lectures and lab exercises would be a good starting point for the assignment. The useful lectures and lab exercises are specified in each section.</i></p>

<br/>
<hr>
<br/>



<h2>1. Data Preprocessing</h2>
<p>In this assignment, you are to use the <b>(MBTI) Myers-Briggs Personality Type Dataset</b>, which includes a large number of people's MBTI types and content written by them. This data was originally collected through the <a href="https://www.personalitycafe.com/">PersonalityCafe forum</a>, as it provides a large selection of people and their MBTI personality type, as well as samples of their writing.<br/>

The Assignment 1 dataset contains forum post labelled with their associated binary personality aspect, <b>T(Thinking) or F(Feeling)</b> labels. Both the training and testing sets are provided in the form of csv files (training_data.csv, testing_data.csv) and can be downloaded from the Google Drive using the provided code in the <b><a href="https://colab.research.google.com/drive/1ReALJEgdJ2tVe-ABznB3ijErR-7E2gZ7?usp=sharing">Assignment 1 Template ipynb</a></b>. Note that we have 7808 rows of training data and 867 rows of testing data. The label(class) is in binary, T or F.
<p>
In this Data Preprocessing section, you are required to complete the following section in the required format:</br>
<ol>
  <li><b>URL removal</b>: You are asked to remove the URL from the post. You are asked to compare, by experimental results (in Section 4.2), when you remove the URL from the post versus keeping the URL. Which will you use? <b>(Section 4.2., Justify your decision)</b>
  <li><b>Preprocess data</b>: You are asked to pre-process the training set by integrating several text pre-processing techniques <b><i>[Lab5]</i></b> (e.g. tokenisation, removing numbers, converting to lowercase, removing stop words, stemming, etc.). You should test and justify the reason why you apply the specific preprocessing techniques based on the test result in section 4.2 <b>(Section 4.2., Justify your decision)</b>
  </li>
 </ol>
</p>


<br/>
<hr>
<br/>


<h2>2. Input Representation</h2>
<p>In this section, you are to implement three input representation components, including 1) Word Embedding Construction Module, 2) Pretrained Word Embedding Module, and 3) Input Concatenation Module. <i>For training, you are free to choose hyperparameters <b><i>[Lab2,Lab4,Lab5]</i></b> (e.g. dimension of embeddings, learning rate, epochs, etc.).</i></p>

<img src="https://github.com/usydnlp/COMP5046NLP_2022/blob/main/architecture.png" width="500px"/>
<p>The detailed information about model architecture can be found in the <b><i>[Lecture5]</i></b></p>

<h3>1)Word Embedding Construction</h3>
First, you are asked to build a word embedding model (for representing word vectors, such as word2vec-CBOW, word2vec-Skip gram, fastText, and Glove) for the input embedding of your sequence model <b><i>[Lab2]</i></b>. Note that we used one-hot vectors as inputs for the sequence model <i>in the Lab3 and Lab4</i>. You are required to complete the following sections in the format:
<ul>
  <li><b>Preprocess data for word embeddings</b>: You are to use and preprocess MBTI dataset (the one provided in the <i>Section 1</i>) for training word embeddings  <b><i>[Lab2]</i></b>. This can be different from the preprocessing technique that you used in Section 1. You can use both the training and testing datasets in order to train the word embedding.</li>
  
  <li><b>Build training model for word embeddings</b>: You are to build a training model for word embeddings. You are required to articulate the hyperparameters <b><i>[Lab2]</i></b> you choose (dimension of embeddings and window size) in Section 4.1. Note that any word embeddings model <b><i>[Lab2]</i></b> (e.g. word2vec-CBOW, word2vec-Skip gram, fasttext, glove) can be applied. <b>(Section 4.1. and Section 4.3., Justify your decision)</b></li>
</ul>


<h3>2)Pretrained Word Embedding</h3>
<p>You are asked to extract and apply the pretrained word embedding. <a href="https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models">Gensim</a> provides several pretrained word embeddings, you can find those in the <a href="https://github.com/RaRe-Technologies/gensim-data#models"><b>gensim github</b></a>. You can select the pretrained word embedding that would be useful for the assignment 1 task, personality type classification.<b>(Section 4.3., Justify your decision)</b><br/></p>


<h3>3)Input Concatenation</h3>
<p>You are asked to concatenate the trained word embedding and pretrained word embedding,<b><i>[Lab5]</i></b> and apply to the sequence model in Section 3 <b>(Section 4.3., Justify your decision)</b></p>


<br/>
<hr>
<br/>

<h2>3. Model Implementation</h2>
<p>Finally, you are to build the Many-to-One (N to 1) Sequence model in order to classify the label (T or F). Note that your model should be the best model selected from the evaluation.  <b><i>[Lab4]</i></b> <b>(Section 4.4., Justify your decision)</b>. You are required to implement the following functions:
<ul>
  <li>Build Sequence Model: You are to build the bi-directional sequence model (e.g. Bi-RNN, Bi-LSTM, or Bi-GRU)</li>
  <li>Train Sequence Model: You are to train the model. Note that it will not be marked if you do not display the Training Loss and the Number of Epochs in the Assignment 1 ipynb.</li>
</ul>

<h5>Note that it will not be marked if you do not display the Training Loss and the Number of Epochs in the Assignment 1 ipynb.</h5>

</p>

<br/>
<hr>
<br/>

<h2>4. Evaluation and Documentation <b>(17 marks)</b></h2>
You need to justify your decision and explain the pattern by testing the performance of your implementation.
The sample result visualisation can be found in the <b> [Lecture 5]</b>

<h3>1)Word Embedding Evaluation<b>(4 marks)</b></h3>
<p><b>[related to Section 2.1]</b><br/>
You are to conduct the Intrinsic Evaluation <b><i>[Lecture3]</i></b>. You are required to apply Semantic-Syntactic word relationship tests for understanding of a wide variety of relationships. The example code is provided <a href="https://colab.research.google.com/drive/1VdNkQpeI6iLPHeTsGe6sdHQFcGyV1Kmi?usp=sharing">here - Word Embedding Intrinsic Evaluation</a> (This is discussed and explained in the <b><i>[Lecture5 Recording]</i></b> ). You also are to visualise the result (the example can be found in the Table 2 and Figure 2 from the <a href="https://nlp.stanford.edu/pubs/glove.pdf">Original GloVe Paper</a>).<br/>
You MUST:
<ul>
  <li>try at least three different dimensions of your word embedding.</li>
  <li>try at least two window sizes (e.g. five)</li>
  <li>explain the trend of performance, and the reason (or your opinion) why the trends show like they do
</ul>
<b>NOTE: You would not receive any mark in this section 4.1, if you do not implement the section 2.1. Word Embedding Construction. The section 2.1. should contain the best word embedding training model based on this evaluation.</b> However, if you successfully complete only the section 2.1 but this section 4.1, you will receive only 1 mark out of 4 marks. 
</p>
<br/>

<h3>2)Performance - Data Preprocessing (4 marks)</h3>
<p><b>[related to Section 1.1 and 1.2]</b><br/>
You are to conduct the personality classification performance evaluation (with testing dataset) by using different preprocessing techniques. <br/>
You are to represent the f1 <b><i>[Lab4 and Lecture 5]</i></b> of your model in the table <b>(Explain the performance)</b></li>
You MUST:
<ul>
  <li>try with URL/without URL.</li>
  <li>try two different data preprocessing combinations</li>
  <li>explain the trend of performance, and the reason (or your opinion) why the trends show like they do
</ul>
<b>NOTE: You would not receive any mark in this section 4.2, if you do not implement the section 1.1. URL Removal and 1.2. Process data. The section 1.2. should contain the best data preprocessing techniques based on this evaluation.</b> However, if you successfully complete only the section 1.1 (0.5 marks) and 1.2 (0.5 marks) but this section 4.2, you will receive only 1 mark out of 4 marks. 
</p>
<b><i>The sample visualisation can be found in the Lecture 5. </i></b>
<br/>

<h3>3)Performance Evaluation - Different Input (4 marks)</h3>
<p><b>[related to Section 2.1, 2.2, and 2.3]</b><br/>
You are conduct the personality classification performance evaluation (with testing dataset) by using different input. <br/>
You are to represent the f1 <b><i>[Lab4 and Lecture 5]</i></b> of your model in the table <b>(Explain the performance)</b></li>
You MUST:
<ul>
  <li>try at least two word vectors training models.</li>
  <li>try at least two pretrained embeddings (from gensim)</li>
  <li>try at least two input concatenation, e.g. word2vec-CBOW(trained by you) + glove-twitter-100 (gensim)</li>
  <li>explain the trend of performance, and the reason (or your opinion) why the trends show like they do</li>
</ul>
<b>NOTE: You would not receive any mark in this section 4.3, if you do not implement the section 2.1, 2.2, 2.3. The section 2.1, 2.2, 2.3 should contain the best word embedding techniques based on this evaluation.</b> However, if you successfully complete only the section 2.1 (0.5 marks), 2.2 (0.5 marks), 2.3 (0.5 marks) but this section 4.3, you will receive only 1.5 mark out of 4 marks. 
</p>
<b><i>The sample visualisation can be found in the Lecture 5. </i></b>
<br/>

<h3>4)Performance Evaluation - Different Sequence Models (3 marks)</h3>
<p><b>[related to Section 3.1 and 3.2]</b><br/>
You are conduct the personality classification performance evaluation (with testing dataset) by using different sequence models. <br/>
You are to represent the f1 <b><i>[Lab4 and Lecture 5]</i></b> of your model in the table <b>(Explain the performance)</b><br/>
You MUST:
<ul>
  <li>use bi-directional recurrent models.</li>
  <li>try at least two bi-directional models</li>
  <li>explain the trend of performance, and the reason (or your opinion) why the trends show like they do</li>
</ul>
<b>NOTE: You would not receive any mark in this section 4.4, if you do not implement the section 3.1, and 3.2. The section 3.1 and 3.2 should contain the best data preprocessing techniques based on this evaluation.</b> However, if you successfully complete only the section 3.1 (0.5 marks), 3.2 (0.5 marks) but this section 4.4, you will receive only 1 mark out of 3 marks. </b>
</p>
<b><i>The sample visualisation can be found in the Lecture 5. </i></b>
<br/>

<h3>5)Performance Evaluation - Hyperparameter Testing (2 marks)</h3>
<p><b>[related to Section 1,2, and 3]</b><br/>
<p>You are to provide the line graph, which shows the hyperparameter testing (with the testing dataset) and explain the optimal number of epochs based on the learning rate you choose. You can have multiple graphs with different learning rates. In the graph, the x-axis would be # of epoch and the y-axis would be the f1.  <b>(Explain the performance)</b></p>
<b>NOTE: You would not receive any mark in this section 4.5, if you do not implement the section 1,2, and 3. </b>
<b><i>The sample visualisation can be found in the Lecture 5. </i></b>
<br/>
<hr>
<br/>

  
  

<h2>5. User Interface (3 marks)</h2>
<p>Based on the evaluation, you are required to design a user interface so that user can input a textual sentence via the colab form fields user interface to get the classification result from your best trained model.<br/>
<b>NOTE: In this section, you need to develop the user interface that can be runnable as a stand-alone testing program. You MUST put all code parts of your best model (including data preprocessing techniques, word embedding model, and sequence model). You would not receive any mark in this section 5, if you do not implement any classification models to classify the personality type in Section 1,2, and 3. <i>[Lab 5]</i></b>
</p>
<img src="https://github.com/usydnlp/COMP5046NLP_2022/blob/main/sample%20result.png" width="300px"/>

<br/>
<hr>
<br/>

  
  

<h2>Submission Instruction</h2>
<p>Submit an ipynb file - (file name: UNIKEY_COMP5046_Ass1_2022.ipynb) that contains all above sections(Section 1,2,3, and 4). <b>NOTE that you need to change the 'UNIKEY' to your unikey (e.g. chan0022)</b><br/>
The ipynb template can be found in the <a href="https://colab.research.google.com/drive/1ReALJEgdJ2tVe-ABznB3ijErR-7E2gZ7?usp=sharing">Assignment 1 template</a></p>

<b>NOTE: You MUST make sure yourself whether it can be runnable in the colab since your marker will actually run your code. If your code is not runnable, we cannot give you any mark in the specific section. Our markers will not modify your code.</b>

<br/>
<hr>
<br/>


<h2>FAQ</h2>
<p>
  <b>Question:</b> What do I need to write in the justification or explanation? How much do I need to articulate?<br/>
  <b>Answer:</b> As you can see the 'Read me' section in the ipynb Assingment 1 template, visualizing the comparison of different testing results is a good way to justify your decision. Explain the trends based on your understanding. You can find another way (other than comparing different models) as well - like showing any theoretical comparison for example</p>

<p>
  <b>Question:</b> Is there any marking scheme/marking criteria available for assignment 1?<br/>
  <b>Answer:</b> The assignment specification is extremely detailed. The marking will be directly conducted based on the specification.
</p>

<p>
  <b>Question:</b> My Word Embedding/ Personality Classification performs really bad (Low F1). What did i do wrong?<br/>
  <b>Answer:</b> Please don't worry about the low performance as our training dataset is very small and your model is very basic deep learning model. Of course, there is something wrong with your code if it comes out below 10% (0.1). 
</p>

<p>
  <b>Question:</b> Do I need to use only the MBTI dataset for training the word embedding?<br/>
  <b>Answer:</b> No, as mentioned in the lecture 5 (assignment 1 specification description), you can use any dataset (including TED, Google News) or MBTI dataset for training your word embedding. Word embedding is just for training the word meaning space so you can use any data. 
  Note: Training word embedding is different from training the Bi-RNN prediction model for personality type classification. For the Bi-RNN personality type classification model training, you MUST use only the training dataset (from the dataset that we provided in the assignment 1 template)
</p>


<p>
  <b>Question:</b> Do I need to save the word embedding model or RNN models?<br/>
  <b>Answer:</b> We highly recommend you to save your best word embedding model, and load it when you use it in your code. Otherwise, it is sometimes removed and overwrite all your code so that you need to re-run the whole code again.
</p>
  

<h5>If you have any question, please come to LiveQA and post it in the Edstem anytime!</h5>
