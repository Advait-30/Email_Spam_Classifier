# 📧 Email Spam Classifier using Sentence Transformers + Naive Bayes

This project implements an email spam classifier that combines the semantic power of **LLM embeddings** (via Sentence Transformers) with the simplicity and speed of a **Naive Bayes classifier**.

Instead of relying on just raw word counts (Bag of Words / TF-IDF), the classifier uses semantic embeddings from a pretrained transformer model to capture meaning, tone, and context in email text — making it more robust against spam tricks.

---

## 🚀 Features

- Uses **all-MiniLM-L6-v2 (Sentence Transformers)** for text embeddings
- Lightweight **Naive Bayes classifier** for classification
- Supports **spam / ham prediction** with confidence score
- Evaluates performance with accuracy, precision, recall, F1, and confusion matrix
- Example predictions included
- Modular, class-based design (`EmailSpamClassifier`)

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/advait-30/Email_Spam_Classifier.git
cd Email_Spam_Classifier
pip install -r requirements.txt

requirements.txt:
numpy
pandas
scikit-learn
tqdm
sentence-transformers

## SMS Spam Collection v.1

1. DESCRIPTION

---

The SMS Spam Collection v.1 (hereafter the corpus) is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

## 1.1. Compilation

This corpus has been collected from free or free for research sources at the Web:

- A collection of between 425 SMS spam messages extracted manually from the Grumbletext Web site. This is a UK forum in which cell phone users make public claims about SMS spam messages, most of them without reporting the very spam message received. The identification of the text of spam messages in the claims is a very hard and time-consuming task, and it involved carefully scanning hundreds of web pages. The Grumbletext Web site is: http://www.grumbletext.co.uk/
- A list of 450 SMS ham messages collected from Caroline Tag's PhD Theses available at http://etheses.bham.ac.uk/253/1/Tagg09PhD.pdf
- A subset of 3,375 SMS ham messages of the NUS SMS Corpus (NSC), which is a corpus of about 10,000 legitimate messages collected for research at the Department of Computer Science at the National University of Singapore. The messages largely originate from Singaporeans and mostly from students attending the University. These messages were collected from volunteers who were made aware that their contributions were going to be made publicly available. The NUS SMS Corpus is avalaible at: http://www.comp.nus.edu.sg/~rpnlpir/downloads/corpora/smsCorpus/
- The amount of 1,002 SMS ham messages and 322 spam messages extracted from the SMS Spam Corpus v.0.1 Big created by Jos� Mar�a G�mez Hidalgo and public available at: http://www.esp.uem.es/jmgomez/smsspamcorpus/

  1.2. Statistics

---

There is one collection:

- The SMS Spam Collection v.1 (text file: smsspamcollection) has a total of 4,827 SMS legitimate messages (86.6%) and a total of 747 (13.4%) spam messages.

  1.3. Format

---

The files contain one message per line. Each line is composed by two columns: one with label (ham or spam) and other with the raw text. Here are some examples:

ham What you doing?how are you?
ham Ok lar... Joking wif u oni...
ham dun say so early hor... U c already then say...
ham MY NO. IN LUTON 0125698789 RING ME IF UR AROUND! H\*
ham Siva is in hostel aha:-.
ham Cos i was out shopping wif darren jus now n i called him 2 ask wat present he wan lor. Then he started guessing who i was wif n he finally guessed darren lor.
spam FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop
spam Sunshine Quiz! Win a super Sony DVD recorder if you canname the capital of Australia? Text MQUIZ to 82277. B
spam URGENT! Your Mobile No 07808726822 was awarded a L2,000 Bonus Caller Prize on 02/09/03! This is our 2nd attempt to contact YOU! Call 0871-872-9758 BOX95QU

Note: messages are not chronologically sorted.

📂 Dataset

The script automatically downloads the UCI SMS Spam Collection dataset.
• ~5,500 labeled SMS messages
• Labels: ham (0) or spam (1)

⸻

▶️ Usage

Run the script directly:
python spam_classifier.py

This will: 1. Download and preprocess the dataset 2. Generate embeddings with Sentence Transformers 3. Train a Naive Bayes classifier 4. Print performance metrics 5. Run example predictions

📊 Example Output
Model Performance Metrics:
Accuracy: 0.9746
Precision: 0.9565
Recall:    0.9403
F1 Score:  0.9483

Confusion Matrix:
[[965   5]
 [ 12 142]]

Example Predictions:

Text: URGENT! You have won a free vacation. Click here to claim now!
Prediction: spam (confidence: 0.9812)

Text: Hi Mom, what time should I come over for dinner tonight?
Prediction: ham (confidence: 0.9976)

📦 Project Structure:
Email_Spam_Classifier/
│── spam_classifier.py   # Main script with EmailSpamClassifier class
│── README.md            # Documentation
│── requirements.txt     # Dependencies
│── SMSSpamCollection    # Dataset (auto-downloaded)

📜 License

MIT License. Free to use and modify.
```
