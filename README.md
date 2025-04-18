BaselineAnalysis

This is a Baseline Survey Analysis Dashboard built for ResGov using Streamlit.
It provides interactive data visualizations and AI-generated insights from participant responses collected in the baseline survey.

---

Features
--------

- District-wise and question-wise analysis of knowledge-based responses (correct/incorrect)
- Open-ended and subjective response distribution
- Most frequently selected options per question
- AI-powered participant summary generation using Hugging Face’s Mistral model

---

Project Structure
-----------------

BaselineAnalysis/
|
├── app.py                  -> Streamlit app entry point
├── BaselineCombined.xlsx   -> Excel file with all survey data
├── requirements.txt        -> Python dependencies
└── README.txt              -> Project documentation

---

Setup Instructions
------------------

1. Clone the repository:

   git clone https://github.com/Banaibor-resgov/BaselineAnalysis.git
   cd BaselineAnalysis
2. Create and activate a virtual environment:

   python -m venv myenv
   myenv\Scripts\activate      (On Windows)
   source myenv/bin/activate   (On Mac/Linux)
3. Install dependencies:

   pip install -r requirements.txt
4. Ensure the file 'BaselineCombined.xlsx' is present in the same directory as 'app.py'.

---

Running the Application
-----------------------

To run the Streamlit dashboard locally:

   streamlit run app.py

This will open the dashboard in your default browser.

---

Requirements
------------

Core Python packages used:

- streamlit
- pandas
- matplotlib
- seaborn
- requests
- openpyxl

---

Contact
-------

Developed for ResGov
Maintainer email: banaibor@resgov.org
