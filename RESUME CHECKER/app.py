import pickle
import re
import nltk
import streamlit as st

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load classifier and TF-IDF vectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


# Function to clean resume text
def cleanResume(txt):
    # Remove URLs
    cleanTxt = re.sub(r'http\S+|www\S+', '', txt)
    # Replace RT and CC
    cleanTxt = re.sub(r'\bRT\b|\bCC\b', ' ', cleanTxt)
    # Remove mentions (e.g. @username)
    cleanTxt = re.sub(r'@\S+', ' ', cleanTxt)
    # Remove hashtags (e.g. #hashtag)
    cleanTxt = re.sub(r'#\S+', ' ', cleanTxt)
    # Remove special characters, keeping spaces (using a raw string to avoid escape issues)
    cleanTxt = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', ' ', cleanTxt)
    # Remove non-ASCII characters
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    # Replace multiple spaces with a single space
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt).strip()
    return cleanTxt


# Web app function
def main():
    st.title("!ScreenMyResume!")

    # Upload file option
    upload_file = st.file_uploader('Upload Resume ', type=['txt', 'pdf'])

    if upload_file is not None:
        try:
            # Read uploaded file
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        # Clean resume text
        cleaned_resume = cleanResume(resume_text)
        cleaned_resume = tfidf.transform([cleaned_resume])

        # Predict category
        prediction_id = clf.predict(cleaned_resume)[0]

        # Define category mappings
        category_mapping = {
            6: 'Data Science',
            12: 'HR',
            0: 'Advocate',
            1: 'Arts',
            24: 'Web Designing',
            16: 'Mechanical Engineer',
            22: 'Sales',
            14: 'Health and fitness',
            5: 'Civil Engineer',
            15: 'Java Developer',
            4: 'Business Analyst',
            21: 'SAP Developer',
            2: 'Automation Testing',
            11: 'Electrical Engineering',
            18: 'Operations Manager',
            20: 'Python Developer',
            8: 'DevOps Engineer',
            17: 'Network Security Engineer',
            19: 'PMO',
            7: 'Database',
            13: 'Hadoop',
            10: 'ETL Developer',
            9: 'DotNet Developer',
            3: 'Blockchain',
            23: 'Testing'
        }

        # Get predicted category name
        category_name = category_mapping.get(prediction_id, "Unknown")

        # Display predicted category
        st.subheader(f"Predicted Category: {category_name}")

        st.subheader("For more info, contact Abhay!")


# Run the app
if __name__ == '__main__':
    main()
